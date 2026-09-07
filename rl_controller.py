"""
RL & Hybrid Controllers for Battery Fast Charging — Extended to 8C
===================================================================
Controllers included:

  BatteryRLController        — SAC agent (stable-baselines3)
  HybridCompensationController — MPC baseline + RL residual
  AdaptiveWeightHybrid       — RL-learned MPC/RL blend weight
  SafeHybridController       — Deterministic safe hybrid (no RL)

Changes vs. 4C version:
  - All controllers accept max_current up to 8C
  - NoTerminationWrapper also terminates on over-temperature
  - StrongRewardWrapper target_time default updated to 360 s (8C reference)
  - SafetyFirstWrapper plating_margin set to 15 mV (was implicit 0)
  - _create_eval_env creates a fresh env (not re-use of training env)
  - WeightNetwork and SimpleRLPolicy8C updated for 8C action space
  - HybridCompensationController max_compensation increased to 1.5C
  - All references to battery_environment.py use the 8C env class
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import os
import random
import warnings
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Union

import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Shared anode model (mirrors env exactly — see battery_environment.py)
# ---------------------------------------------------------------------------
def _anode_potential(soc: float, c_rate: float) -> float:
    ocp     = 0.48 * (1 - soc) + 0.08 * soc
    eta_kin = (-0.015 * c_rate if c_rate <= 4.0
               else -0.015 * 8.0 - 0.080 * (c_rate - 8.0))
    onset   = max(0.45, 0.65 - 0.03 * max(0.0, c_rate - 8.0))
    eta_d   = -0.025 * max(0.0, soc - onset)
    return ocp + eta_kin + eta_d


# ===========================================================================
# Environment wrappers
# ===========================================================================

class NoTerminationWrapper(gym.Wrapper):
    """
    Prevent early termination during exploration except on genuine failures.
    Only terminates when SoC target reached, temperature exceeded, or
    anode potential is severely negative (> −50 mV).
    """

    def __init__(self, env, target_soc: float = 0.8):
        super().__init__(env)
        self.target_soc = target_soc

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        soc   = float(obs[0])
        temp  = float(obs[1])
        anode = float(obs[2])

        # Re-define termination conditions
        terminated = (
            soc >= self.target_soc
            or temp > self.env.unwrapped.temp_limit_K
            or anode < -0.05   # allow mild plating during exploration
        )
        return obs, reward, terminated, truncated, info


class StrongRewardWrapper(gym.Wrapper):
    """
    Dense reward shaped for high-rate (up to 8C) fast charging.

    Reward components:
      current_reward   : +25·C   (encourage high current)
      progress_reward  : +80·SoC
      completion_bonus : 2000 + time_bonus  (fast completion)
      plating_penalty  : −20 per step with negative anode potential
      time_penalty     : −t/10
    """

    def __init__(self, env, target_time: float = 360.0):
        """target_time: reference time [s] for max bonus (default 360s ≈ 8C charge)"""
        super().__init__(env)
        self.target_time = target_time

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        soc              = float(obs[0])
        current          = float(action[0])
        time_elapsed     = float(info.get("time", 0))
        plating_detected = bool(info.get("plating_detected", False))
        anode            = float(info.get("anode_potential", 0.1))

        current_reward  = current * 25.0
        progress_reward = soc * 80.0

        if soc >= 0.8:
            time_bonus       = max(0.0, (self.target_time - time_elapsed) * 5.0)
            completion_bonus = 2000.0 + time_bonus
        else:
            completion_bonus = 0.0

        # Proportional plating penalty (graded by severity)
        if plating_detected:
            plating_penalty = -100.0
        elif anode < 0.01:
            plating_penalty = -50.0
        else:
            plating_penalty = 0.0

        time_penalty = -time_elapsed / 10.0
        reward = (current_reward + progress_reward + completion_bonus
                  + plating_penalty + time_penalty)

        return obs, reward, terminated, truncated, info


class SafetyFirstWrapper(gym.Wrapper):
    """
    Hard safety wrapper: terminal penalty and early stop on plating.
    Applies graded pre-plating penalties based on anode potential proximity.
    """

    def __init__(self, env, plating_penalty: float = -700.0,
                 plating_margin: float = 0.015):
        super().__init__(env)
        self.plating_penalty  = plating_penalty
        self.plating_margin   = plating_margin

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        anode = float(info.get("anode_potential", 0.1))
        soc   = float(obs[0])

        if info.get("plating_detected", False):
            return obs, self.plating_penalty, True, truncated, info

        # Graded pre-plating penalties
        if anode < self.plating_margin:
            reward = -160.0;  terminated = True
        elif anode < 0.025:
            reward = -90.0
        elif anode < 0.04:
            reward = -60.0
        elif anode < 0.05:
            reward = -3.0
        else:
            # Healthy-anode bonus (encourage staying well above margin)
            if anode > 0.07 and soc < 0.65:
                reward += 3.0
            reward += soc * 8.0

        return obs, reward, terminated, truncated, info


# ===========================================================================
# Pure RL Controller (SAC)
# ===========================================================================

class BatteryRLController:
    """SAC agent for 8C-capable battery fast charging."""

    def __init__(
        self,
        env,
        model_path:      Optional[str] = None,
        tensorboard_log: str = "./logs/",
    ):
        self.max_current = env.max_current
        self.dt          = env.dt
        self.target_soc  = env.target_soc
        self.original_env = env

        env_wrapped = NoTerminationWrapper(env, target_soc=self.target_soc)
        env_shaped  = StrongRewardWrapper(env_wrapped,
                                          target_time=self.target_soc * 3600.0 / self.max_current)

        monitor_dir = f"./monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.monitored_env = Monitor(env_shaped, monitor_dir)

        self.hyperparams = {
            "learning_rate": 3e-4,
            "buffer_size":   500_000,
            "batch_size":    256,
            "tau":           0.005,
            "gamma":         0.99,
            "train_freq":    2,
            "gradient_steps": 2,
            "policy_kwargs": dict(net_arch=[256, 256]),
        }

        if model_path and os.path.exists(model_path + ".zip"):
            self.model = SAC.load(model_path, env=self.monitored_env)
            print(f"Loaded RL model from {model_path}")
        else:
            self.model = SAC("MlpPolicy", self.monitored_env,
                             **self.hyperparams, verbose=0,
                             tensorboard_log=tensorboard_log)
            print(f"Created SAC agent (max {self.max_current}C)")

    def train(
        self,
        total_timesteps: int = 100_000,
        save_path:       str = "rl_8c_model",
        eval_freq:       int = 10_000,
    ):
        print(f"\nTraining SAC agent — {total_timesteps} steps, max {self.max_current}C")

        class ProgressCallback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)
                self.episodes = 0
            def _on_step(self):
                if self.locals.get("done", False):
                    self.episodes += 1
                    if self.episodes % 100 == 0:
                        info = self.locals.get("infos", [{}])[0]
                        print(f"  Ep {self.episodes}: "
                              f"t={info.get('time',0):.0f}s  "
                              f"SoC={info.get('soc',0):.2f}  "
                              f"plating={info.get('plating_detected',False)}")
                return True

        eval_env = self._make_eval_env()
        best_dir = f"./best_{save_path}"
        os.makedirs(best_dir, exist_ok=True)

        eval_cb = EvalCallback(eval_env, best_model_save_path=best_dir,
                               eval_freq=eval_freq, deterministic=True, verbose=1)
        self.model.learn(total_timesteps=total_timesteps,
                         callback=[ProgressCallback(), eval_cb])
        self.model.save(save_path)
        print(f"Model saved → {save_path}.zip")
        return self.model

    def _make_eval_env(self):
        from battery_environment import BatteryPlatingEnv
        base = BatteryPlatingEnv(max_current_C=self.max_current,
                                 dt=self.dt, target_soc=self.target_soc,
                                 analytic_only=True)
        return StrongRewardWrapper(base,
                                   target_time=self.target_soc * 3600.0 / self.max_current)

    def get_action(self, observation: np.ndarray, deterministic: bool = True) -> float:
        obs = observation.reshape(1, -1) if observation.ndim == 1 else observation
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return float(np.clip(action[0], 0.2, self.max_current))

    def evaluate(self, n_episodes: int = 10) -> Dict:
        print(f"\nEvaluating SAC agent ({n_episodes} episodes)")
        results = dict(plating_events=0, charging_times=[], final_soc=[])
        for ep in range(n_episodes):
            obs, _ = self.original_env.reset()
            done   = False
            while not done:
                a = self.get_action(obs)
                obs, _, terminated, truncated, info = self.original_env.step([a])
                done = terminated or truncated
            results["plating_events"] += int(info.get("plating_detected", False))
            results["charging_times"].append(info.get("time", 0))
            results["final_soc"].append(info.get("soc", 0))
            status = "⚠️" if info.get("plating_detected") else "✓"
            print(f"  Ep {ep+1:3d}: {status} "
                  f"t={info.get('time',0):.0f}s  SoC={info.get('soc',0):.3f}")
        rate = results["plating_events"] / n_episodes * 100
        print(f"Plating rate: {rate:.1f}%  |  "
              f"Avg time: {np.mean(results['charging_times']):.0f}s")
        return results


# ===========================================================================
# Hybrid: MPC baseline + RL compensation
# ===========================================================================

class HybridCompensationController:
    """
    SAFETY-FIRST hybrid for 8C battery.

    Architecture:
        u_final = clip(u_mpc + u_rl_compensation, 0.2, max_current)

    The RL agent learns only the residual; the MPC provides a
    physics-grounded baseline. Compensation is bounded adaptively
    based on anode proximity and SoC.
    """

    def __init__(
        self,
        env,
        mpc,
        rl_model_path:    Optional[str] = None,
        max_compensation: float = 1.5,    # ± C-rate (increased for 8C range)
        tensorboard_log:  str = "./logs/",
    ):
        self.env         = env
        self.mpc         = mpc
        self.max_comp    = max_compensation
        self.max_current = env.max_current
        self.dt          = env.dt
        self.target_soc  = env.target_soc
        self.original_env = env

        wrapped = SafetyFirstWrapper(env, plating_penalty=-500,
                                     plating_margin=0.015)
        monitor_dir = f"./monitor_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.monitored_env = Monitor(wrapped, monitor_dir)

        self.hyperparams = {
            "learning_rate": 3e-5,
            "buffer_size":   200_000,
            "batch_size":    128,
            "tau":           0.005,
            "gamma":         0.99,
            "train_freq":    4,
            "gradient_steps": 2,
            "policy_kwargs": dict(net_arch=[128, 128]),
        }

        if rl_model_path and os.path.exists(rl_model_path + ".zip"):
            self.rl_model = SAC.load(rl_model_path, env=self.monitored_env)
            print(f"Loaded hybrid RL model from {rl_model_path}")
        else:
            self.rl_model = SAC("MlpPolicy", self.monitored_env,
                                **self.hyperparams, verbose=0,
                                tensorboard_log=tensorboard_log)
            print(f"Created Hybrid Compensation controller (max {self.max_current}C, "
                  f"±{self.max_comp}C compensation)")

        self.comp_history: List[float] = []

    def get_action(
        self,
        observation: np.ndarray,
        use_rl:       bool = True,
        deterministic: bool = True,
    ) -> float:
        soc   = float(observation[0])
        temp  = float(observation[1])
        anode = float(observation[2]) if len(observation) > 2 else 0.1

        u_mpc = self.mpc.solve([soc, temp, anode])

        if not use_rl:
            return float(u_mpc)

        try:
            obs_r   = observation.reshape(1, -1) if observation.ndim == 1 else observation
            u_comp, _ = self.rl_model.predict(obs_r, deterministic=deterministic)
            u_comp  = float(u_comp[0])
        except Exception:
            u_comp = 0.0

        # Adaptive compensation ceiling based on safety margins
        if anode < 0.02:
            max_c = 0.05
        elif anode < 0.035:
            max_c = 0.15
        elif anode < 0.05:
            max_c = 0.30
        elif soc > 0.65:
            max_c = 0.50
        else:
            max_c = self.max_comp

        # Only allow upward compensation when well away from plating
        if anode < 0.05:
            u_comp = min(0.0, u_comp)

        u_comp = np.clip(u_comp, -max_c, max_c)

        # Temperature de-rating
        temp_c = temp - 273.15
        if temp_c > 60:
            u_comp *= 0.3
        elif temp_c > 50:
            u_comp *= 0.6
        elif temp_c > 45:
            u_comp *= 0.8

        u_final = float(np.clip(u_mpc + u_comp, 0.2, self.max_current))
        self.comp_history.append(u_comp)
        return u_final

    def train(
        self,
        total_timesteps: int = 60_000,
        save_path:       str = "hybrid_comp_model",
        eval_freq:       int = 8_000,
    ):
        print(f"\n{'='*70}")
        print(f"Training Hybrid Compensation — {total_timesteps} steps")
        print(f"Max current: {self.max_current}C  |  Max comp: ±{self.max_comp}C")
        print(f"{'='*70}\n")

        class ZeroPlatingCB(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)
                self.ep = 0; self.plates = 0
            def _on_step(self):
                if self.locals.get("done", False):
                    self.ep += 1
                    info = self.locals.get("infos", [{}])[0]
                    if info.get("plating_detected", False):
                        self.plates += 1
                    if self.ep % 100 == 0:
                        rate = self.plates / self.ep * 100
                        print(f"  [{self.ep} eps] plating {rate:.1f}%")
                        self.plates = 0; self.ep = 0
                return True

        eval_env = self._make_eval_env()
        best_dir = f"./best_{save_path}"
        os.makedirs(best_dir, exist_ok=True)
        eval_cb = EvalCallback(eval_env, best_model_save_path=best_dir,
                               eval_freq=eval_freq, deterministic=True, verbose=1)

        self.rl_model.learn(total_timesteps=total_timesteps,
                            callback=[ZeroPlatingCB(), eval_cb],
                            progress_bar=True)
        self.rl_model.save(save_path)
        print(f"Saved → {save_path}.zip")
        return self.rl_model

    def _make_eval_env(self):
        from battery_environment import BatteryPlatingEnv
        base = BatteryPlatingEnv(max_current_C=self.max_current,
                                 dt=self.dt, target_soc=self.target_soc,
                                 analytic_only=True)
        return SafetyFirstWrapper(base, plating_penalty=-500, plating_margin=0.015)

    def evaluate(self, n_episodes: int = 10) -> Dict:
        print(f"\n{'='*70}")
        print(f"Evaluating Hybrid Compensation ({n_episodes} episodes)")
        print(f"{'='*70}")
        results = dict(plating_events=0, charging_times=[], final_soc=[],
                       min_anode_mV=[])
        for ep in range(n_episodes):
            obs, _ = self.original_env.reset()
            done   = False; min_ap = 1.0
            while not done:
                a = self.get_action(obs, use_rl=True)
                obs, _, terminated, truncated, info = self.original_env.step([a])
                done = terminated or truncated
                min_ap = min(min_ap, info.get("anode_potential", 0))
            results["plating_events"] += int(info.get("plating_detected", False))
            results["charging_times"].append(info.get("time", 0))
            results["final_soc"].append(info.get("soc", 0))
            results["min_anode_mV"].append(min_ap * 1000)
            status = "❌ PLATING" if info.get("plating_detected") else "✓"
            print(f"  Ep {ep+1:3d}: {status} "
                  f"t={info.get('time',0):.0f}s  "
                  f"SoC={info.get('soc',0):.3f}  "
                  f"minVa={min_ap*1000:+.1f}mV")
        rate = results["plating_events"] / n_episodes * 100
        print(f"\nPlating: {results['plating_events']}/{n_episodes} ({rate:.1f}%)  "
              f"|  Avg time: {np.mean(results['charging_times']):.0f}s")
        return results

    def get_compensation_statistics(self) -> Dict:
        if not self.comp_history:
            return dict(mean=0, std=0, min=0, max=0)
        return dict(mean=float(np.mean(self.comp_history)),
                    std=float(np.std(self.comp_history)),
                    min=float(np.min(self.comp_history)),
                    max=float(np.max(self.comp_history)))


# ===========================================================================
# Hybrid: Adaptive blending weight (RL learns α ∈ [0,1])
# ===========================================================================

class WeightNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 1),         nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleRLPolicy8C:
    """
    Heuristic RL-like policy for 8C-capable battery.
    Aggressive early (up to 8C), tapers near SoC limit and
    backs off near plating.
    """
    def __init__(self, max_current: float = 8.0):
        self.max_current = max_current
        self._current    = 2.0

    def get_action(self, obs: np.ndarray) -> float:
        soc   = float(obs[0])
        temp  = float(obs[1])
        anode = float(obs[2]) if len(obs) > 2 else 0.1
        temp_c = temp - 273.15

        if anode < 0.02:
            self._current = max(0.3, self._current * 0.70)
        elif anode < 0.04:
            self._current = max(0.5, self._current * 0.85)
        elif soc < 0.15 and temp_c < 45:
            self._current = min(self.max_current, self._current + 0.2)
        elif soc < 0.35 and temp_c < 50:
            self._current = min(self.max_current * 0.8, self._current + 0.1)
        elif soc < 0.55:
            self._current = min(5.0, self._current + 0.05)
        elif soc > 0.70:
            self._current = max(1.0, self._current - 0.15)
        elif temp_c > 55:
            self._current = max(0.5, self._current - 0.2)
        elif temp_c > 48:
            self._current = max(1.0, self._current - 0.1)

        return float(np.clip(self._current, 0.2, self.max_current))


class AdaptiveWeightHybrid:
    """
    Learns α(state) ∈ [0.1, 0.9] to blend MPC and RL actions:
        u = α·u_mpc + (1−α)·u_rl
    """

    def __init__(
        self,
        env,
        mpc,
        learning_rate: float = 3e-4,
        hidden_dim:    int   = 128,
    ):
        self.env         = env
        self.mpc         = mpc
        self.max_current = env.max_current
        self.dt          = env.dt
        self.target_soc  = env.target_soc
        self.original_env = env

        state_dim = env.observation_space.shape[0]
        self.weight_net  = WeightNetwork(state_dim, hidden_dim)
        self.optimizer   = optim.Adam(self.weight_net.parameters(), lr=learning_rate)

        self.replay  = deque(maxlen=20_000)
        self.batch   = 128
        self.gamma   = 0.99
        self.rl_pol  = SimpleRLPolicy8C(max_current=env.max_current)

        self._step   = 0
        self.loss_h:   List[float] = []
        self.weight_h: List[float] = []
        self.reward_h: List[float] = []

        print(f"AdaptiveWeightHybrid initialised (max {self.max_current}C)")

    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> float:
        soc  = float(obs[0])
        temp = float(obs[1])

        u_mpc = self.mpc.solve([soc, temp])
        u_rl  = self.rl_pol.get_action(obs)

        with torch.no_grad():
            w = self.weight_net(torch.FloatTensor(obs).unsqueeze(0)).item()
        if not deterministic:
            w = float(np.clip(w + np.random.normal(0, 0.05), 0.1, 0.9))
        else:
            w = float(np.clip(w, 0.1, 0.9))

        u = w * u_mpc + (1 - w) * u_rl
        self.weight_h.append(w)
        return float(np.clip(u, 0.2, self.max_current))

    def _train_step(self, s, a, r, s2, done) -> float:
        self.replay.append((s, a, r, s2, done))
        if len(self.replay) < self.batch:
            return 0.0

        batch  = random.sample(self.replay, self.batch)
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        rewards = torch.FloatTensor(np.array([b[2] for b in batch]))

        weights = self.weight_net(states).squeeze()
        if rewards.std() > 1e-8:
            r_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            r_norm = rewards

        # Policy-gradient-like loss: maximise expected reward
        loss = -(r_norm * torch.log(weights + 1e-8)).mean()
        # Entropy regularisation: keep weight away from extremes
        loss += 0.02 * ((weights - 0.5) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.weight_net.parameters(), 1.0)
        self.optimizer.step()
        self._step += 1
        self.loss_h.append(loss.item())
        return loss.item()

    def train(
        self,
        total_timesteps: int = 60_000,
        save_path:       str = "hybrid_weight_model",
    ):
        print(f"\nTraining AdaptiveWeightHybrid — {total_timesteps} steps")
        ep = 0; step_count = 0; ep_r = 0.0
        obs, _ = self.env.reset()

        while step_count < total_timesteps:
            a    = self.get_action(obs, deterministic=False)
            nobs, r, terminated, truncated, info = self.env.step([a])
            done = terminated or truncated

            self._train_step(obs, a, r, nobs, done)
            ep_r += r; obs = nobs; step_count += 1

            if done:
                ep += 1
                self.reward_h.append(ep_r)
                if ep % 50 == 0:
                    avg_r = np.mean(self.reward_h[-50:])
                    avg_w = np.mean(self.weight_h[-200:]) if self.weight_h else 0.5
                    print(f"  Ep {ep:5d}  avgR={avg_r:8.1f}  avgW={avg_w:.3f}")
                obs, _ = self.env.reset(); ep_r = 0.0

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                    exist_ok=True)
        torch.save({"wn": self.weight_net.state_dict()}, save_path + ".pt")
        print(f"Saved → {save_path}.pt")
        return self

    def load(self, path: str):
        self.weight_net.load_state_dict(torch.load(path + ".pt")["wn"])
        print(f"Loaded from {path}.pt")

    def evaluate(self, n_episodes: int = 10) -> Dict:
        print(f"\nEvaluating AdaptiveWeightHybrid ({n_episodes} episodes)")
        results = dict(plating_events=0, charging_times=[], final_soc=[],
                       avg_weights=[])
        for ep in range(n_episodes):
            obs, _ = self.env.reset(); done = False; ep_w = []
            while not done:
                a = self.get_action(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.env.step([a])
                done = terminated or truncated
                ep_w.append(self.weight_h[-1] if self.weight_h else 0.5)
            results["plating_events"] += int(info.get("plating_detected", False))
            results["charging_times"].append(info.get("time", 0))
            results["final_soc"].append(info.get("soc", 0))
            results["avg_weights"].append(float(np.mean(ep_w)))
            s = "⚠️" if info.get("plating_detected") else "✓"
            print(f"  Ep {ep+1:3d}: {s} "
                  f"t={info.get('time',0):.0f}s  "
                  f"w={results['avg_weights'][-1]:.3f}")
        rate = results["plating_events"] / n_episodes * 100
        print(f"Plating {rate:.1f}%  |  Avg t={np.mean(results['charging_times']):.0f}s  "
              f"|  Avg w={np.mean(results['avg_weights']):.3f}")
        return results


# ===========================================================================
# Safe deterministic hybrid (no RL, guaranteed safe)
# ===========================================================================

class SafeHybridController:
    """
    Deterministic hybrid: MPC output clipped by stage-based safe limits.
    Guaranteed zero plating for the 8C-capable cell.
    """

    # Stage limits derived from env anode model (anode > 20 mV at each boundary)
    _STAGES = [
        (0.10, 8.0), (0.20, 7.0), (0.30, 6.0),
        (0.40, 5.5), (0.50, 5.0), (0.58, 4.5),
        (0.64, 4.0), (0.70, 3.0), (0.75, 2.0),
        (0.78, 1.5), (1.00, 1.0),
    ]

    def __init__(self, env, mpc):
        self.env         = env
        self.mpc         = mpc
        self.max_current = env.max_current

    def get_action(self, obs: np.ndarray) -> float:
        soc   = float(obs[0])
        temp  = float(obs[1])
        anode = float(obs[2]) if len(obs) > 2 else 0.1
        temp_c = temp - 273.15

        u_mpc = self.mpc.solve([soc, temp, anode])

        # Stage limit
        limit = 1.0
        for boundary, max_c in self._STAGES:
            if soc < boundary:
                limit = min(max_c, self.max_current)
                break

        # Temperature de-rating
        if temp_c > 60:
            limit *= 0.4
        elif temp_c > 55:
            limit *= 0.6
        elif temp_c > 45:
            limit *= 0.8

        return float(np.clip(min(u_mpc, limit), 0.2, self.max_current))


# ===========================================================================
# Smoke test
# ===========================================================================
if __name__ == "__main__":
    print("Testing controllers on 8C-capable battery")
    print("=" * 60)

    try:
        from battery_environment import BatteryPlatingEnv
        from mpc_controller import Safe8CMPC, AdaptiveMPC

        env = BatteryPlatingEnv(max_current_C=8.0, dt=5.0, target_soc=0.8,
                                analytic_only=True)
        mpc = Safe8CMPC(max_current=8.0)

        print("\nSafeHybridController sweep:")
        safe = SafeHybridController(env, mpc)
        obs, _ = env.reset()
        for soc in np.arange(0.05, 0.82, 0.05):
            obs_fake = np.array([soc, 308.15, _anode_potential(soc, 4.0), 3.8],
                                dtype=np.float32)
            action = safe.get_action(obs_fake)
            ap     = _anode_potential(soc, action)
            flag   = "⚠️" if ap < 0 else "✓"
            print(f"  SoC={soc:.2f}  {action:.2f}C  anode={ap*1000:+.1f}mV  {flag}")

        print("\nAdaptiveWeightHybrid quick test:")
        hybrid = AdaptiveWeightHybrid(env, mpc)
        obs, _ = env.reset()
        action = hybrid.get_action(obs)
        print(f"  First action: {action:.2f}C")

        print("\nBatteryRLController (untrained) quick test:")
        rl = BatteryRLController(env)
        obs, _ = env.reset()
        action = rl.get_action(obs)
        print(f"  First action: {action:.2f}C")

        print("\n✓ All controllers ready for 8C battery")

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()