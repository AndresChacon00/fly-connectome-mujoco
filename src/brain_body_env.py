"""Fase 3: el conectoma real en el lazo cuerpo-cerebro.

Arquitectura por paso de control:
  1. SENTIDOS -> CEREBRO: el tacto de cada una de las 6 patas de MuJoCo estimula
     su grupo (fijo) de neuronas mecanosensoriales reales; las neuronas de comando
     de marcha (DNp09, DNa01, DNa02) reciben un impulso constante de "camina".
  2. El conectoma se simula BRAIN_STEPS milisegundos.
  3. CEREBRO -> OBSERVACIÓN: la tasa de disparo suavizada de las 815 motoneuronas
     se añade a la observación. La política de PPO (el "decodificador") aprende a
     traducir esa actividad + los sentidos del cuerpo en las 59 señales musculares.

El cerebro NO se entrena: es un generador de dinámica congelado (reservorio).
"""
import gymnasium as gym
import numpy as np

from fly_env import make_env as make_body_env
from sim_brain import BrainSim

BRAIN_STEPS = 3        # ms de cerebro por paso de control del cuerpo
CMD_RATE = 120.0       # Hz del comando "camina hacia adelante"
TOUCH_RATE = 250.0     # Hz máximo del estímulo táctil por pata
RATE_TAU = 0.9         # suavizado exponencial de las tasas motoras


class BrainBodyEnv(gym.Env):
    def __init__(self, seed=0):
        super().__init__()
        self.body = make_body_env()
        self.brain = BrainSim(threshold=200.0, seed=seed)
        neu = self.brain.neurons

        mech = neu[neu["Class"].isin(["mechanosensory_tactile",
                                      "mechanosensory_proprioceptive"])]
        mech_idx = self.brain.idx_of(mech["Root ID"]).copy()
        rng = np.random.default_rng(42)  # fija: misma asignación siempre
        rng.shuffle(mech_idx)
        self.leg_groups = np.array_split(mech_idx, 6)

        cmd_mask = neu["Primary Cell Type"].astype(str).str.contains(
            "DNp09|DNa01|DNa02", case=False, na=False)
        self.cmd_idx = self.brain.idx_of(neu[cmd_mask]["Root ID"])

        motor_mask = neu["Super Class"].isin(["vnc_motor", "cb_motor"])
        self.motor_idx = self.brain.idx_of(neu[motor_mask]["Root ID"])
        self.motor_rates = np.zeros(len(self.motor_idx), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(
            dict(self.body.observation_space.spaces,
                 brain_motor=gym.spaces.Box(-np.inf, np.inf,
                                            (len(self.motor_idx),), np.float32)))
        self.action_space = self.body.action_space
        self._spikes = np.zeros(self.brain.n, dtype=bool)

    def _think(self, touch):
        """Corre BRAIN_STEPS ms de cerebro estimulado por el tacto y el comando."""
        for _ in range(BRAIN_STEPS):
            spikes = self._spikes.copy()
            # estímulo externo: comando de marcha + tacto por pata
            p_cmd = CMD_RATE * self.brain.dt / 1000.0
            forced = self.brain.rng.random(len(self.cmd_idx)) < p_cmd
            ext = list(self.cmd_idx[forced])
            for leg, group in enumerate(self.leg_groups):
                p = float(np.clip(touch[leg], 0, 1)) * TOUCH_RATE * self.brain.dt / 1000.0
                if p > 0:
                    forced = self.brain.rng.random(len(group)) < p
                    ext.extend(group[forced])
            if ext:
                spikes[np.array(ext)] = True
            self._spikes = self.brain.step(spikes_in=spikes)
            self.motor_rates = (RATE_TAU * self.motor_rates
                                + (1 - RATE_TAU) * self._spikes[self.motor_idx])

    def _obs(self, body_obs):
        out = dict(body_obs)
        out["brain_motor"] = self.motor_rates * 10.0  # escala ~O(1)
        return out

    def reset(self, seed=None, options=None):
        body_obs, info = self.body.reset(seed=seed)
        self.brain.reset()
        self._spikes[:] = False
        self.motor_rates[:] = 0
        self._think(body_obs["walker/touch"])
        return self._obs(body_obs), info

    def step(self, action):
        body_obs, reward, term, trunc, info = self.body.step(action)
        self._think(body_obs["walker/touch"])
        return self._obs(body_obs), reward, term, trunc, info


def make_brain_env():
    return BrainBodyEnv()


if __name__ == "__main__":
    import time
    env = make_brain_env()
    obs, _ = env.reset()
    print("obs brain_motor:", obs["brain_motor"].shape,
          "| comando:", len(env.cmd_idx), "neuronas | tacto:",
          [len(g) for g in env.leg_groups])
    t0 = time.perf_counter()
    n_spk = []
    for i in range(100):
        obs, r, te, tr, _ = env.step(env.action_space.sample() * 0.1)
        n_spk.append(env._spikes.sum())
    dt = (time.perf_counter() - t0) / 100 * 1000
    print(f"paso completo cerebro+cuerpo: {dt:.1f} ms  (~{1000/dt:.0f} pasos/s por entorno)")
    print(f"actividad del cerebro: {np.mean(n_spk):.0f} spikes/paso | "
          f"motoneuronas activas (tasa>0): {(env.motor_rates>0).sum()}")
