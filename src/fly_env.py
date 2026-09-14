"""Envuelve la tarea walk_on_ball de flybody en la API de Gymnasium (para SB3)."""
import gymnasium as gym
import numpy as np
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

from flybody.fly_envs import walk_on_ball


class Float32Obs(gym.ObservationWrapper):
    """Convierte las observaciones a float32 (SB3 no acepta float64)."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            k: gym.spaces.Box(
                low=np.float32(space.low) if hasattr(space, "low") else -np.inf,
                high=np.float32(space.high) if hasattr(space, "high") else np.inf,
                shape=space.shape,
                dtype=np.float32,
            )
            for k, space in env.observation_space.spaces.items()
        })

    def observation(self, obs):
        return {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}


def make_env():
    dm_env = walk_on_ball()
    env = DmControlCompatibilityV0(dm_env)
    env = Float32Obs(env)
    return env


if __name__ == "__main__":
    env = make_env()
    obs, info = env.reset()
    print("Espacio de observación:")
    for k, v in obs.items():
        print(f"  {k}: {np.asarray(v).shape}")
    print("Espacio de acción:", env.action_space.shape)
    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample() * 0.1)
    print("10 pasos OK, reward:", reward)
