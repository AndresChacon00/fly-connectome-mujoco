"""Genera un video de la mosca usando el modelo entrenado (o el más reciente checkpoint).

Uso:
    .venv\\Scripts\\python video_walk.py [ruta_modelo.zip]
"""
import glob
import os
import sys

import imageio
import numpy as np
from stable_baselines3 import PPO

from fly_env import make_env


def latest_model():
    candidates = glob.glob("checkpoints/walk_*.zip") + glob.glob("modelos/walk_ppo.zip")
    if not candidates:
        sys.exit("No hay ningún modelo entrenado todavía (ni checkpoints/ ni walk_ppo.zip).")
    return max(candidates, key=os.path.getmtime)


model_path = sys.argv[1] if len(sys.argv) > 1 else latest_model()
print("Usando modelo:", model_path)
model = PPO.load(model_path)

env = make_env()
obs, _ = env.reset()
frames = []
total_reward = 0.0
for i in range(600):  # ~10 s de simulación
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if i % 2 == 0:
        frames.append(env.unwrapped._env.physics.render(camera_id=1, width=640, height=480))
    if terminated or truncated:
        obs, _ = env.reset()

out = "videos/fases_entrenamiento/fase1_ppo_solo_cuerpo.mp4"
imageio.mimsave(out, frames, fps=30)
print(f"Video guardado en {out} | reward acumulado: {total_reward:.1f}")
