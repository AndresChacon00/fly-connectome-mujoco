"""Video de la mosca controlada por conectoma + decodificador.

Uso:
    .venv\\Scripts\\python video_brain.py [ruta_modelo.zip]
"""
import glob
import os
import sys

import imageio
from stable_baselines3 import PPO

from brain_body_env import make_brain_env


def latest_model():
    candidates = glob.glob("checkpoints_brain/brain_*.zip") + glob.glob("modelos/brain_ppo.zip")
    if not candidates:
        sys.exit("No hay modelo entrenado de la Fase 3 todavía.")
    return max(candidates, key=os.path.getmtime)


model_path = sys.argv[1] if len(sys.argv) > 1 else latest_model()
print("Usando modelo:", model_path)
model = PPO.load(model_path)

env = make_brain_env()
obs, _ = env.reset()
frames = []
total_reward = 0.0
for i in range(600):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if i % 2 == 0:
        frames.append(env.body.unwrapped._env.physics.render(camera_id=1, width=640, height=480))
    if terminated or truncated:
        obs, _ = env.reset()

out = "videos/fases_entrenamiento/fase3_ppo_con_conectoma.mp4"
imageio.mimsave(out, frames, fps=30)
print(f"Video guardado en {out} | reward acumulado: {total_reward:.1f}")
