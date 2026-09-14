"""Entrena a la mosca a caminar sobre la bola con PPO.

Uso:
    .venv\\Scripts\\python train_walk.py [pasos_totales]

Reanuda automáticamente desde el último checkpoint si existe.
Checkpoints en checkpoints/, modelo final en walk_ppo.zip.
Progreso detallado en TensorBoard:  .venv\\Scripts\\tensorboard --logdir tb_logs
"""
import glob
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from fly_env import make_env

N_ENVS = 8  # moscas entrenando en paralelo


def main():
    total_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000

    env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    env = VecMonitor(env)

    checkpoints = sorted(glob.glob("checkpoints/walk_*.zip"), key=os.path.getmtime)
    if checkpoints:
        print(f"Reanudando desde {checkpoints[-1]}")
        model = PPO.load(checkpoints[-1], env=env)
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=512,
            batch_size=1024,
            learning_rate=3e-4,
            gamma=0.95,
            ent_coef=1e-3,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log="tb_logs",
            verbose=1,
        )

    save_cb = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 1),  # cada ~100k pasos totales
        save_path="checkpoints",
        name_prefix="walk",
    )

    model.learn(
        total_timesteps=total_steps,
        callback=save_cb,
        progress_bar=True,
        reset_num_timesteps=not checkpoints,
    )
    model.save("modelos/walk_ppo")
    print("Entrenamiento terminado. Modelo guardado en walk_ppo.zip")


if __name__ == "__main__":
    main()
