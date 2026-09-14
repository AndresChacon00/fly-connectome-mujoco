"""Fase 3: entrena el decodificador (motoneuronas del conectoma -> músculos) con PPO.

Uso:
    .venv\\Scripts\\python train_brain.py [pasos_totales]

Reanuda desde checkpoints_brain/ si existe. Comparar contra la Fase 1 en TensorBoard
(tb_logs: corridas PPO_* = solo cuerpo, BRAIN_* = con conectoma).
"""
import glob
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from brain_body_env import make_brain_env

N_ENVS = 8


def main():
    total_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000

    env = SubprocVecEnv([make_brain_env for _ in range(N_ENVS)])
    env = VecMonitor(env)

    checkpoints = sorted(glob.glob("checkpoints_brain/brain_*.zip"), key=os.path.getmtime)
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
        save_freq=max(100_000 // N_ENVS, 1),
        save_path="checkpoints_brain",
        name_prefix="brain",
    )

    model.learn(
        total_timesteps=total_steps,
        callback=save_cb,
        progress_bar=True,
        reset_num_timesteps=not checkpoints,
        tb_log_name="BRAIN",
    )
    model.save("modelos/brain_ppo")
    print("Listo. Modelo guardado en brain_ppo.zip")


if __name__ == "__main__":
    main()
