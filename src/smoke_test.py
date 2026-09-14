"""Prueba de humo: carga la mosca de flybody en MuJoCo y renderiza imágenes."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flybody.fly_envs import walk_on_ball

env = walk_on_ball()
timestep = env.reset()

print("Tarea: caminar sobre una bola")
print("Observaciones:", len(timestep.observation), "canales")
print("Acciones (músculos/articulaciones):", env.action_spec().shape)

# Renderizar desde varias cámaras
frames = []
for cam in range(3):
    try:
        frames.append(env.physics.render(camera_id=cam, width=640, height=480))
    except Exception as e:
        print(f"cámara {cam}: {e}")

# Dar unos pasos con acciones aleatorias suaves para verificar la física
spec = env.action_spec()
for _ in range(50):
    action = np.random.uniform(spec.minimum, spec.maximum) * 0.1
    timestep = env.step(action)
print("Física OK, reward del último paso:", timestep.reward)

frames.append(env.physics.render(camera_id=1, width=640, height=480))

fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 4))
for ax, f in zip(np.atleast_1d(axes), frames):
    ax.imshow(f)
    ax.axis("off")
fig.suptitle("flybody en MuJoCo — mosca sobre bola (última: tras 50 pasos)")
fig.tight_layout()
fig.savefig("videos/fases_entrenamiento/fly_render.png", dpi=100)
print("Guardado fly_render.png")
