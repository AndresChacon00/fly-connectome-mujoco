"""Visor interactivo de MuJoCo con la mosca de flybody.

Controles dentro de la ventana:
  - Arrastrar con botón izquierdo: rotar cámara
  - Botón derecho: mover cámara
  - Scroll: zoom
  - Doble clic en una parte del cuerpo y Ctrl+arrastrar: aplicarle fuerza (¡jálale una pata!)
  - Espacio: pausar / reanudar
  - Backspace: reiniciar la simulación
"""
import numpy as np
from dm_control import viewer

from flybody.fly_envs import walk_on_ball

env = walk_on_ball()
spec = env.action_spec()


def policy(timestep):
    # Acciones aleatorias suaves: la mosca "tiembla" pero aún no sabe caminar.
    # Cuando entrenemos un cerebro, esta función será reemplazada por él.
    return 0.1 * np.random.uniform(spec.minimum, spec.maximum)


viewer.launch(env, policy=policy)
