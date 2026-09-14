"""Mosca libre en el piso — física pura, sin tarea ni episodios.

Controles:
  - Clic izquierdo arrastrar: rotar cámara | derecho: mover | scroll: zoom
  - Doble clic en una parte del cuerpo para seleccionarla, luego:
      Ctrl + arrastrar derecho  -> aplicar fuerza (jalar/empujar)
      Ctrl + arrastrar izquierdo -> aplicar torque (girar)
  - Espacio: pausar | Backspace: reiniciar
  - En el panel izquierdo puedes activar/desactivar visualizaciones
    (contactos, fuerzas, transparencia, etc.)
"""
import os

import flybody
import mujoco
import mujoco.viewer

xml = os.path.join(os.path.dirname(flybody.__file__), "fruitfly", "assets", "floor.xml")
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)

print(f"Modelo cargado: {model.nbody} cuerpos, {model.njnt} articulaciones, "
      f"{model.nu} actuadores, {model.nsensor} sensores")

mujoco.viewer.launch(model, data)
