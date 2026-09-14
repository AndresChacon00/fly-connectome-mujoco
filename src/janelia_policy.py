"""Carga las políticas pre-entrenadas del paper de flybody (Janelia / DeepMind).

Los SavedModel se guardaron con TF 2.8 + TFP antigua; la TFP moderna registra sus
tipos con otro nombre, así que inyectamos un alias en el registro antes de cargar.

Uso:
    from janelia_policy import load_policy
    policy = load_policy("walking")     # o "flight", "vision-bumps", "vision-trench"
    action = policy(timestep.observation)
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import type_spec_registry as reg

# fuerza la carga diferida de las distribuciones y registra los nombres antiguos
_ = tfp.distributions.Independent, tfp.distributions.Normal, tfp.distributions.Deterministic, tfp.bijectors.Tanh
_LEGACY = {
    "tensorflow_probability.python.distributions.independent.Independent_ACTTypeSpec": "tfp.distributions.Independent_ACTTypeSpec",
    "tensorflow_probability.python.distributions.normal.Normal_ACTTypeSpec": "tfp.distributions.Normal_ACTTypeSpec",
    "tensorflow_probability.python.distributions.deterministic.Deterministic_ACTTypeSpec": "tfp.distributions.Deterministic_ACTTypeSpec",
}
for old, new in _LEGACY.items():
    if new in reg._NAME_TO_TYPE_SPEC and old not in reg._NAME_TO_TYPE_SPEC:
        reg._NAME_TO_TYPE_SPEC[old] = reg._NAME_TO_TYPE_SPEC[new]

POLICY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "policies")


class JaneliaPolicy:
    def __init__(self, name):
        self.model = tf.saved_model.load(os.path.join(POLICY_DIR, name))
        self.fn = self.model.__call__
        spec = self.fn.concrete_functions[0].structured_input_signature[0][0]
        self.keys = list(spec.keys())

    def __call__(self, observation, deterministic=True):
        batch = {k: tf.convert_to_tensor(np.asarray(observation[k], dtype=np.float32)[None])
                 for k in self.keys}
        dist = self.fn(batch)
        act = dist.mean() if deterministic else dist.sample()
        return np.array(act[0], dtype=np.float64, copy=True)  # flybody escribe sobre la acción


def load_policy(name):
    return JaneliaPolicy(name)
