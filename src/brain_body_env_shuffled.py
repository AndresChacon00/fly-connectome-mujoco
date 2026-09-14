"""Igual que brain_body_env, pero con el conectoma barajado (control científico)."""
from brain_body_env import BrainBodyEnv
from sim_brain import BrainSim


class ShuffledBrainBodyEnv(BrainBodyEnv):
    def __init__(self, seed=0):
        # mismo constructor pero cargando la matriz barajada
        self._npz = "data/brain_shuffled.npz"
        super().__init__(seed=seed)
        # umbral calibrado para igualar el nivel de actividad del conectoma real
        # (motoneuronas activas ~379 vs 336 reales; ver conversación de calibración)
        self.brain = BrainSim(threshold=130.0, seed=seed, npz_path=self._npz)


def make_shuffled_env():
    return ShuffledBrainBodyEnv()


if __name__ == "__main__":
    env = make_shuffled_env()
    obs, _ = env.reset()
    for _ in range(20):
        obs, r, te, tr, _ = env.step(env.action_space.sample() * 0.1)
    print("Entorno barajado OK | spikes:", int(env._spikes.sum()),
          "| motoneuronas activas:", int((env.motor_rates > 0).sum()))
