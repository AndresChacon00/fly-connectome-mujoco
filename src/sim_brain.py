"""Simulador de spikes (leaky integrate-and-fire) para el conectoma completo.

Modelo simplificado estilo Shiu et al. 2024:
  - Cada neurona tiene un voltaje v que decae con constante tau.
  - Recibe corriente = suma de (peso sinaptico con signo) de las neuronas que
    dispararon en el paso anterior.
  - Si v supera el umbral, dispara: v se resetea y entra en refractario.
  - El cerebro está en silencio salvo lo que tú estimules.
"""
import numpy as np
import pandas as pd
from scipy import sparse


class BrainSim:
    def __init__(self, threshold=200.0, tau_ms=20.0, dt_ms=1.0, refractory_ms=2.0,
                 seed=0, npz_path="data/brain.npz"):
        self.W = sparse.load_npz(npz_path)  # W[post, pre]
        self.neurons = pd.read_parquet("data/neurons.parquet")
        self.n = self.W.shape[0]
        self.id2idx = pd.Series(np.arange(self.n), index=self.neurons["Root ID"].to_numpy())
        self.threshold = threshold
        self.decay = np.exp(-dt_ms / tau_ms)
        self.dt = dt_ms
        self.refr_steps = int(refractory_ms / dt_ms)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.v = np.zeros(self.n, dtype=np.float32)
        self.refr = np.zeros(self.n, dtype=np.int32)

    def idx_of(self, root_ids):
        return self.id2idx[list(root_ids)].to_numpy()

    def step(self, spikes_in=None, stim_idx=None, stim_rate_hz=0.0):
        """Un paso de dt ms. stim_idx: neuronas estimuladas externamente
        (disparan con probabilidad stim_rate_hz*dt). Devuelve vector booleano de spikes."""
        spikes = np.zeros(self.n, dtype=bool)
        if spikes_in is not None:
            self.v += self.W @ spikes_in.astype(np.float32)
        self.v *= self.decay
        ready = self.refr <= 0
        spikes = (self.v >= self.threshold) & ready
        if stim_idx is not None and stim_rate_hz > 0:
            p = stim_rate_hz * self.dt / 1000.0
            forced = self.rng.random(len(stim_idx)) < p
            spikes[stim_idx[forced]] = True
        self.v[spikes] = 0.0
        self.refr[spikes] = self.refr_steps
        self.refr -= 1
        return spikes

    def run(self, ms, stim_idx=None, stim_rate_hz=100.0, stim_window=None,
            record_idx=None):
        """Simula `ms` milisegundos. Devuelve (raster de record_idx, spikes totales por paso).
        stim_window = (inicio_ms, fin_ms) de la estimulación."""
        steps = int(ms / self.dt)
        raster = np.zeros((steps, len(record_idx)), dtype=bool) if record_idx is not None else None
        total = np.zeros(steps, dtype=np.int32)
        spikes = np.zeros(self.n, dtype=bool)
        for t in range(steps):
            t_ms = t * self.dt
            stim_on = stim_window is None or (stim_window[0] <= t_ms < stim_window[1])
            spikes = self.step(
                spikes_in=spikes,
                stim_idx=stim_idx if stim_on else None,
                stim_rate_hz=stim_rate_hz,
            )
            total[t] = spikes.sum()
            if raster is not None:
                raster[t] = spikes[record_idx]
        return raster, total
