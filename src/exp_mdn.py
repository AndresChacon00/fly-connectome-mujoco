"""Experimento 1: estimular las neuronas MDN ("moonwalker", comando de caminar
hacia atrás) y observar si la actividad llega a las motoneuronas de las patas.

Primero escanea el umbral para encontrar un régimen de actividad razonable
(ni cerebro mudo ni ataque epiléptico), luego corre el experimento completo.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim_brain import BrainSim

# --- escaneo de umbral ---
print("Escaneando umbrales (1 s de simulación cada uno)...")
for th in [50, 100, 200, 400, 800]:
    sim = BrainSim(threshold=th)
    mdn = sim.idx_of(sim.neurons[sim.neurons["Primary Cell Type"].astype(str)
                                 .str.contains("MDN", case=False, na=False)]["Root ID"])
    _, total = sim.run(1000, stim_idx=mdn, stim_rate_hz=150, stim_window=(100, 900))
    rate = total.sum() / sim.n / 1.0  # spikes por neurona por segundo (promedio global)
    print(f"  umbral {th:>4}: {total.sum():>9,} spikes totales | "
          f"tasa media global {rate:6.2f} Hz | pico {total.max():,} neuronas/ms")

th = float(input("\nElige umbral (enter = 200): ") or 200) if False else 200.0
print(f"\nExperimento completo con umbral {th} ...")

sim = BrainSim(threshold=th)
neu = sim.neurons
mdn_ids = neu[neu["Primary Cell Type"].astype(str).str.contains("MDN", case=False, na=False)]["Root ID"]
mdn = sim.idx_of(mdn_ids)
motor_mask = neu["Super Class"].isin(["vnc_motor", "cb_motor"])
motor = sim.idx_of(neu[motor_mask]["Root ID"])
print(f"Estimulando {len(mdn)} MDN a 150 Hz entre t=100 y t=900 ms; "
      f"grabando {len(motor)} motoneuronas")

raster, total = sim.run(1200, stim_idx=mdn, stim_rate_hz=150,
                        stim_window=(100, 900), record_idx=motor)

active = raster.sum(0)
n_act = (active > 0).sum()
print(f"Motoneuronas que dispararon: {n_act} de {len(motor)}")

# top tipos celulares motores activados
mn = neu[motor_mask].reset_index(drop=True)
top = (mn.assign(spikes=active).query("spikes>0")
       .groupby("Primary Cell Type")["spikes"].sum().sort_values(ascending=False))
print("\nTop tipos de motoneurona activados:")
print(top.head(12).to_string())

# --- figura ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
order = np.argsort(-active)
sub = order[:120]  # las 120 motoneuronas más activas
t_idx, n_idx = np.where(raster[:, sub])
axes[0].scatter(t_idx, n_idx, s=1.5, c="k")
axes[0].axvspan(100, 900, color="tab:orange", alpha=0.12, label="estímulo MDN (150 Hz)")
axes[0].set_ylabel("motoneurona (ordenadas por actividad)")
axes[0].set_title("Conectoma real: estímulo a MDN (caminar hacia atrás) → motoneuronas de las patas")
axes[0].legend(loc="upper right")
axes[1].plot(total, lw=0.8)
axes[1].axvspan(100, 900, color="tab:orange", alpha=0.12)
axes[1].set_ylabel("spikes/ms (todo el cerebro)")
axes[1].set_xlabel("tiempo (ms)")
fig.tight_layout()
fig.savefig("videos/experimentos_conectoma/exp_mdn.png", dpi=110)
print("\nFigura guardada: exp_mdn.png")
