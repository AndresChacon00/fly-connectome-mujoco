"""Corre experimentos de estimulación sobre el conectoma desde la terminal.

Uso básico:
    .venv\\Scripts\\python experimento.py --stim tipo:MDN
    .venv\\Scripts\\python experimento.py --stim id:10001,10010 --ms 400 --window 100,150
    .venv\\Scripts\\python experimento.py --stim clase:gustatory --rate 80
    .venv\\Scripts\\python experimento.py --stim clase:olfactory --record super:descending_neuron

Selectores (para --stim y --record):
    tipo:TEXTO    busca en "Primary Cell Type" (contiene, sin mayúsculas)
    clase:TEXTO   busca en "Class"  (gustatory, olfactory, visual, mechanosensory...)
    super:TEXTO   busca en "Super Class" (vnc_motor, descending_neuron, ...) admite varios con coma
    id:1,2,3      Root IDs exactos

Opciones:
    --rate HZ        frecuencia del estímulo (default 150)
    --ms MS          duración total (default 1200)
    --window A,B     ventana del estímulo en ms (default 100,900)
    --umbral U       umbral de disparo (default 200)
    --zoom A,B       recorta el eje de tiempo de la figura (para ver ritmos finos)
    --out NOMBRE     nombre del png (default: exp_<selector>.png)
"""
import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim_brain import BrainSim


def select(neurons, selector):
    kind, _, value = selector.partition(":")
    if kind == "id":
        ids = [int(x) for x in value.split(",")]
        return neurons[neurons["Root ID"].isin(ids)]
    col = {"tipo": "Primary Cell Type", "clase": "Class", "super": "Super Class"}[kind]
    if kind == "super":
        return neurons[neurons[col].isin(value.split(","))]
    return neurons[neurons[col].astype(str).str.contains(value, case=False, na=False)]


p = argparse.ArgumentParser()
p.add_argument("--stim", required=True)
p.add_argument("--record", default="super:vnc_motor,cb_motor")
p.add_argument("--rate", type=float, default=150)
p.add_argument("--ms", type=float, default=1200)
p.add_argument("--window", default="100,900")
p.add_argument("--umbral", type=float, default=200)
p.add_argument("--zoom", default=None)
p.add_argument("--out", default=None)
args = p.parse_args()

w0, w1 = (float(x) for x in args.window.split(","))
out = args.out or "videos/experimentos_conectoma/exp_" + re.sub(r"[^A-Za-z0-9]+", "_", args.stim) + ".png"

sim = BrainSim(threshold=args.umbral)
neu = sim.neurons

stim_neu = select(neu, args.stim)
rec_neu = select(neu, args.record)
if stim_neu.empty:
    raise SystemExit(f"El selector --stim '{args.stim}' no encontró neuronas.")
stim = sim.idx_of(stim_neu["Root ID"])
rec = sim.idx_of(rec_neu["Root ID"])

print(f"Estimulando {len(stim)} neuronas [{args.stim}] a {args.rate:.0f} Hz, "
      f"t={w0:.0f}-{w1:.0f} ms | grabando {len(rec)} [{args.record}] | umbral {args.umbral:.0f}")

raster, total = sim.run(args.ms, stim_idx=stim, stim_rate_hz=args.rate,
                        stim_window=(w0, w1), record_idx=rec)

active = raster.sum(0)
print(f"Respondieron {(active > 0).sum()} de {len(rec)} neuronas grabadas "
      f"| spikes totales en el cerebro: {total.sum():,}")

top = (rec_neu.reset_index(drop=True).assign(spikes=active).query("spikes>0")
       .groupby("Primary Cell Type")["spikes"].sum().sort_values(ascending=False))
if len(top):
    print("\nTop tipos celulares que respondieron:")
    print(top.head(12).to_string())

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
order = np.argsort(-active)
sub = order[:120]
t_idx, n_idx = np.where(raster[:, sub])
axes[0].scatter(t_idx * sim.dt, n_idx, s=1.5, c="k")
axes[0].axvspan(w0, w1, color="tab:orange", alpha=0.12,
                label=f"estímulo {args.stim} ({args.rate:.0f} Hz)")
axes[0].set_ylabel(f"neuronas grabadas [{args.record}]\n(ordenadas por actividad)")
axes[0].set_title(f"Estímulo: {args.stim}  →  respuesta de {args.record}")
axes[0].legend(loc="upper right")
axes[1].plot(np.arange(len(total)) * sim.dt, total, lw=0.8)
axes[1].axvspan(w0, w1, color="tab:orange", alpha=0.12)
axes[1].set_ylabel("spikes/ms (cerebro entero)")
axes[1].set_xlabel("tiempo (ms)")
if args.zoom:
    z0, z1 = (float(x) for x in args.zoom.split(","))
    axes[0].set_xlim(z0, z1)
fig.tight_layout()
fig.savefig(out, dpi=110)
print(f"\nFigura guardada: {out}")
