"""Fase 2, paso 1: convierte los CSVs del conectoma a una matriz dispersa binaria.

Entrada:  data/connections_princeton.csv, data/neurons.csv
Salida:   data/brain.npz (matriz de pesos con signo) y data/neurons.parquet (metadatos)

El signo de cada sinapsis viene del neurotransmisor de la neurona PREsináptica:
  ACH (acetilcolina) -> excitatoria (+)
  GABA, GLUT (glutamato en mosca) -> inhibitorias (-)
  resto (DA, SER, OCT, desconocido) -> excitatorias débiles (+)
El peso = número de sinapsis entre el par de neuronas.
"""
import numpy as np
import pandas as pd
from scipy import sparse

print("Cargando neurons.csv ...")
neurons = pd.read_csv("data/neurons.csv", low_memory=False)
neurons.columns = [c.strip() for c in neurons.columns]
print(f"  {len(neurons):,} neuronas")

# Índice denso: root_id -> 0..N-1
root_ids = neurons["Root ID"].to_numpy()
id2idx = pd.Series(np.arange(len(root_ids)), index=root_ids)

print("Cargando connections_princeton.csv (~9M filas, toma 1-2 min) ...")
conns = pd.read_csv(
    "data/connections_princeton.csv",
    usecols=["pre_root_id", "post_root_id", "syn_count", "nt_type"],
    dtype={"pre_root_id": np.int64, "post_root_id": np.int64, "syn_count": np.int32,
           "nt_type": "category"},
)
print(f"  {len(conns):,} filas de conexión")

# Agregar por par (las filas vienen separadas por neuropilo)
print("Agregando sinapsis por par de neuronas ...")
pairs = conns.groupby(["pre_root_id", "post_root_id"], observed=True).agg(
    syn_count=("syn_count", "sum"),
    nt_type=("nt_type", "first"),
).reset_index()
print(f"  {len(pairs):,} pares únicos pre->post")

# Filtrar pares cuyas neuronas no están en neurons.csv
mask = pairs["pre_root_id"].isin(id2idx.index) & pairs["post_root_id"].isin(id2idx.index)
print(f"  descartando {(~mask).sum():,} pares con neuronas fuera del catálogo")
pairs = pairs[mask]

pre = id2idx[pairs["pre_root_id"]].to_numpy()
post = id2idx[pairs["post_root_id"]].to_numpy()

# Signo por neurotransmisor de la presináptica (viene de neurons.csv, no del CSV de conexiones)
SIGN = {"ACH": +1.0, "GABA": -1.0, "GLUT": -1.0, "DA": +0.5, "SER": +0.5, "OCT": +0.5, "HIST": -1.0}
nt_by_neuron = pd.Series(
    neurons["Predicted NT type"].astype(str).map(SIGN).fillna(+0.5).to_numpy(dtype=np.float32),
    index=root_ids,
)
sign = nt_by_neuron[pairs["pre_root_id"]].to_numpy(dtype=np.float32)
weight = sign * pairs["syn_count"].to_numpy(dtype=np.float32)

n = len(root_ids)
# W[j, i] = peso de i -> j  (filas = receptora), para propagar con W @ spikes
W = sparse.csr_matrix((weight, (post, pre)), shape=(n, n))
sparse.save_npz("data/brain.npz", W)

neurons.to_parquet("data/neurons.parquet")

exc = (weight > 0).sum()
inh = (weight < 0).sum()
print(f"\nCerebro construido: {n:,} neuronas, {W.nnz:,} conexiones")
print(f"  excitatorias: {exc:,} ({100*exc/len(weight):.0f}%) | inhibitorias: {inh:,} ({100*inh/len(weight):.0f}%)")
print(f"  sinapsis totales: {int(np.abs(weight).sum()):,}")
print("Guardado: data/brain.npz y data/neurons.parquet")
