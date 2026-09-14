"""Tabla neurona -> distribución de sinapsis de salida por región (neuropilo).
Alimenta el mapa cerebral de cerebro_regiones.py.  Salida: data/neuron_neuropil_out.parquet"""
import pandas as pd

c = pd.read_csv("data/connections_princeton.csv", usecols=["pre_root_id", "neuropil", "syn_count"])
tab = c.groupby(["pre_root_id", "neuropil"])["syn_count"].sum().reset_index()
tab.to_parquet("data/neuron_neuropil_out.parquet")
print(f"{len(tab):,} pares neurona-región | {tab['neuropil'].nunique()} regiones -> data/neuron_neuropil_out.parquet")
