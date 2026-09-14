# Datos (no incluidos en el repo por tamaño)

Coloca aquí:

1. **Conectoma del sistema nervioso central de la mosca macho** (Princeton / Janelia / Google, 2026),
   descargable desde el portal del conectoma masculino (Codex / FlyWire):
   - `connections_princeton.csv` (~204 MB): `pre_root_id, post_root_id, neuropil, syn_count, nt_type`
   - `neurons.csv` (~18 MB): metadatos por neurona (tipo celular, clase, neurotransmisor, lado del soma…)

   Luego genera la matriz dispersa, el control barajado y la tabla de regiones:
   ```
   .venv\Scripts\python src\build_brain.py            # -> data/brain.npz, data/neurons.parquet
   .venv\Scripts\python src\build_brain_shuffled.py   # -> data/brain_shuffled.npz (control)
   .venv\Scripts\python src\build_regiones.py         # -> data/neuron_neuropil_out.parquet (mapa cerebral)
   ```

2. **Políticas pre-entrenadas de flybody** (Janelia / Google DeepMind): archivo
   `trained-fly-policies.zip` (6.5 MB) del Figshare de Janelia
   (https://doi.org/10.25378/janelia.25309105). Descomprímelo en `data/policies/`
   (debe quedar `data/policies/walking/saved_model.pb`, `data/policies/flight/…`).
