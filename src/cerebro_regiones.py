"""Mapa esquemático de regiones (neuropilos) del cerebro + cordón nervioso, coloreado
por la actividad real del conectoma simulado.

La actividad de cada región = suma de las tasas de disparo de las neuronas, ponderada
por la fracción de sinapsis de salida que cada neurona tiene en esa región (dato real
del CSV de conexiones). Las posiciones en el dibujo son un esquema anatómico dorsal
(izquierda de la mosca = izquierda del dibujo), NO coordenadas medidas.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

# (x, y) esquemáticos en [0,1]; y alto = anterior/dorsal del cerebro, VNC abajo.
# Regiones pareadas se dan para el lado L; el R se refleja en x -> 1-x.
LAYOUT = {
    # --- cerebro central, superior ---
    'SMP': (0.40, 0.955), 'SLP': (0.30, 0.94), 'SIP': (0.36, 0.91), 'ATL': (0.43, 0.90),
    'MB_CA': (0.27, 0.885), 'MB_PED': (0.33, 0.865), 'MB_VL': (0.36, 0.835), 'MB_ML': (0.40, 0.815),
    'LH': (0.22, 0.90), 'PB': (0.50, 0.965), 'FB': (0.50, 0.905), 'EB': (0.50, 0.855), 'NO': (0.50, 0.82),
    'CRE': (0.43, 0.845), 'LAL': (0.40, 0.775), 'BU': (0.45, 0.79), 'IB': (0.50, 0.78),
    'ICL': (0.31, 0.80), 'SCL': (0.26, 0.83), 'AOTU': (0.19, 0.86), 'AVLP': (0.17, 0.80),
    'PVLP': (0.15, 0.75), 'PLP': (0.21, 0.74), 'WED': (0.22, 0.69), 'VES': (0.36, 0.72),
    'EPA': (0.30, 0.70), 'GOR': (0.34, 0.67), 'SPS': (0.44, 0.735), 'IPS': (0.39, 0.70),
    'AL': (0.31, 0.62), 'CAN': (0.42, 0.635), 'FLA': (0.45, 0.60), 'PRW': (0.50, 0.58),
    'SAD': (0.50, 0.64), 'CB_UNASGD': (0.50, 0.72),
    # --- lóbulos ópticos ---
    'LA': (0.06, 0.80), 'AME': (0.10, 0.72), 'OL_UNASGD': (0.08, 0.86),
    # --- cordón nervioso ventral ---
    'NTct_UTct_T1': (0.44, 0.505), 'LegNp_T1': (0.30, 0.46), 'mVAC_T1': (0.42, 0.455),
    'WTct_UTct_T2': (0.40, 0.40), 'IntTct': (0.50, 0.37), 'LegNp_T2': (0.28, 0.345),
    'mVAC_T2': (0.42, 0.34), 'Ov': (0.22, 0.30), 'HTct_UTct_T3': (0.40, 0.28),
    'LegNp_T3': (0.30, 0.23), 'mVAC_T3': (0.42, 0.225), 'LTct': (0.50, 0.20),
    'AbNT': (0.45, 0.12), 'AB': (0.50, 0.08), 'VNC_UNASGD': (0.50, 0.30),
    'ADMN': (0.36, 0.53), 'PDMN': (0.36, 0.27), 'DMetaN': (0.24, 0.18), 'MesoLN': (0.18, 0.36),
    'MetaLN': (0.20, 0.22), 'ProLN': (0.20, 0.47), 'AbN4': (0.40, 0.10),
}
UNPAIRED = {'PB', 'FB', 'EB', 'NO', 'IB', 'PRW', 'SAD', 'CB_UNASGD', 'IntTct', 'LTct', 'AB', 'VNC_UNASGD'}
ETIQUETAS = {'AL': 'AL (olfato)', 'MB_CA': 'MB cáliz', 'FB': 'FB', 'EB': 'EB', 'LA': 'lámina', 'LH': 'LH',
             'LegNp_T1': 'patas T1', 'LegNp_T2': 'patas T2', 'LegNp_T3': 'patas T3',
             'WTct_UTct_T2': 'alas', 'HTct_UTct_T3': 'halterios', 'SAD': 'SAD', 'PRW': 'PRW (probóscide)',
             'AVLP': 'AVLP', 'PVLP': 'PVLP (visión)', 'LAL': 'LAL', 'SMP': 'SMP', 'AbNT': 'abdomen', 'NTct_UTct_T1': 'cuello'}


class MapaCerebral:
    def __init__(self, brain_sim, min_syn=3000, tau_ms=60.0):
        tab = pd.read_parquet('data/neuron_neuropil_out.parquet')
        tot = tab.groupby('neuropil')['syn_count'].sum()
        tab = tab[tab['neuropil'].isin(tot[tot >= min_syn].index)]
        self.regiones = sorted(tab['neuropil'].unique())
        ridx = {r: i for i, r in enumerate(self.regiones)}
        idx = brain_sim.id2idx
        tab = tab[tab['pre_root_id'].isin(idx.index)]
        per_neuron = tab.groupby('pre_root_id')['syn_count'].transform('sum')
        frac = (tab['syn_count'] / per_neuron).to_numpy(np.float32)
        rows = tab['neuropil'].map(ridx).to_numpy()
        cols = idx[tab['pre_root_id']].to_numpy()
        self.R = sparse.csr_matrix((frac, (rows, cols)), shape=(len(self.regiones), brain_sim.n))
        self.total_syn = tot.reindex(self.regiones).to_numpy()
        self.n = brain_sim.n
        self.rate = np.zeros(self.n, np.float32)      # Hz por neurona (EMA)
        self.decay = np.exp(-1.0 / tau_ms)
        self.act = np.zeros(len(self.regiones), np.float32)
        self.peak = 1.0
        # geometría
        self.xy = np.zeros((len(self.regiones), 2)); self.size = np.zeros(len(self.regiones))
        for r, i in ridx.items():
            base = r[:-2] if r.endswith(('_L', '_R')) else r
            if base not in LAYOUT:
                self.xy[i] = (0.5, 0.02); self.size[i] = 0; continue
            x, y = LAYOUT[base]
            if r.endswith('_R'):
                x = 1 - x
            self.xy[i] = (x, y)
            self.size[i] = 40 + 900 * np.sqrt(self.total_syn[i] / self.total_syn.max())

    def actualizar(self, spikes):
        """Llamar cada ms con el vector booleano de spikes."""
        self.rate = self.decay * self.rate + (1 - self.decay) * spikes * 1000.0
        self.act = self.R @ self.rate       # "spikes·Hz" ubicados en cada región
        self.peak = max(self.peak * 0.999, float(self.act.max()), 1.0)

    def top(self, k=6):
        o = np.argsort(-self.act)[:k]
        return [(self.regiones[i], float(self.act[i])) for i in o if self.act[i] > 0]

    def dibujar(self, ax, titulo='Regiones del conectoma'):
        ax.set_facecolor('#0d0d12'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        # silueta: cerebro (elipse) + lóbulos + cordón
        from matplotlib.patches import Ellipse, FancyBboxPatch
        ax.add_patch(Ellipse((0.5, 0.79), 0.62, 0.40, color='#1a1a26', zorder=0))
        ax.add_patch(Ellipse((0.09, 0.79), 0.16, 0.24, color='#1a1a26', zorder=0))
        ax.add_patch(Ellipse((0.91, 0.79), 0.16, 0.24, color='#1a1a26', zorder=0))
        ax.add_patch(FancyBboxPatch((0.30, 0.05), 0.40, 0.50, boxstyle='round,pad=0.02,rounding_size=0.12',
                                    color='#1a1a26', zorder=0))
        ax.add_patch(FancyBboxPatch((0.44, 0.52), 0.12, 0.10, boxstyle='round,pad=0.01', color='#1a1a26', zorder=0))
        v = np.log1p(self.act) / np.log1p(self.peak)
        cmap = plt.get_cmap('inferno')
        colors = cmap(np.clip(0.08 + 0.92 * v, 0, 1))
        colors[:, 3] = 0.35 + 0.65 * np.clip(v, 0, 1)
        ax.scatter(self.xy[:, 0], self.xy[:, 1], s=self.size, c=colors, edgecolors='#333', linewidths=0.4, zorder=2)
        for i, r in enumerate(self.regiones):
            base = r[:-2] if r.endswith(('_L', '_R')) else r
            if base in ETIQUETAS and (r.endswith('_L') or base in UNPAIRED):
                ax.text(self.xy[i, 0], self.xy[i, 1] - 0.028, ETIQUETAS[base], color='#bbb', fontsize=6.5,
                        ha='center', va='top', zorder=3)
        ax.text(0.02, 0.985, titulo, color='#ddd', fontsize=10, fontweight='bold', va='top')
        ax.text(0.02, 0.945, 'izq', color='#777', fontsize=7); ax.text(0.95, 0.945, 'der', color='#777', fontsize=7)
        ax.text(0.5, 0.005, 'cerebro (arriba) · cordón nervioso ventral (abajo) · esquema anatómico dorsal',
                color='#666', fontsize=6.5, ha='center')


if __name__ == '__main__':
    from sim_brain import BrainSim
    sim = BrainSim()
    mapa = MapaCerebral(sim)
    print(len(mapa.regiones), 'regiones en el mapa')
    neu = sim.neurons
    gust = sim.idx_of(neu[(neu['Class'] == 'gustatory') & (neu['Super Class'] == 'cb_sensory')]['Root ID'])
    spikes = np.zeros(sim.n, bool)
    for t in range(400):
        spikes = sim.step(spikes_in=spikes, stim_idx=gust if t > 50 else None, stim_rate_hz=100)
        mapa.actualizar(spikes)
    print('top regiones tras estímulo gustativo:', mapa.top())
    fig, ax = plt.subplots(figsize=(5.8, 7.2), dpi=100); fig.patch.set_facecolor('#0d0d12')
    mapa.dibujar(ax, 'Prueba: estímulo gustativo')
    fig.tight_layout(); fig.savefig('videos/pruebas_cuerpo/test_mapa_cerebral.png'); print('guardado videos/test_mapa_cerebral.png')
