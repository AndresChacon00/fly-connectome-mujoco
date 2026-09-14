"""Control científico: conectoma BARAJADO.

Mismas neuronas, mismo número de conexiones, mismos pesos y signos (cada neurona
conserva su neurotransmisor y sus conexiones salientes) — pero el DESTINO de cada
conexión se permuta al azar. Se destruye únicamente el cableado específico que
construyó la evolución. Si este cerebro rinde igual que el real, la ventaja era
solo "tener un reservorio grande"; si rinde peor, el cableado real importa.
"""
import numpy as np
from scipy import sparse

rng = np.random.default_rng(7)

W = sparse.load_npz("data/brain.npz").tocoo()
post = W.row.copy()
rng.shuffle(post)  # permuta los destinos; origen y peso quedan juntos
Ws = sparse.csr_matrix((W.data, (post, W.col)), shape=W.shape)
sparse.save_npz("data/brain_shuffled.npz", Ws)
print(f"Conectoma barajado: {Ws.nnz:,} conexiones (original: {W.nnz:,})")
print("Guardado: data/brain_shuffled.npz")
