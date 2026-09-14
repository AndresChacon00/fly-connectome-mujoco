"""¿Puede el cerebro DIRIGIR (no sólo permitir) la búsqueda de comida?

Navegación 100 % cerebral: sin rumbo heurístico. El giro sale del índice lateral
(izq-der)/(izq+der) de las neuronas de proyección olfativas + células de Kenyon, corregido
por su sesgo de base (el conectoma simulado es asimétrico), y la marcha sólo avanza si las
motoneuronas de patas superan el umbral. Se prueba desde varios ángulos iniciales.

Uso: .venv\Scripts\python test_navegacion_cerebral.py [K] [angulos separados por coma en grados]
"""
import os, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from mundo_abierto_env import Cuerpo, Fantasma, Z_WALK, CTRL_DT, WALK_EVERY
from janelia_policy import load_policy
from sim_brain import BrainSim

K = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
ANGULOS = [float(a) for a in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['60', '-60'])]
FOOD = np.array([5.0, 0.0]); LAM = 3.0; ESCALA = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0; UMBRAL_PATAS = 1.5; T_MAX = 8.0

sim = BrainSim(threshold=200.0)
neu = sim.neurons; side = neu['Soma side'].astype(str); sup = neu['Super Class']; cls = neu['Class'].astype(str)
tipo = neu['Primary Cell Type'].astype(str)
orn = (neu['Class'] == 'olfactory') & (sup == 'cb_sensory')
ornL = sim.idx_of(neu[orn & (side == 'left')]['Root ID']); ornR = sim.idx_of(neu[orn & (side == 'right')]['Root ID'])
lat = cls.isin(['ALPN', 'Kenyon_Cell'])
latL = sim.idx_of(neu[lat & (side == 'left')]['Root ID']); latR = sim.idx_of(neu[lat & (side == 'right')]['Root ID'])
dnp09 = sim.idx_of(neu[tipo.str.contains('DNp09', na=False)]['Root ID'])
patas = sim.idx_of(neu[(sup == 'vnc_motor') & ~tipo.str.contains('DLM|DVM|TTM', na=False)]['Root ID'])

class Lector:
    def __init__(self, tau=80.0):
        self.d = np.exp(-1 / tau); self.L = 0.0; self.R = 0.0; self.patas = np.zeros(len(patas), np.float32)
        self.dp = np.exp(-1 / 40.0)
    def paso(self, spikes):
        self.L = self.d * self.L + (1 - self.d) * spikes[latL].mean() * 1000
        self.R = self.d * self.R + (1 - self.d) * spikes[latR].mean() * 1000
        self.patas = self.dp * self.patas + (1 - self.dp) * spikes[patas] * 1000
    def indice(self):
        return (self.L - self.R) / (self.L + self.R + 1e-6)

def paso_cerebro(spikes, hzL, hzR, marcha, lector):
    s = spikes.copy()
    if hzL > 0: s[ornL[sim.rng.random(len(ornL)) < hzL / 1000]] = True
    if hzR > 0: s[ornR[sim.rng.random(len(ornR)) < hzR / 1000]] = True
    if marcha: s[dnp09[sim.rng.random(len(dnp09)) < 0.15]] = True
    out = sim.step(spikes_in=s); lector.paso(out); return out

# --- calibración del sesgo de base: olor simétrico a 3 intensidades ---
print('Calibrando sesgo lateral de base (olor simétrico)...', flush=True)
ev_cal, idx_cal = [], []
for hz in [6, 20, 60, 100, 160]:
    sim.reset(); lec = Lector(); sp = np.zeros(sim.n, bool)
    for t in range(500):
        sp = paso_cerebro(sp, hz / 2, hz / 2, True, lec)
    ev_cal.append((lec.L + lec.R) / 2); idx_cal.append(lec.indice())
    print(f'  {hz} Hz -> evidencia {ev_cal[-1]:.1f} Hz, indice base {idx_cal[-1]:+.3f}')
def SESGO_FN(lec):   # el sesgo depende de la intensidad: se interpola por la evidencia (L+R)/2
    return float(np.interp((lec.L + lec.R) / 2, ev_cal, idx_cal))
print(f'K = {K}')

def correr(ang0):
    cuerpo = Cuerpo(food_positions=[tuple(FOOD)])
    pw = load_policy('walking')
    sim.reset(); lec = Lector(); sp = np.zeros(sim.n, bool)
    yaw0 = np.deg2rad(ang0)
    from mundo_abierto_env import quat_heading
    cuerpo.walker.set_pose(cuerpo.physics, np.array([0, 0, Z_WALK]), quat_heading(yaw0, 0)); cuerpo.physics.forward()
    gw = Fantasma([0, 0], yaw0, Z_WALK); walk_act = np.zeros(59); acc = 0.0; t0 = time.time()
    traza = []
    i = 0
    while i * CTRL_DT < T_MAX:
        t = i * CTRL_DT; p, _ = cuerpo.pose(); yaw = cuerpo.heading()
        d = np.linalg.norm(p[:2] - FOOD); c = np.exp(-d / LAM)
        b = (np.arctan2(FOOD[1] - p[1], FOOD[0] - p[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        hzL = ESCALA * c * (1 + 0.6 * np.sin(b)); hzR = ESCALA * c * (1 - 0.6 * np.sin(b))   # antenas izq/der
        acc += CTRL_DT * 1000
        while acc >= 1:
            sp = paso_cerebro(sp, hzL, hzR, d > 0.6, lec); acc -= 1
        if d < 0.6:
            return True, t, traza
        if i % WALK_EVERY == 0:
            camina = lec.patas.mean() > UMBRAL_PATAS
            giro = float(np.clip(K * (lec.indice() - SESGO_FN(lec)), -2.0, 2.0)) if (camina and t > 0.3) else 0.0
            walk_act = pw(cuerpo.obs_caminar(gw.futuro(65, WALK_EVERY * CTRL_DT, 2.0 if (camina and t > 0.3) else 0.0, giro)))
        ctrl = cuerpo.ctrl_base(); cuerpo.poner_caminata(ctrl, walk_act); cuerpo.alas_plegadas(ctrl); cuerpo.probóscide(ctrl, 0); cuerpo.paso(ctrl)
        if i % 2500 == 0:
            traza.append((t, d, np.rad2deg(b), lec.indice() - SESGO_FN(lec)))
            print(f'   t={t:4.1f}s dist={d:5.2f} cm  comida a {np.rad2deg(b):+6.1f} grados  indice-sesgo={lec.indice()-SESGO_FN(lec):+.3f}  giro={giro:+.2f}  [{time.time()-t0:.0f}s]', flush=True)
        i += 1
    return False, T_MAX, traza

res = []
for a in ANGULOS:
    print(f'\n=== Inicio con la comida a {a:+.0f} grados, 5 cm ===', flush=True)
    ok, t, tr = correr(a); res.append((a, ok, t))
    print('  ->', 'LLEGÓ en %.1f s' % t if ok else 'NO llegó en %.0f s' % T_MAX)
print('\nResumen:', res)
