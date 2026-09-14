"""VIDA DE MOSCA — mundo abierto continuo con el conectoma como cerebro.

Un solo cuerpo físico (patas + alas + probóscide) en un mundo con comida y un
depredador. El conectoma real (166,700 neuronas) recibe los sentidos de la escena y
sus neuronas deciden: caminar, comer, escapar volando, aterrizar y volver a buscar.
Izquierda del video: mapa de regiones del cerebro/cordón nervioso coloreado por la
actividad real. Derecha: el mundo 3D.

Qué decide el cerebro (lecturas reales del conectoma):
  * caminar        -> tasa de motoneuronas de patas (DNp09 = "hambre/explorar")
  * comer          -> tasa de motoneuronas de probóscide MN1-13 (estimuladas por el gusto);
                      la extensión de la probóscide ES esa tasa
  * despegar       -> spike en TTMn (looming visual LPLC2 -> Giant Fiber)
  * aterrizar      -> neuronas descendentes activadas por el olor (DNb05, DNg56, DNp12...)
  * RUMBO           -> índice lateral (izq-der) de las neuronas de proyección olfativas y
                      células de Kenyon, corregido por el sesgo de base del conectoma simulado
                      (calibrado con olor simétrico). Sin instrucción de dirección: el cerebro dirige.
Qué es heurístico (documentado): el "zigzag" exploratorio cuando NO hay olor; la ejecución
motora fina (políticas de imitación de Janelia); el salto de despegue y la nivelación al aterrizar.

Uso:  .venv\\Scripts\\python vida_de_mosca.py [--rapido]
"""
import argparse
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cerebro_regiones import MapaCerebral
from janelia_policy import load_policy
from mundo_abierto_env import CTRL_DT, WALK_EVERY, Z_WALK, Cuerpo, Fantasma
from sim_brain import BrainSim

# ----------------------------------------------------------------------------- mundo
FOODS = [np.array([5.0, 1.5]), np.array([24.0, -5.0])]
LAMBDA_SUELO, LAMBDA_AIRE = 3.0, 12.0    # alcance del olor caminando / volando (cm; en el aire la pluma llega más lejos)
UMBRAL_PATAS = 1.5
UMBRAL_PROBO = 6.0
UMBRAL_DN_OLOR = 3.0
V_VUELO = 15.0
ESCALA_OLOR = 100.0    # Hz máximos de las ORN (calibrado: con 200 la red se satura cerca de la comida)
K_GIRO = 110.0         # ganancia del giro cerebral (probada desde 0, ±60, ±120 y 180 grados)
ESCALA_AIRE = 120.0    # en vuelo la pluma llega más lejos (LAMBDA_AIRE) pero sin saturar la red
K_GIRO_AIRE = 200.0
FRAME_CADA = 50                           # pasos de control por frame (10 ms) -> 2.5x cámara lenta a 25 fps


class Cerebro:
    def __init__(self):
        self.sim = BrainSim(threshold=200.0)
        neu = self.sim.neurons
        tipo = neu["Primary Cell Type"].astype(str)
        sup = neu["Super Class"]; side = neu["Soma side"].astype(str)

        def idx(mask):
            return self.sim.idx_of(neu[mask]["Root ID"])

        olf = (neu["Class"] == "olfactory") & (sup == "cb_sensory")
        self.entradas = {
            "orn_L": idx(olf & (side == "left")), "orn_R": idx(olf & (side == "right")),
            "gusto": idx((neu["Class"] == "gustatory") & (sup == "cb_sensory")),
            "loom_L": idx((tipo == "LPLC2") & (side == "left")), "loom_R": idx((tipo == "LPLC2") & (side == "right")),
            "dnp09": idx(tipo.str.contains("DNp09", na=False)),
        }
        mech = idx(neu["Class"].isin(["mechanosensory_tactile", "mechanosensory_proprioceptive"])).copy()
        np.random.default_rng(42).shuffle(mech)
        self.tacto = np.array_split(mech, 6)
        vuelo = tipo.str.contains("DLM|DVM|TTM", na=False)
        self.salidas = {
            "patas": idx((sup == "vnc_motor") & ~vuelo),
            "probóscide": idx((sup == "cb_motor") & tipo.str.startswith("MN")),
            "alas": idx(tipo.str.contains("DLM|DVM", na=False)),
            "TTMn": idx(tipo == "TTMn"), "GF": idx(tipo.str.contains("DNp01", na=False)),
            "DNa02_L": idx((tipo == "DNa02") & (side == "left")), "DNa02_R": idx((tipo == "DNa02") & (side == "right")),
            "DN_olor": idx(tipo.isin(["DNb05", "DNg56", "DNg33", "DNp12", "DNp32"])),
        }
        # navegación cerebral: índice lateral (izq-der) de neuronas de proyección olfativas + Kenyon
        lat = neu["Class"].isin(["ALPN", "Kenyon_Cell"])
        self.lat_L = idx(lat & (side == "left")); self.lat_R = idx(lat & (side == "right"))
        self.latL_rate = 0.0; self.latR_rate = 0.0; self.lat_decay = np.exp(-1.0 / 80.0)
        self.rates = {k: np.zeros(len(v), np.float32) for k, v in self.salidas.items()}
        self.decay = np.exp(-1.0 / 40.0)
        self.spikes = np.zeros(self.sim.n, bool)
        self.t_ms = 0
        self.ttmn_ultimo = -1e9
        self.mapa = MapaCerebral(self.sim)
        self.historia = []

    def paso_ms(self, estim, tacto=None):
        """estim: {nombre_entrada: Hz}."""
        s = self.spikes.copy()
        forzar = []
        rng = self.sim.rng
        for k, hz in estim.items():
            if hz > 0:
                ids = self.entradas[k]
                forzar.append(ids[rng.random(len(ids)) < hz / 1000.0])
        if tacto is not None:
            for g, tv in zip(self.tacto, tacto):
                p = float(np.clip(tv, 0, 1)) * 0.10
                if p > 0:
                    forzar.append(g[rng.random(len(g)) < p])
        if forzar:
            s[np.concatenate(forzar)] = True
        self.spikes = self.sim.step(spikes_in=s)
        self.t_ms += 1
        for k, ids in self.salidas.items():
            self.rates[k] = self.decay * self.rates[k] + (1 - self.decay) * self.spikes[ids] * 1000.0
        self.latL_rate = self.lat_decay * self.latL_rate + (1 - self.lat_decay) * self.spikes[self.lat_L].mean() * 1000.0
        self.latR_rate = self.lat_decay * self.latR_rate + (1 - self.lat_decay) * self.spikes[self.lat_R].mean() * 1000.0
        if self.spikes[self.salidas["TTMn"]].any():
            self.ttmn_ultimo = self.t_ms
        self.mapa.actualizar(self.spikes)
        if self.t_ms % 10 == 0:
            self.historia.append((self.t_ms, {k: self.tasa(k) for k in self.salidas}, dict(estim)))

    def tasa(self, k):
        return float(self.rates[k].mean())

    def indice_lateral(self):
        return (self.latL_rate - self.latR_rate) / (self.latL_rate + self.latR_rate + 1e-6)

    def calibrar_sesgo(self, intensidades=(6, 20, 60, 100, 160)):
        """El conectoma simulado es asimétrico: mide el índice lateral con olor SIMÉTRICO a varias
        intensidades para restarlo después (el sesgo depende de la intensidad; se interpola por la
        evidencia (L+R)/2). Es una calibración del sensor, no un rumbo."""
        self.ev_cal, self.idx_cal = [], []
        for hz in intensidades:
            self.sim.reset(); self.spikes[:] = False; self.latL_rate = self.latR_rate = 0.0
            for _ in range(500):
                self.paso_ms({"orn_L": hz / 2, "orn_R": hz / 2, "dnp09": 150.0})
            self.ev_cal.append((self.latL_rate + self.latR_rate) / 2); self.idx_cal.append(self.indice_lateral())
        self.sim.reset(); self.spikes[:] = False; self.latL_rate = self.latR_rate = 0.0
        for k in self.rates: self.rates[k][:] = 0
        self.mapa.rate[:] = 0; self.t_ms = 0; self.historia.clear()

    def giro_cerebral(self, K):
        """Velocidad de giro dictada por el cerebro: K * (índice lateral - sesgo(intensidad))."""
        sesgo = float(np.interp((self.latL_rate + self.latR_rate) / 2, self.ev_cal, self.idx_cal))
        return float(K * (self.indice_lateral() - sesgo))

    def ttmn_reciente(self, ms=30):
        return self.t_ms - self.ttmn_ultimo < ms


# ----------------------------------------------------------------------------- panel
COL = {"patas": "#4fc3f7", "probóscide": "#81c784", "alas": "#ce93d8", "TTMn": "#ff7043", "GF": "#ffd54f",
       "DNa02_L": "#80cbc4", "DNa02_R": "#26a69a", "DN_olor": "#ffab91"}
COLOR_MODO = {"BUSCA": "#4fc3f7", "COME": "#81c784", "ESCAPA": "#ff7043", "VUELA": "#ce93d8",
              "ATERRIZA": "#ffd54f", "DESCANSA": "#aaa"}


def panel(cb, modo, texto, t_s, estim, w=560, h=720):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor("#0d0d12")
    ax_m = fig.add_axes([0.0, 0.34, 1.0, 0.66])
    cb.mapa.dibujar(ax_m, "Conectoma en vivo: regiones activas")
    ax_t = fig.add_axes([0.04, 0.245, 0.92, 0.085]); ax_t.axis("off")
    ax_t.text(0, 0.75, modo, color=COLOR_MODO.get(modo, "#ddd"), fontsize=20, fontweight="bold", va="center")
    ax_t.text(1, 0.78, f"t = {t_s:5.2f} s", color="#aaa", fontsize=10, ha="right", va="center")
    ax_t.text(0, 0.15, texto, color="#ddd", fontsize=8.5, va="center")
    activos = [k for k, v in estim.items() if v > 0]
    ax_t.text(1, 0.15, "sentidos: " + (", ".join(activos) if activos else "—"), color="#ffd54f", fontsize=8, ha="right", va="center")
    ax_b = fig.add_axes([0.10, 0.04, 0.87, 0.18]); ax_b.set_facecolor("#181820")
    names = list(COL)
    vals = [cb.tasa(k) for k in names]
    ax_b.bar(range(len(names)), [max(v, 0.1) for v in vals], color=[COL[k] for k in names])
    ax_b.set_yscale("log"); ax_b.set_ylim(0.1, 1000)
    ax_b.set_xticks(range(len(names))); ax_b.set_xticklabels(names, color="#ccc", fontsize=7, rotation=20)
    ax_b.tick_params(colors="#aaa", labelsize=7); ax_b.set_ylabel("Hz", color="#aaa", fontsize=8)
    for sp in ax_b.spines.values():
        sp.set_color("#444")
    ax_b.set_title("Lecturas motoras / de decisión", color="#ccc", fontsize=8.5)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


# ----------------------------------------------------------------------------- simulación
def ang_norm(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="sin panel cerebral (sólo prueba del cuerpo)")
    ap.add_argument("--out", default="videos/final/vida_de_mosca.mp4")
    args = ap.parse_args()
    t0 = time.time()
    eventos = []

    def log(s):
        print(s, flush=True); eventos.append(s)

    cuerpo = Cuerpo(food_positions=[tuple(f) for f in FOODS])
    pw, pf = load_policy("walking"), load_policy("flight")
    cb = Cerebro()
    cb.calibrar_sesgo()
    log("Sesgo lateral calibrado (olor simétrico): " + ", ".join(f"{e:.0f}Hz:{i:+.3f}" for e, i in zip(cb.ev_cal, cb.idx_cal)))
    log(f"Mundo listo: {len(FOODS)} fuentes de comida; cerebro {cb.sim.n:,} neuronas, {len(cb.mapa.regiones)} regiones")

    pos, _ = cuerpo.pose()
    gw = Fantasma(pos[:2], cuerpo.heading(), Z_WALK)
    gf = None
    modo = "BUSCA"
    hambre = 1.0
    comidas = 0
    walk_act = np.zeros(59)
    frames = []
    brain_acc = 0.0
    t_modo = 0.0
    t_pred_inicio = None
    pred_pos = None
    pred_activo = False
    texto = "DNp09 activo: el cerebro busca comida"
    probo_llegada = None
    comidos = set()
    t_volcada = None
    marcas = []
    t_fin = 16.0
    despegues = 0
    i = 0
    while True:
        t = i * CTRL_DT
        if t > t_fin:
            break
        p, _ = cuerpo.pose(); yaw = cuerpo.heading()
        # ---------- sentidos de la escena ----------
        lam = LAMBDA_AIRE if modo in ("VUELA", "ESCAPA") else LAMBDA_SUELO
        dists = [np.linalg.norm(p[:2] - f) for f in FOODS]
        conc = [np.exp(-d / lam) if k not in comidos else 0.0 for k, d in enumerate(dists)]
        j = int(np.argmax(conc)); c = conc[j]; d_food = dists[j]
        bearing = ang_norm(np.arctan2(FOODS[j][1] - p[1], FOODS[j][0] - p[0]) - yaw)  # + = comida a la izquierda
        estim = {}
        if hambre > 0.2 and c > 0.01:
            esc = ESCALA_AIRE if modo in ("VUELA", "ESCAPA") else ESCALA_OLOR
            estim["orn_L"] = esc * c * (1 + 0.6 * np.sin(bearing))
            estim["orn_R"] = esc * c * (1 - 0.6 * np.sin(bearing))
        en_comida = d_food < 0.6 and modo in ("BUSCA", "COME") and j not in comidos
        if en_comida and hambre > 0.2:
            estim["gusto"] = 120.0
        if modo == "BUSCA" and hambre > 0.2 and not en_comida:
            estim["dnp09"] = 150.0
        # depredador: aparece 1.2 s después de empezar a comer la primera vez
        if pred_activo:
            dp = np.linalg.norm(pred_pos[:2] - p[:2])
            if dp > 1.2:                                       # se acerca y se queda encima
                pred_pos = pred_pos + pred_vel * CTRL_DT
            tam_ang = np.clip(1.6 / max(dp, 0.8), 0, 1.0)      # tamaño angular aparente
            b_pred = ang_norm(np.arctan2(pred_pos[1] - p[1], pred_pos[0] - p[0]) - yaw)
            hz = 350 * tam_ang ** 1.5
            estim["loom_L"] = hz * (0.5 + 0.5 * np.sin(b_pred)) if b_pred > -1.2 else 0
            estim["loom_R"] = hz * (0.5 - 0.5 * np.sin(b_pred)) if b_pred < 1.2 else 0
            cuerpo.mover_depredador(pred_pos)
            if t - t_pred_inicio > 3.0:
                pred_activo = False; cuerpo.mover_depredador(np.array([0, 0, -5.0]))
        # ---------- cerebro (1 ms por cada 5 pasos de control) ----------
        brain_acc += CTRL_DT * 1000
        while brain_acc >= 1.0:
            cb.paso_ms(estim, tacto=cuerpo.touch() if modo in ("BUSCA", "COME", "ATERRIZA") else None)
            brain_acc -= 1.0
        # ---------- decisiones ----------
        ctrl = cuerpo.ctrl_base()
        ext = 0.0
        # enderezamiento (simplificación): si queda volcada en tierra > 0.4 s, se endereza
        if modo in ("BUSCA", "COME", "ATERRIZA", "DESCANSA"):
            if cuerpo.upright() < 0.3:
                if t_volcada is None:
                    t_volcada = t
                elif t - t_volcada > 0.4:
                    cuerpo.nivelar(); gw = Fantasma(p[:2], yaw, Z_WALK); t_volcada = None
                    log(f"t={t:.2f}s  estaba volcada: se endereza");
            else:
                t_volcada = None
        if modo in ("BUSCA", "COME"):
            if cb.ttmn_reciente() and pred_activo:
                gf = cuerpo.despegar(v_fwd=V_VUELO)
                yaw_escape = ang_norm(np.arctan2(p[1] - pred_pos[1], p[0] - pred_pos[0]))
                modo, t_modo = "ESCAPA", t; despegues += 1
                comidos.add(j)   # el lugar del depredador queda marcado como peligroso: buscará otra comida
                texto = "¡TTMn disparó! Giant Fiber -> salto y vuelo de escape"
                log(f"t={t:.2f}s  TTMn disparó ({cb.t_ms - t_loom0} ms tras aparecer el depredador) -> DESPEGUE"); marcas.append(("despegue", t))
        if modo == "BUSCA":
            # el olor ya excita algo las MN de probóscide; la decisión de comer exige el
            # AUMENTO que produce el gusto respecto a la tasa que traía al llegar
            if en_comida and probo_llegada is None:
                probo_llegada = cb.tasa("probóscide")
            if not en_comida:
                probo_llegada = None
            if en_comida and hambre > 0.2 and cb.tasa("probóscide") > max(UMBRAL_PROBO, 1.5 * probo_llegada + 3):
                modo, t_modo = "COME", t; texto = "Gusto -> MN de probóscide: comiendo (extensión = tasa neuronal)"
                log(f"t={t:.2f}s  llegó a la comida #{j+1}; MN probóscide {cb.tasa('probóscide'):.1f} Hz -> COME"); marcas.append(("come", t))
                if comidas == 0:
                    t_pred_inicio = t + 0.6
            if i % WALK_EVERY == 0:
                camina = cb.tasa("patas") > UMBRAL_PATAS and not en_comida
                speed = 2.0 if camina else 0.0
                # RUMBO 100 % CEREBRAL: giro = K * (índice lateral olfativo del conectoma - sesgo)
                if c > 0.01:
                    yaw_rate = float(np.clip(cb.giro_cerebral(K_GIRO), -2.0, 2.0)) if camina else 0.0
                else:
                    yaw_rate = 0.6 * np.sin(2 * np.pi * 0.4 * t) if camina else 0.0   # sin olor: zigzag exploratorio
                walk_act = pw(cuerpo.obs_caminar(gw.futuro(65, WALK_EVERY * CTRL_DT, speed, yaw_rate)))
                texto = (f"DNp09 -> patas; rumbo por índice lateral olfativo ({cb.indice_lateral():+.2f})" if camina and c > 0.01 else
                         "DNp09 -> patas: explorando en zigzag" if camina else "esperando la decisión de las patas")
            cuerpo.poner_caminata(ctrl, walk_act); cuerpo.alas_plegadas(ctrl)
        elif modo == "COME":
            ext = np.clip(cb.tasa("probóscide") / (3 * UMBRAL_PROBO), 0, 1)
            hambre -= 0.35 * CTRL_DT * ext
            if i % WALK_EVERY == 0:
                walk_act = pw(cuerpo.obs_caminar(gw.futuro(65, WALK_EVERY * CTRL_DT, 0.0, 0.0)))
            cuerpo.poner_caminata(ctrl, walk_act); cuerpo.alas_plegadas(ctrl)
            texto = f"comiendo: probóscide {ext*100:3.0f}%  · hambre {hambre*100:3.0f}%"
            if t_pred_inicio is not None and not pred_activo and t >= t_pred_inicio and comidas == 0 and despegues == 0:
                pred_activo = True
                pred_pos = p + np.array([-6.0 * np.cos(yaw + 0.8), -6.0 * np.sin(yaw + 0.8), 0.8])
                pred_vel = (p - pred_pos) / 1.0; pred_vel[2] = 0
                t_loom0 = cb.t_ms
                log(f"t={t:.2f}s  aparece un depredador a 6 cm (estímulo looming a LPLC2)"); marcas.append(("depredador", t))
            if hambre <= 0.2:
                comidas += 1; comidos.add(j)
                queda = len(comidos) < len(FOODS)
                hambre = 1.0 if queda else 0.0
                modo, t_modo = ("BUSCA", t) if queda else ("DESCANSA", t)
                gw = Fantasma(p[:2], yaw, Z_WALK)
                log(f"t={t:.2f}s  saciada de la comida #{j+1}" + ("" if queda else " -> no queda comida: descansa"))
        elif modo == "ESCAPA":
            err = ang_norm(yaw_escape - gf.yaw)
            z_rate = 4.0 if gf.z < 1.0 else 0.0
            act = pf(cuerpo.obs_volar(gf.futuro(6, CTRL_DT, V_VUELO, np.clip(3 * err, -2.5, 2.5), z_rate=z_rate)))
            cuerpo.alas_volando(ctrl, act[:11], act[11]); cuerpo.patas_retraidas(ctrl)
            texto = "vuelo de escape, alejándose del depredador"
            if t - t_modo > 0.8:
                modo, t_modo = "VUELA", t; texto = "volando: buscando olor de comida (ORN)"
                log(f"t={t:.2f}s  escape completado; sigue volando y busca olor"); marcas.append(("vuela", t))
        elif modo == "VUELA":
            # rumbo cerebral si hay olor; sin olor no hay información: explora en círculos amplios
            yaw_cmd = float(np.clip(cb.giro_cerebral(K_GIRO_AIRE), -2.5, 2.5)) if c > 0.01 else 1.2
            cerca = d_food < 3.0 and cb.tasa("DN_olor") > UMBRAL_DN_OLOR
            z_obj = 1.0
            speed = (V_VUELO if c < 0.05 else 10.0) if not cerca else 7.0   # con olor, vuela más despacio (giros más cerrados)
            if cerca:
                z_rate = -1.6
                texto = f"DN activadas por el olor ({cb.tasa('DN_olor'):.0f} Hz): decide ATERRIZAR"
            else:
                z_rate = np.clip((z_obj - gf.z) * 4, -2, 2)
                texto = f"volando; rumbo por índice lateral olfativo ({cb.indice_lateral():+.2f})" if c > 0.01 else "volando: buscando olor"
            act = pf(cuerpo.obs_volar(gf.futuro(6, CTRL_DT, speed, yaw_cmd, z_rate=z_rate)))
            cuerpo.alas_volando(ctrl, act[:11], act[11]); cuerpo.patas_retraidas(ctrl)
            if cerca and cuerpo.thorax_height() < 0.22:
                cuerpo.nivelar(); modo, t_modo = "ATERRIZA", t
                gw = Fantasma(p[:2], yaw, Z_WALK)
                log(f"t={t:.2f}s  aterrizando a {d_food:.1f} cm de la comida #{j+1} (DN_olor {cb.tasa('DN_olor'):.1f} Hz)"); marcas.append(("aterriza", t))
        elif modo == "ATERRIZA":
            if i % WALK_EVERY == 0:
                gw = Fantasma(p[:2], yaw, Z_WALK)
                walk_act = pw(cuerpo.obs_caminar(gw.futuro(65, WALK_EVERY * CTRL_DT, 0.0, 0.0)))
            cuerpo.poner_caminata(ctrl, walk_act); cuerpo.alas_plegadas(ctrl)
            texto = "patas extendidas, estabilizándose en el suelo"
            if t - t_modo > 0.5:
                modo, t_modo = "BUSCA", t; gw = Fantasma(p[:2], yaw, Z_WALK)
                log(f"t={t:.2f}s  contacto con el suelo (upright {cuerpo.upright():.2f}) -> vuelve a buscar caminando")
        elif modo == "DESCANSA":
            if i % WALK_EVERY == 0:
                walk_act = pw(cuerpo.obs_caminar(gw.futuro(65, WALK_EVERY * CTRL_DT, 0.0, 0.0)))
            cuerpo.poner_caminata(ctrl, walk_act); cuerpo.alas_plegadas(ctrl)
            texto = "saciada: descansa"
            if t - t_modo > 1.5:
                t_fin = min(t_fin, t + 0.2)
        cuerpo.probóscide(ctrl, ext)
        cuerpo.paso(ctrl)
        # ---------- video ----------
        if i % FRAME_CADA == 0:
            mundo = cuerpo.render(1, 720, 720)
            if args.rapido:
                frames.append(mundo)
            else:
                frames.append(np.concatenate([panel(cb, modo, texto, t, estim), mundo], axis=1))
        if i % 5000 == 0:
            print(f"   t={t:5.2f}s modo={modo:8s} pos=({p[0]:6.2f},{p[1]:6.2f},{p[2]:5.2f}) up={cuerpo.upright():5.2f} "
                  f"hambre={hambre:.2f} patas={cb.tasa('patas'):5.1f}Hz probo={cb.tasa('probóscide'):5.1f}Hz c={c:.2f} rumbo_comida={np.rad2deg(bearing):+4.0f} idx={cb.indice_lateral():+.3f} "
                  f"cómputo={time.time()-t0:5.0f}s", flush=True)
        i += 1

    imageio.mimsave(args.out, frames, fps=25)
    log(f"Video: {args.out} ({len(frames)} frames, {len(frames)/25:.0f} s de video, {t:.1f} s de vida) | cómputo {time.time()-t0:.0f} s")
    with open("videos/final/vida_de_mosca_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(eventos))
    # timeline
    hist = cb.historia
    if hist:
        tt = np.array([h[0] for h in hist]) / 1000
        fig, ax = plt.subplots(figsize=(13, 5))
        for k in ["patas", "probóscide", "TTMn", "GF", "DN_olor", "DNa02_L", "DNa02_R"]:
            ax.plot(tt, [h[1][k] for h in hist], color=COL[k], lw=1.2, label=k)
        ax.set_yscale("symlog", linthresh=1); ax.set_xlabel("tiempo (s)"); ax.set_ylabel("tasa media (Hz)")
        for nombre, tm in marcas:
            ax.axvline(tm, color="k", ls=":", lw=0.8); ax.text(tm, ax.get_ylim()[1] * 0.85, nombre, rotation=90, fontsize=8, ha="right", va="top")
        ax.legend(ncol=7, fontsize=8, loc="upper left"); ax.set_title("Vida de mosca: lecturas del conectoma y decisiones")
        fig.tight_layout(); fig.savefig("videos/final/vida_de_mosca_timeline.png", dpi=110)
    print("Eventos:"); print("\n".join(eventos))


if __name__ == "__main__":
    main()
