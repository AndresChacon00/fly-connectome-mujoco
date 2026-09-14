"""UN DÍA DE MOSCA — el conectoma decide, el cuerpo ejecuta.

Arquitectura jerárquica (como la mosca real):
  * CEREBRO (decisiones): el conectoma completo (166,700 neuronas reales) simulado
    con spikes. Recibe estímulos de la escena y sus motoneuronas dictan qué hacer.
  * CUERPO (ejecución): el modelo físico flybody en MuJoCo, movido por las políticas
    de caminata y vuelo entrenadas por imitación de moscas reales (Janelia/DeepMind).

Escenas y circuitos reales usados:
  1. EXPLORAR  comando de marcha DNp09 -> motoneuronas de patas -> camina hacia la comida
  2. COMER     neuronas gustativas -> motoneuronas de probóscide (MN1-13) -> extensión
               de la probóscide PROPORCIONAL a la actividad neuronal
  3. AMENAZA   detectores de looming LPLC2 -> Giant Fiber -> TTMn (músculo del salto)
               -> despegue en cuanto TTMn dispara
  4. VOLAR     Giant Fiber también recluta motoneuronas de vuelo (DLM); política de vuelo

Uso:
    .venv\\Scripts\\python dia_de_mosca.py --calibrar     # mide tasas para fijar umbrales
    .venv\\Scripts\\python dia_de_mosca.py                # genera videos/dia_de_mosca.mp4
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

from sim_brain import BrainSim

WALK_DT_MS = 2.0      # control timestep de caminata (ms)
FLY_DT_MS = 0.2       # control timestep de vuelo (ms)
RATE_TAU_MS = 40.0    # suavizado de tasas (ms)
RASTER_MS = 600       # ventana del raster en el panel

# Umbrales de decisión (Hz medios de la población). Se fijan con --calibrar.
UMBRAL_PATAS = 1.5        # calibrado: DNp09 lleva las patas a ~5.5 Hz; reposo 0 Hz
UMBRAL_PROBOSCIDE = 6.0   # calibrado: gusto -> ~12 Hz; sin sabor (sólo marcha) ~2 Hz

EVENTOS = []  # (nombre, t_ms del cerebro) para la gráfica final

ESTADOS = {
    "EXPLORAR": ("Comando de marcha DNp09 activo", "tab:blue"),
    "COMER": ("Sabor detectado: neuronas gustativas activas", "tab:green"),
    "AMENAZA": ("¡Objeto que se acerca! LPLC2 -> Giant Fiber", "tab:red"),
    "VOLAR": ("Escape: TTMn disparó, vuelo activo", "tab:purple"),
}


class CerebroDeEscena:
    """El conectoma + las poblaciones que usamos como sentidos y como lectura."""

    def __init__(self, seed=0):
        self.sim = BrainSim(threshold=200.0, seed=seed)
        neu = self.sim.neurons
        tipo = neu["Primary Cell Type"].astype(str)
        sup = neu["Super Class"]

        def idx(mask):
            return self.sim.idx_of(neu[mask]["Root ID"])

        # --- entradas (sentidos / comandos) ---
        self.in_marcha = idx(tipo.str.contains("DNp09", na=False))
        self.in_gusto = idx((neu["Class"] == "gustatory") & (sup == "cb_sensory"))
        self.in_looming = idx(tipo == "LPLC2")
        mech = idx(neu["Class"].isin(["mechanosensory_tactile", "mechanosensory_proprioceptive"])).copy()
        np.random.default_rng(42).shuffle(mech)
        self.in_tacto = np.array_split(mech, 6)

        # --- salidas (motoneuronas que leemos) ---
        es_vuelo = tipo.str.contains("DLM|DVM|TTM", na=False)
        self.out_patas = idx((sup == "vnc_motor") & ~es_vuelo)
        self.out_probo = idx((sup == "cb_motor") & tipo.str.startswith("MN"))
        self.out_alas = idx(tipo.str.contains("DLM|DVM", na=False))
        self.out_ttmn = idx(tipo == "TTMn")
        self.out_gf = idx(tipo.str.contains("DNp01", na=False))
        self.grupos = {"patas": self.out_patas, "probóscide": self.out_probo,
                       "alas": self.out_alas, "TTMn": self.out_ttmn, "GF": self.out_gf}

        self.rates = {k: np.zeros(len(v), dtype=np.float32) for k, v in self.grupos.items()}
        self.decay = np.exp(-1.0 / RATE_TAU_MS)
        self.spikes = np.zeros(self.sim.n, dtype=bool)
        self.t_ms = 0
        # raster: para cada grupo, lista de (t, neurona_local)
        self.raster = {k: [] for k in self.grupos}
        self.ttmn_ultimo_spike = -1e9
        self.historia = []  # (t_ms, {grupo: Hz}, estímulos) cada 5 ms, para la gráfica final

    def paso_ms(self, marcha=False, gusto=False, looming=False, tacto=None):
        """Un milisegundo de cerebro con los estímulos indicados."""
        s = self.spikes.copy()
        forzar = []
        if marcha:
            forzar.append(self.in_marcha[self.sim.rng.random(len(self.in_marcha)) < 0.15])
        if gusto:
            forzar.append(self.in_gusto[self.sim.rng.random(len(self.in_gusto)) < 0.10])
        if looming:
            forzar.append(self.in_looming[self.sim.rng.random(len(self.in_looming)) < 0.20])
        if tacto is not None:
            for g, touch in zip(self.in_tacto, tacto):
                p = float(np.clip(touch, 0, 1)) * 0.10
                if p > 0:
                    forzar.append(g[self.sim.rng.random(len(g)) < p])
        if forzar:
            s[np.concatenate(forzar)] = True
        self.spikes = self.sim.step(spikes_in=s)
        self.t_ms += 1
        for k, ids in self.grupos.items():
            sp = self.spikes[ids]
            self.rates[k] = self.decay * self.rates[k] + (1 - self.decay) * sp * 1000.0
            if sp.any():
                for j in np.flatnonzero(sp):
                    self.raster[k].append((self.t_ms, j))
        if self.spikes[self.out_ttmn].any():
            self.ttmn_ultimo_spike = self.t_ms
        if self.t_ms % 5 == 0:
            self.historia.append((self.t_ms, {k: self.tasa(k) for k in self.grupos},
                                  dict(marcha=marcha, gusto=gusto, looming=looming)))
        # recortar rasters viejos
        for k in self.raster:
            self.raster[k] = [(t, j) for (t, j) in self.raster[k] if t > self.t_ms - RASTER_MS]

    def tasa(self, grupo):
        return float(self.rates[grupo].mean())

    def ttmn_disparo_reciente(self, ventana_ms=30):
        return (self.t_ms - self.ttmn_ultimo_spike) < ventana_ms


# ----------------------------------------------------------------------------
def calibrar():
    print("Calibración: tasas medias de cada grupo motor bajo cada estímulo (400 ms c/u)")
    cb = CerebroDeEscena()
    for nombre, kw in [("reposo", {}), ("marcha DNp09", {"marcha": True}),
                       ("gusto", {"gusto": True}), ("looming LPLC2", {"looming": True})]:
        for _ in range(400):
            cb.paso_ms(**kw)
        print(f"  {nombre:14s}: " + " | ".join(f"{g} {cb.tasa(g):6.2f} Hz" for g in cb.grupos)
              + f" | TTMn spikes recientes: {cb.ttmn_disparo_reciente(400)}")
        for _ in range(300):  # dejar que se apague
            cb.paso_ms()


# ----------------------------------------------------------------------------
def panel(cb, estado, t_sim_ms, extra="", size=(560, 480)):
    """Dibuja el panel del cerebro como imagen RGB."""
    w, h = size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor("#111")
    desc, color = ESTADOS[estado]

    ax_t = fig.add_axes([0.02, 0.86, 0.96, 0.12]); ax_t.axis("off")
    ax_t.text(0, 0.7, estado, color=color, fontsize=22, fontweight="bold", va="center")
    ax_t.text(0, 0.1, desc, color="#ddd", fontsize=9, va="center")
    ax_t.text(1, 0.7, f"t = {t_sim_ms/1000:5.2f} s", color="#aaa", fontsize=10, ha="right", va="center")
    if extra:
        ax_t.text(1, 0.1, extra, color="#ffd54f", fontsize=9, ha="right", va="center")

    # raster
    ax_r = fig.add_axes([0.12, 0.36, 0.86, 0.43])
    ax_r.set_facecolor("#181818")
    grupos = [("patas", 40, "#4fc3f7"), ("probóscide", 30, "#81c784"),
              ("alas", 20, "#ce93d8"), ("TTMn", 2, "#ff7043"), ("GF", 2, "#ffd54f")]
    y0 = 0
    yticks, ylabels = [], []
    for g, nmax, col in grupos:
        n = min(nmax, len(cb.grupos[g]))
        pts = [(t, j) for (t, j) in cb.raster[g] if j < n]
        if pts:
            ts, js = zip(*pts)
            ax_r.scatter(np.array(ts) - cb.t_ms, np.array(js) + y0, s=3, c=col, marker="|")
        ax_r.axhspan(y0 - 0.5, y0 + n - 0.5, color=col, alpha=0.05)
        yticks.append(y0 + n / 2); ylabels.append(g)
        y0 += n + 2
    ax_r.set_xlim(-RASTER_MS, 0); ax_r.set_ylim(-1, y0)
    ax_r.set_yticks(yticks); ax_r.set_yticklabels(ylabels, color="#ccc", fontsize=8)
    ax_r.set_xlabel("ms (últimos %d)" % RASTER_MS, color="#aaa", fontsize=8)
    ax_r.tick_params(colors="#aaa", labelsize=7)
    for sp in ax_r.spines.values():
        sp.set_color("#444")
    ax_r.set_title("Motoneuronas del conectoma (spikes)", color="#ccc", fontsize=9)

    # barras de tasa
    ax_b = fig.add_axes([0.12, 0.06, 0.86, 0.22])
    ax_b.set_facecolor("#181818")
    names = ["patas", "probóscide", "alas", "TTMn", "GF"]
    vals = [cb.tasa(g) for g in names]
    cols = [c for _, _, c in grupos]
    ax_b.bar(names, [max(v, 0.1) for v in vals], color=cols)
    ax_b.axhline(UMBRAL_PATAS, color="#4fc3f7", ls="--", lw=0.8, alpha=0.6)
    ax_b.axhline(UMBRAL_PROBOSCIDE, color="#81c784", ls="--", lw=0.8, alpha=0.6)
    ax_b.set_yscale("log"); ax_b.set_ylim(0.1, 1000)
    for x, v in enumerate(vals):
        if v >= 0.1:
            ax_b.text(x, v * 1.15, f"{v:.0f}", color="#ccc", fontsize=7, ha="center")
    ax_b.set_ylabel("Hz (log)", color="#aaa", fontsize=8)
    ax_b.tick_params(colors="#ccc", labelsize=8)
    for sp in ax_b.spines.values():
        sp.set_color("#444")

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def componer(render, pan):
    return np.concatenate([render, pan], axis=1)


# ----------------------------------------------------------------------------
def escena_caminar_y_comer(cb, frames, log):
    from flybody.fly_envs import walk_imitation
    from flybody.tasks.trajectory_loaders import constant_speed_trajectory
    from janelia_policy import load_policy

    policy = load_policy("walking")

    # 1) El cerebro decide caminar: medimos cuánto tarda en pasar el umbral
    t0 = cb.t_ms
    log("Escena 1: encendiendo comando de marcha DNp09...")
    while cb.tasa("patas") < UMBRAL_PATAS and cb.t_ms - t0 < 2000:
        cb.paso_ms(marcha=True)
    latencia = cb.t_ms - t0
    log(f"  motoneuronas de patas cruzaron {UMBRAL_PATAS} Hz a los {latencia} ms "
        f"(tasa {cb.tasa('patas'):.2f} Hz) -> DECISIÓN: caminar")
    EVENTOS.append(("decide caminar", cb.t_ms))

    # Trayectoria: espera lo que tardó el cerebro, camina 2.4 cm, se detiene (comer+amenaza)
    n_espera = int(latencia / WALK_DT_MS) + 50
    n_camina = 600
    n_quieta = 1300
    qp1, qv1 = constant_speed_trajectory(n_steps=n_espera, speed=0.0, control_timestep=0.002)
    qp2, qv2 = constant_speed_trajectory(n_steps=n_camina, speed=2.0, control_timestep=0.002)
    qp3 = np.repeat(qp2[-1:], n_quieta + 80, axis=0); qv3 = np.zeros((n_quieta + 80, 6))
    qpos = np.concatenate([qp1, qp2, qp3]); qvel = np.concatenate([qv1, qv2, qv3])
    # 1er tramo: el cerebro ya decidió, así que el tramo de espera sólo simula la latencia.
    env = walk_imitation()
    env.task._traj_generator.set_next_trajectory(qpos, qvel)
    ts = env.reset()
    qp = env.physics.data.qpos
    n_total = n_espera + n_camina + n_quieta
    paso_llegada = n_espera + n_camina
    paso_amenaza = paso_llegada + 700
    estado = "EXPLORAR"
    t_gusto_on = None; t_probo_on = None; t_loom_on = None
    ext = 0.0
    for i in range(n_total):
        a = policy(ts.observation)
        touch = np.asarray(ts.observation["walker/touch"])
        # --- estímulos que la escena le da al cerebro ---
        marcha = i < paso_llegada
        gusto = i >= paso_llegada
        looming = i >= paso_amenaza
        for _ in range(int(WALK_DT_MS)):
            cb.paso_ms(marcha=marcha, gusto=gusto, looming=looming, tacto=touch)
        # --- decisiones leídas del cerebro ---
        if gusto and t_gusto_on is None:
            t_gusto_on = cb.t_ms; log(f"  t={cb.t_ms} ms: llegó a la comida; estímulo gustativo ON")
        if gusto and estado == "EXPLORAR" and cb.tasa("probóscide") > UMBRAL_PROBOSCIDE:
            estado = "COMER"; t_probo_on = cb.t_ms
            EVENTOS.append(("decide comer", cb.t_ms))
            log(f"  t={cb.t_ms} ms: motoneuronas de probóscide > {UMBRAL_PROBOSCIDE} Hz "
                f"({latencia_str(t_probo_on - t_gusto_on)}) -> DECISIÓN: comer")
        if looming and t_loom_on is None:
            t_loom_on = cb.t_ms; estado = "AMENAZA"
            log(f"  t={cb.t_ms} ms: estímulo looming (LPLC2) ON")
        if estado == "AMENAZA" and cb.ttmn_disparo_reciente():
            EVENTOS.append(("TTMn: despegue", cb.t_ms))
            log(f"  t={cb.t_ms} ms: ¡TTMn disparó! ({latencia_str(cb.t_ms - t_loom_on)} tras el looming) "
                f"-> DECISIÓN: despegar")
            frames.append(componer(env.physics.render(camera_id="walker/hero", width=640, height=480),
                                   panel(cb, "AMENAZA", cb.t_ms, extra="¡TTMn! -> SALTO")))
            for _ in range(8):
                frames.append(frames[-1])
            return {"latencia_marcha_ms": latencia,
                    "latencia_comer_ms": (t_probo_on - t_gusto_on) if t_probo_on else None,
                    "latencia_escape_ms": cb.t_ms - t_loom_on}
        # --- el cerebro mueve la probóscide: extensión proporcional a la tasa ---
        if estado in ("COMER", "AMENAZA"):
            objetivo = np.clip(cb.tasa("probóscide") / (3 * UMBRAL_PROBOSCIDE), 0, 1)
            ext = 0.9 * ext + 0.1 * objetivo
            qp[10] = -1.2 * ext; qp[12] = -1.5 * ext; qp[13] = qp[14] = 1.0 * ext
        ts = env.step(a)
        if ts.last():
            log(f"  (episodio del cuerpo terminó en paso {i})"); break
        if i % 4 == 0:
            cam = "walker/hero" if estado != "EXPLORAR" else 1
            extra = f"extensión probóscide {ext*100:3.0f}%" if estado == "COMER" else ""
            frames.append(componer(env.physics.render(camera_id=cam, width=640, height=480),
                                   panel(cb, estado, cb.t_ms, extra=extra)))
    return {"latencia_marcha_ms": latencia, "latencia_comer_ms": None, "latencia_escape_ms": None}


def escena_volar(cb, frames, log):
    from flybody.fly_envs import flight_imitation
    from flybody.tasks.trajectory_loaders import constant_speed_trajectory
    from janelia_policy import load_policy

    policy = load_policy("flight")
    env = flight_imitation()
    n = 5000  # 1 s de vuelo
    qpos, qvel = constant_speed_trajectory(n_steps=n + 20, speed=20, init_pos=(0, 0, 1),
                                           body_rot_angle_y=-47.5, control_timestep=0.0002)
    env.task._traj_generator.set_next_trajectory(qpos, qvel)
    ts = env.reset()
    log("Escena 4: vuelo (política de imitación de vuelo); leyendo motoneuronas de vuelo DLM/DVM")
    acc = 0.0
    for i in range(n):
        a = policy(ts.observation)
        acc += FLY_DT_MS
        while acc >= 1.0:
            cb.paso_ms(looming=(i < 500))  # el estímulo se apaga al despegar
            acc -= 1.0
        ts = env.step(a)
        if ts.last():
            log(f"  (vuelo terminó en paso {i})"); break
        if i % 40 == 0:
            frames.append(componer(env.physics.render(camera_id=1, width=640, height=480),
                                   panel(cb, "VOLAR", cb.t_ms, extra="cámara lenta 8x")))
    log(f"  tasa alas al final: {cb.tasa('alas'):.2f} Hz")


def latencia_str(ms):
    return f"{ms} ms" if ms is not None else "?"


def grafica_timeline(cb, eventos, out="videos/final/dia_de_mosca_timeline.png"):
    """Actividad de cada grupo motor a lo largo de todo el 'día', con los estímulos."""
    t = np.array([h[0] for h in cb.historia]) / 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1, 4]})
    est = {k: np.array([h[2][k] for h in cb.historia]) for k in ["marcha", "gusto", "looming"]}
    for i, (k, col) in enumerate([("marcha", "tab:blue"), ("gusto", "tab:green"), ("looming", "tab:red")]):
        axes[0].fill_between(t, i, i + est[k] * 0.9, color=col, step="post", alpha=0.8)
    axes[0].set_yticks([0.45, 1.45, 2.45]); axes[0].set_yticklabels(["DNp09 (marcha)", "gusto", "looming LPLC2"])
    axes[0].set_title("Estímulos que la escena le dio al conectoma")
    cols = {"patas": "#039be5", "probóscide": "#43a047", "alas": "#8e24aa", "TTMn": "#f4511e", "GF": "#fbc02d"}
    for k, col in cols.items():
        axes[1].plot(t, [h[1][k] for h in cb.historia], color=col, lw=1.4, label=k)
    axes[1].set_yscale("symlog", linthresh=1.0); axes[1].set_ylabel("tasa media (Hz)")
    axes[1].set_xlabel("tiempo del cerebro (s)")
    for nombre, tm in eventos:
        axes[1].axvline(tm / 1000, color="k", ls=":", lw=0.9)
        axes[1].text(tm / 1000, axes[1].get_ylim()[1] * 0.9, nombre, rotation=90, fontsize=8, va="top", ha="right")
    axes[1].legend(loc="upper left", ncol=5, fontsize=9)
    axes[1].set_title("Respuesta de las motoneuronas del conectoma (decisiones)")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrar", action="store_true")
    ap.add_argument("--out", default="videos/final/dia_de_mosca.mp4")
    args = ap.parse_args()
    if args.calibrar:
        calibrar(); return

    os.makedirs("videos", exist_ok=True)
    lineas = []

    def log(s):
        print(s); lineas.append(s)

    t0 = time.time()
    cb = CerebroDeEscena()
    frames = []
    res = escena_caminar_y_comer(cb, frames, log)
    escena_volar(cb, frames, log)
    imageio.mimsave(args.out, frames, fps=25)
    grafica_timeline(cb, EVENTOS)
    log(f"Video: {args.out} ({len(frames)} frames, {len(frames)/25:.0f} s) | cómputo {time.time()-t0:.0f} s")
    log("Gráfica: videos/dia_de_mosca_timeline.png")
    with open("videos/final/dia_de_mosca_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    print(res)


if __name__ == "__main__":
    main()
