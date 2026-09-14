"""Análisis objetivo de marcha: ¿qué tanto usa cada pata cada modelo?

Corre N episodios por modelo y mide, por pata:
  - % de tiempo en contacto con la bola (alto + pocos pasos = arrastre)
  - pasos por segundo (transiciones aire->contacto; el "ritmo" de la pata)

Uso:
    .venv\\Scripts\\python analiza_marcha.py [episodios] [pasos_por_episodio]
    (defaults: 5 episodios de 600 pasos)
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

LEGS = ["T1 izq\n(delantera)", "T1 der\n(delantera)", "T2 izq\n(media)",
        "T2 der\n(media)", "T3 izq\n(trasera)", "T3 der\n(trasera)"]
DT = 0.002  # timestep de control aprox (s)


def evaluar(nombre, model_path, env_maker, episodes, steps):
    print(f"\n[{nombre}] evaluando {model_path} ({episodes} episodios x {steps} pasos)...")
    model = PPO.load(model_path)
    env = env_maker()
    touch_all, rewards = [], []
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_touch, ep_rew = [], 0.0
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            ep_touch.append(np.asarray(obs["walker/touch"]) > 0)
            ep_rew += r
            if term or trunc:
                break
        touch_all.append(np.array(ep_touch))
        rewards.append(ep_rew)
        print(f"  episodio {ep+1}: reward {ep_rew:.1f}")
    touch = np.concatenate(touch_all)
    contact = touch.mean(0) * 100  # % de tiempo en contacto
    # pasos por segundo: transiciones 0->1
    trans = np.concatenate([np.diff(t.astype(int), axis=0) > 0 for t in touch_all])
    dur_s = len(touch) * DT
    step_rate = trans.sum(0) / dur_s
    print(f"  reward medio: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    return dict(contact=contact, step_rate=step_rate,
                reward=np.mean(rewards), reward_sd=np.std(rewards))


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 600

    import glob, os
    from fly_env import make_env
    from brain_body_env import make_brain_env

    modelos = []
    if os.path.exists("modelos/walk_ppo.zip"):
        modelos.append(("Fase 1: solo cuerpo", "modelos/walk_ppo.zip", make_env))
    bs = glob.glob("checkpoints_brain/brain_*.zip") + glob.glob("modelos/brain_ppo.zip")
    if bs:
        best = max(bs, key=os.path.getmtime)
        modelos.append(("Fase 3: conectoma", best, make_brain_env))
    br = glob.glob("checkpoints_shuffled/shuf_*.zip") + glob.glob("modelos/shuffled_ppo.zip")
    if br:
        best = max(br, key=os.path.getmtime)
        from brain_body_env_shuffled import make_shuffled_env  # opcional, si existe
        modelos.append(("Control: barajado", best, make_shuffled_env))

    resultados = {n: evaluar(n, p, m, episodes, steps) for n, p, m in modelos}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(6)
    w = 0.8 / len(resultados)
    for i, (nombre, r) in enumerate(resultados.items()):
        off = (i - (len(resultados) - 1) / 2) * w
        lbl = f"{nombre} (reward {r['reward']:.0f}±{r['reward_sd']:.0f})"
        axes[0].bar(x + off, r["contact"], w, label=lbl)
        axes[1].bar(x + off, r["step_rate"], w, label=lbl)
    for ax, title, ylab in [(axes[0], "% del tiempo en contacto con la bola\n(alto + pocos pasos = arrastre)", "% contacto"),
                            (axes[1], "Pasos por segundo\n(ritmo de cada pata)", "pasos/s")]:
        ax.set_xticks(x); ax.set_xticklabels(LEGS, fontsize=8)
        ax.set_title(title); ax.set_ylabel(ylab); ax.legend(fontsize=8)
    fig.suptitle("Análisis de marcha por pata")
    fig.tight_layout()
    fig.savefig("videos/fases_entrenamiento/marcha.png", dpi=110)
    print("\nFigura guardada: marcha.png")


if __name__ == "__main__":
    main()
