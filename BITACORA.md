# Bitácora — Proyecto Mosca 🪰🧠

Objetivo: reproducir, en una PC de casa, la idea de los proyectos que combinan el
conectoma de la mosca (mapa de Google/Janelia/Princeton) con un cuerpo simulado en 3D,
y llegar a una mosca que hace **cosas de mosca** (caminar, comer, escapar, volar) con
**su cerebro de mosca tomando las decisiones**.

Fechas: 13–14 de septiembre de 2026. Hardware: 12 núcleos CPU, GTX 1050 Ti (no se usó:
la física es CPU y las redes son pequeñas), Python 3.11 en `.venv`.

---

## Resultado principal: `videos/final/dia_de_mosca.mp4`

"Un día de mosca" — 19 s, cámara lenta. Izquierda: el cuerpo en MuJoCo. Derecha: las
motoneuronas del conectoma real en vivo (raster + tasas) y la decisión en curso.

| Escena | Estímulo que recibe el conectoma | Circuito real que responde | Lo que hace el cuerpo |
|---|---|---|---|
| EXPLORAR | Comando de marcha: 2 neuronas **DNp09** | 308 motoneuronas de patas | Camina hacia la comida (política de imitación de Janelia) |
| COMER | 275 neuronas **gustativas** de la cabeza | Motoneuronas de probóscide **MN1–MN13** | Se detiene y extiende la probóscide **en proporción a la tasa de disparo** |
| AMENAZA | 185 detectores de looming **LPLC2** | **Giant Fiber → TTMn** (músculo del salto) | Despega en el instante en que TTMn dispara |
| VOLAR | — (GF recluta las motoneuronas de vuelo DLM) | DLM/DVM | Vuela (política de imitación de vuelo) |

Latencias de decisión medidas en el conectoma (ver `videos/final/dia_de_mosca_log.txt`):
- decidir caminar tras encender DNp09: **~15 ms**
- decidir comer tras saborear: **~176 ms**
- TTMn disparando tras la amenaza visual: **~356 ms**

Gráfica de todo el día: `videos/final/dia_de_mosca_timeline.png` (estímulos arriba, tasas de
cada grupo motor abajo, con las decisiones marcadas).

### Qué es real y qué es simplificación (honestidad)
- **Real**: las 166,700 neuronas y 6.2M conexiones vienen del CSV del conectoma del macho.
  El signo de cada sinapsis viene del neurotransmisor predicho de la neurona presináptica.
  Los circuitos DNp09→patas, gusto→probóscide, LPLC2→GF→TTMn, GF→DLM emergen del cableado;
  nadie los programó.
- **Simplificación 1 — el modelo neuronal**: leaky integrate-and-fire con un único
  umbral global (200 unidades de "sinapsis") y peso = número de sinapsis. La mosca real
  tiene umbrales y ganancias distintos por neurona. Artefacto conocido: las motoneuronas
  de vuelo (DLM/DVM) se sobre-reclutan con casi cualquier estímulo (tasas de 200–300 Hz).
- **Simplificación 2 — la ejecución motora**: el conectoma **decide** (y en el caso de la
  probóscide, controla directamente la amplitud), pero los movimientos finos de caminar y
  volar los ejecutan las políticas entrenadas por imitación de moscas reales
  (Janelia/DeepMind, `data/policies/`). Es una jerarquía parecida a la biológica
  (cerebro decide, cordón nervioso ejecuta), pero la traducción motoneurona→músculo
  completa sigue siendo investigación abierta (ver Fase 3).
- **Simplificación 3 — el cambio de escena**: caminar/comer ocurren en un mismo mundo
  físico; el vuelo es otro entorno (flybody desactiva las patas en vuelo), así que hay un
  corte de cámara en el despegue.

---

## Cronología de pruebas

### Fase 0 — Instalación
- Python 3.11 + MuJoCo 3.13 + dm_control + flybody. Windows nativo funciona para todo
  menos el entrenamiento oficial (TensorFlow/Acme, solo Linux) → usamos PyTorch + SB3.
- Las políticas pre-entrenadas del paper (TF 2.8) cargan en TF 2.21 tras registrar un
  alias del tipo `Independent_ACTTypeSpec` (ver `janelia_policy.py`).

### Fase 1 — Línea base: cuerpo + red genérica (PPO), tarea "caminar sobre bola"
- `train_walk.py`, 2M pasos, ~1 h (690 pasos/s con 8 entornos).
- Recompensa media por episodio: **56 → 169**.
- Video: `videos/fases_entrenamiento/fase1_ppo_solo_cuerpo.mp4`.
- Hallazgo: **reward hacking**. La recompensa solo pide "gira la bola"; PPO rema con las
  patas delanteras y arrastra las traseras. Confirmado con `analiza_marcha.py`
  (`videos/fases_entrenamiento/marcha.png`): T1 dominan en pasos/s, T3 casi siempre en contacto.

### Fase 2 — El conectoma solo (`build_brain.py`, `sim_brain.py`, `experimento.py`)
- CSV → matriz dispersa 166,700 × 166,700 con 6.2M conexiones (63 % excitatorias,
  37 % inhibitorias; el `nt_type` del CSV de conexiones venía vacío, se tomó de `neurons.csv`).
- Escaneo de umbral: 50 = "ataque epiléptico" (28 Hz globales), 200 = régimen realista
  (~1 Hz), 800 = mudo.
- **Giant Fiber (DNp01) → TTMn**: estimulando 2 neuronas, la única motoneurona que
  responde de 815 es TTMn, la del músculo del salto. 36 spikes en todo el cerebro.
  El reflejo de escape más famoso de la neurociencia de insectos, reproducido.
  (`videos/experimentos_conectoma/exp_giant_fiber.png`)
- **MDN (moonwalker) → 311 motoneuronas** de patas y alas (`videos/experimentos_conectoma/exp_mdn.png`).
- **LPLC2 (looming) → TTMn** dispara (38 spikes) — el circuito visual de escape completo
  (`videos/experimentos_conectoma/exp_looming_TTMn.png`).
- **Gusto → probóscide**: 48 de 107 motoneuronas de cabeza responden; las más activas
  MN5, MN3L, MN9, MN7 (`videos/experimentos_conectoma/exp_gusto_proboscide.png`).
- **DNp09 → 308 motoneuronas** (`videos/experimentos_conectoma/exp_DNp09_patas.png`).

### Fase 3 — Conectoma en el lazo de control (`brain_body_env.py`, `train_brain.py`)
- Tacto de las 6 patas → 4,000 mecanosensoriales; DNp09/DNa01/DNa02 → comando "camina";
  las tasas de 815 motoneuronas se dan a la política PPO, que aprende el "decodificador"
  (el conectoma queda congelado, como reservorio).
- 136–149 pasos/s; 2M pasos en ~4 h.
- Resultado: **191** de recompensa final vs 169 de la Fase 1 (+13 %), y aprendió más
  rápido (157 a los 852k pasos vs ~120 de la línea base). Evaluación determinista de
  5 episodios: 90.3 ± 11.1 vs 80.3.
- Video: `videos/fases_entrenamiento/fase3_ppo_con_conectoma.mp4`. Mismo reward hacking (las traseras se
  arrastran): la recompensa define *qué* se aprende; el cerebro solo aceleró el *cuánto*.
- Control científico (conectoma **barajado**: mismos pesos y signos, destinos permutados):
  `train_shuffled.py`. Hallazgos previos al entrenamiento: con el mismo umbral, el cerebro
  real sostiene **22× más actividad** que el barajado (619 vs 28 spikes/paso) y el
  barajado pasa de mudo a saturado en un rango de umbral estrechísimo (percolación de red
  aleatoria) mientras el real tiene regímenes intermedios estables. Se calibró el umbral
  del barajado a 130 para igualar actividad. Resultado del entrenamiento: ver sección
  "Control barajado" al final (se completa cuando termine la corrida).

### Prueba clave — ¿los reflejos están en el cableado o en el volumen? (real vs barajado)
Mismo estímulo, mismos pesos y signos, umbral calibrado para igualar actividad global:

| Prueba | Conectoma REAL | Conectoma BARAJADO |
|---|---|---|
| Looming (LPLC2) → spikes en TTMn (salto) | **38** | **0** — el reflejo de escape desaparece |
| Gusto → motoneuronas de probóscide activadas | **11 de 61** (3.0 % de los spikes motores) | 2 de 61 (0.4 %) |
| Motoneuronas activadas por looming | 348 de 815 | 149 de 815 |

Conclusión: los circuitos que usa el demo (escape, alimentación) **no** son un artefacto de
"tener una red grande": al destruir el cableado específico de la mosca se pierden, aunque la
red conserve neuronas, pesos y signos.

### Fase 4 — Políticas de imitación de Janelia (`janelia_policy.py`)
- `walking`: recompensa de seguimiento 1.0 (perfecta) sobre trayectoria sintética;
  marcha trípode natural. `videos/pruebas_cuerpo/janelia_walking_test.mp4`.
- `flight`: 0.85 de seguimiento; vuelo estable con aleteo. `videos/pruebas_cuerpo/janelia_flight_test.mp4`.
- Truco para "comer": el entorno de caminata no expone actuadores de probóscide, pero sí
  las articulaciones (`rostrum`, `haustellum`, `labrum`); se animan por posición con
  amplitud dictada por la tasa de las motoneuronas MN del conectoma.
  `videos/pruebas_cuerpo/test_proboscide.mp4`.

---

## Cómo reproducir

```
.venv\Scripts\python dia_de_mosca.py --calibrar   # tasas por estímulo
.venv\Scripts\python dia_de_mosca.py              # video completo (~6 min)
.venv\Scripts\python experimento.py --stim tipo:LPLC2 --record tipo:TTMn --ms 500 --window 100,300
.venv\Scripts\python analiza_marcha.py 5 600      # marcha por pata de todos los modelos
.venv\Scripts\python viewer_libre.py              # mosca libre interactiva
```

## Ideas siguientes
- Mismo conectoma como "cerebro de decisiones" para otro cuerpo: un carrito que se
  estaciona, un personaje de juego (el patrón es idéntico: estímulos → conectoma → lectura).
- Reward shaping o imitación para arreglar la marcha en las Fases 1 y 3 por igual y
  repetir la comparación con marcha bonita.
- Modelo neuronal más fiel (umbrales por clase, sinapsis con retardo) para quitar el
  artefacto de las motoneuronas de vuelo.

## Fase 5 — MUNDO ABIERTO: `videos/final/vida_de_mosca.mp4` (14-sep)

Una sola escena continua. Un cuerpo con patas + alas + probóscide activas a la vez (flybody
los trae separados; `mundo_abierto_env.py` los une mapeando articulaciones y actuadores por
nombre para las dos políticas de Janelia). Izquierda del video: mapa esquemático de las
**107 regiones** (neuropilos) del cerebro y el cordón nervioso, coloreado con la actividad
real del conectoma (`cerebro_regiones.py`: cada spike se reparte en las regiones donde esa
neurona tiene sus sinapsis de salida, dato del CSV). Derecha: el mundo 3D.

Secuencia que emerge (log en `videos/final/vida_de_mosca_log.txt`):
1. **Busca**: DNp09 + olor (ORN izq/der según el ángulo a la comida) → motoneuronas de patas
   → camina hacia el olor. Rumbo: índice lateral olfativo del conectoma (ver más abajo).
2. **Come**: al llegar, gusto → MN de probóscide suben de ~14 Hz (sólo olor) a ~28 Hz → decide
   comer; extensión de la probóscide = tasa (llega a ~85 Hz).
3. **Amenaza**: aparece un depredador (esfera) que se acerca; su tamaño angular estimula LPLC2
   izq/der → Giant Fiber → **TTMn dispara ~800 ms después** → salto y vuelo de escape.
4. **Vuela** alejándose; en el aire el índice lateral olfativo del conectoma la guía a la segunda comida.
5. **Aterriza** cuando las neuronas descendentes activadas por el olor (DNb05, DNg56, DNp12…)
   superan el umbral cerca de la comida; nivela, extiende patas, camina y come. Saciada: descansa.

Simplificaciones (honestas): salto de despegue y nivelación al aterrizar son "teletransportes"
cortos de postura (las políticas de Janelia no incluyen despegue ni aterrizaje); el zigzag/círculos
cuando no hay olor son heurísticos; el mapa de regiones es un esquema anatómico, no una malla
3D; el modelo LIF de umbral único sobre-recluta las motoneuronas de vuelo.

Pruebas intermedias del mundo abierto: `test_mundo_caminar.mp4` (caminata con alas puestas —
falló hasta corregir el orden de `actuator_activation`), `test_mundo_volar.mp4`,
`test_mundo_despegue_aterrizaje.mp4` (el aterrizaje volcaba por velocidad angular residual y
alas abiertas; se resolvió plegándolas al tocar tierra), `vida_rapido.mp4` (integración sin panel).

### ¿El cerebro dirige o sólo permite? — navegación 100 % cerebral (14-sep, tarde)
El usuario preguntó si la dirección hacia la comida la decidía el cerebro o el guion. Respuesta
honesta en la primera versión: el guion (rumbo = ángulo al olor; DNa02 sólo lo modulaba). Se
intentó quitar la heurística por completo:

1. **Neuronas descendentes por lado**: no llevan información de lado del olor. El lado derecho
   responde más siempre (índice −0.13), venga el olor por la antena izquierda o la derecha.
   Anatómicamente razonable: los lóbulos antenales se comunican y la lateralidad se diluye.
2. **Etapas tempranas** (neuronas de proyección del lóbulo antenal ALPN + células de Kenyon): sí
   hay señal lateral, monótona con la proporción izq/der del olor (índice +0.172 con todo el
   olor a la izquierda → +0.063 con todo a la derecha), pero con dos defectos: un **sesgo de
   base** (+0.136 con olor simétrico: el conectoma simulado es asimétrico) que además **depende
   de la intensidad** (+0.164 con olor débil → +0.063 con olor fuerte), y **saturación**: con olor
   muy intenso la evidencia (L+R)/2 se aplana (80→115 Hz para 16× más olor).
3. Solución: `giro = K · (índice − sesgo(evidencia))`, con el sesgo calibrado al inicio con olor
   simétrico a 5 intensidades (una calibración del sensor, no un rumbo), y la intensidad de las
   ORN limitada (escala 100 Hz en tierra, 120 en el aire) para no entrar en saturación salvo a
   milímetros de la comida. La marcha sólo avanza si las motoneuronas de patas superan el umbral.
4. Resultado (`test_navegacion_cerebral.py`, comida a 5 cm): con K=40/escala 200 se acercaba a
   0.7–1.5 cm y luego orbitaba (saturación cerca). Con K=110/escala 100: **6 de 6 llegadas** desde
   0°, ±60°, ±120° y 180° (2.5–7.0 s). En vuelo hizo falta una pluma de olor de más alcance
   (λ=12 cm), escala 120 y volar más despacio con olor (10 cm/s); con escala 250 la red se
   saturaba y la mosca orbitaba a 10 cm de la comida.
5. Escena final (`vida_de_mosca.py`): **sin ninguna instrucción de dirección**. Llegó a la comida
   #1 en 2.96 s (más rápido que con la heurística), escapó del depredador, voló guiada por el
   índice lateral hasta la comida #2 (aterrizó a 1.2 cm), caminó hasta ella y comió.

Lo que sigue siendo del guion: el mundo y sus eventos, la codificación sensorial (qué neuronas
estimula cada estímulo y a qué frecuencia), qué poblaciones se leen y sus umbrales, la máquina
de estados que traduce lecturas en comportamientos, el zigzag/círculos cuando **no hay olor** (sin
información el cerebro no puede dirigir), la ejecución motora (políticas de Janelia) y los
"teletransportes" cortos de despegue/aterrizaje. Lo que es del cerebro: que cada circuito
responda como responde, cuándo lo hace (latencias), cuánto (la probóscide), **hacia dónde** ir
cuando hay olor, y qué regiones se encienden.

Hallazgo de paso: el olor por la antena izquierda activa DNa02 izquierda (24 spikes vs 0 derecha),
pero el lado derecho apenas responde (1 vs 1): DNa02 solo no sirve para navegar en este modelo.

## Control barajado — resultado (14-sep, corrida nocturna de 2M pasos)

| Modelo (misma tarea, mismo PPO, 2M pasos) | Recompensa final de entrenamiento | Evaluación (5 ep. determinista) |
|---|---|---|
| Fase 1: solo cuerpo | 169 | 80.3 |
| Fase 3: conectoma **real** | **191** | **93.8 ± 8.7** |
| Control: conectoma **barajado** | 147 | 91.3 ± 1.9 |

Lectura honesta:
- Durante el entrenamiento, el orden es claro: real (191) > cuerpo solo (169) > barajado (147).
  El cableado real aceleró y elevó el aprendizaje; el barajado, con la misma cantidad de
  actividad, lo **empeoró** respecto a no tener cerebro.
- En la evaluación determinista, real y barajado quedan parecidos (93.8 vs 91.3; la diferencia
  cabe en el ruido) y ambos por encima de la línea base. Es decir: parte de la ventaja de la
  Fase 3 sí es "efecto reservorio" (tener memoria recurrente extra ayuda a la política), y la
  parte atribuible al cableado real se ve sobre todo en la velocidad/techo del entrenamiento.
- Donde el cableado real es inequívocamente distinto es en los **reflejos específicos** (tabla
  "real vs barajado" arriba): looming→TTMn y gusto→probóscide desaparecen al barajar. Para la
  tarea de la bola, la política aprendió a explotar cualquier dinámica disponible; para
  comportamientos de mosca (el demo), sólo el cableado real los tiene.
- Marcha por pata de los tres modelos: `videos/fases_entrenamiento/marcha_3_brazos.png` (todos arrastran las
  traseras: la recompensa, no el cerebro, decide el estilo de marcha).
