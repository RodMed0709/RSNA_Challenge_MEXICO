# exp077_seqhead_gru

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp074_raptor224_distill |
| Fecha | 2026-10-17 |
| Rama | exp077_seqhead_gru |
| Estado | spec |

Idea: cabeza de secuencia (GRU unidireccional + attention-MIL) sobre embeddings de ventana del backbone exp074 congelado

Autorizado por Rodrigo el 17-oct (técnica de 1.º Lumbar 2024 Bi-LSTM + attention MIL, 1.º Abdominal 2023 GRU y 2.º
Abdominal BiLSTM). Hermano: exp078_seqhead_bigru (misma cabeza, bidireccional).

## 0. Cómo agrega hoy Raptor (exp074) y qué se reemplaza

- **Entrada:** volumen del corpus d96/384; solo slots 0–2 (Sag fluido 26 + Sag 22 + Cor fluido 18 = cortes 0–65,
  concatenados en ese orden). Cada **ventana** = 3 cortes consecutivos como RGB, recorte central 320 → 224 bilineal.
- **Ventanas por estudio:** entrenamiento K = 12 centros al azar (con reemplazo y barajados: el orden se pierde);
  evaluación del notebook K_EVAL = 24 centros por `linspace` sobre los centros válidos; kernel `exp074-student224`
  K_EVAL = 64 (todas las posiciones de los 66 cortes).
- **Backbone:** CoAtNet-2 (`coatnet_rmlp_2_rw`, img 224), un vector de 1024 por ventana (avg pool). No hay parámetros por
  slot: los slots solo deciden de qué cortes salen las ventanas.
- **Cabeza (`RaptorClassifier.head`):** LayerNorm(1024) → atención por hallazgo (Linear 1024→256, tanh, dropout 0.2,
  Linear 256→12) con softmax sobre las ventanas → 12 vectores agregados (uno por hallazgo) → logit = ⟨clsW_k, v_k⟩ + b_k.
  Es una **bolsa** (MIL invariante a permutación): no usa el orden de los cortes ni el contexto entre ventanas más allá
  de los 3 cortes de cada una.
- **Lo que se reemplaza:** entre la LayerNorm y la atención se inserta una **GRU unidireccional** (1024 → 256, 1 capa)
  que recorre la secuencia de ventanas **ordenada por corte** (Sag fluido → Sag → Cor fluido) + LayerNorm(256); la misma
  atención por hallazgo + clasificador lineal operan sobre los estados de la GRU (dim 256) en vez de sobre los embeddings.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** el contexto entre cortes (qué ventanas vecinas también se ven anormales, extensión de una lesión, paso de
una serie a otra) tiene señal que la bolsa MIL no usa; una GRU sobre la secuencia ordenada de embeddings la captura, como
en las soluciones ganadoras de Lumbar 2024 y Abdominal 2023, sin tocar el backbone ni el coste de inferencia.

**Protocolo común (idéntico para el control y para la cabeza nueva):** backbone = `swa_f0.pt` de exp074 (SWA de las
épocas 14, 15, 13; entrenado en folds 1–4) **congelado**. Se extraen una sola vez los embeddings de 64 ventanas por
estudio (exactamente las ventanas del kernel: `linspace` de 64 sobre los centros válidos de los slots 0–2, ordenadas por
corte; sin aug; bf16) para los 4407 estudios (train, fold 0 y gold). Solo se entrena la cabeza, con los hiperparámetros
de cabeza del padre: AdamW lr 1e-3, wd 0.02, OneCycle pct 0.15, 16 épocas, batch 8, BCE con el mismo pos_weight, clip 3,
semilla 42, SWA = media de las 3 mejores épocas por pseudo L2. Objetivos = los de exp074 (0.5 × teacher + 0.5 × consenso,
celda 2 sin cambios). Evaluación: pseudo L2 fold 0 y gold con las mismas 64 ventanas.

**Única variable cambiada (contra el control):**

| Parámetro o celda | Control (cabeza del padre reentrenada) | Este experimento |
|---|---|---|
| Agregación de ventanas (celda 3) | LayerNorm → attention-MIL por hallazgo | LayerNorm → GRU uni (256) → LayerNorm → attention-MIL por hallazgo |

**Control obligatorio:** en el mismo notebook se reentrena desde cero la cabeza original (`norm`, `att`, `clsW`, `clsb`)
sobre los mismos embeddings congelados con el mismo protocolo. Separa "cabeza nueva" de "backbone congelado + 64
ventanas ordenadas sin aug" (cambios del protocolo que comparten control y cabeza nueva). Informativo (no decide): la
cabeza del padre sin reentrenar aplicada a los mismos embeddings (padre con 64 ventanas, como en el kernel) y 2 semillas
extra (7, 123) de control y cabeza nueva para ver el ruido.

**Efecto esperado:** Δ pseudo fold 0 SWA ≈ +0.003 a +0.008 contra el control; inferencia igual (+ < 1 ms por estudio).

**Criterios (fijados por Rodrigo, no cambian después de correr; todo con la SWA de fold 0):**
- `worked` si pseudo SWA fold 0 ≥ **0.9037** (padre 0.8987 + 0.005) **y** > pseudo SWA fold 0 del control.
- `killed` si gold SWA fold 0 < **0.8997** (padre 0.9197 − 0.02), diga lo que diga pseudo.
- Si no: `didnt`. Gold nunca elige época, checkpoint ni hiperparámetro. Las semillas extra no deciden.
- Comparación de decisión: contra el control **y** contra el padre (0.8987 / 0.9197 de `results/exp074_raptor224_distill/metrics.json`).
- Si `worked`: all-data = backbone all-data de exp074 congelado (`kaggle_export/exp074_raptor224_distill/swa_alldata.pt`),
  embeddings de los 4349 no-gold, cabeza con SWA sobre las épocas que eligió la SWA de fold 0 → kernel = `exp074-student224`
  con la cabeza nueva → validación (COMPLETE, sin Traceback/SystemExit) → CANDIDATE.json. Pregunta LB: ¿la cabeza GRU
  mejora el alumno eficiente? `worked` en LB si ≥ 0.950 (exp074-student224 = 0.948). Si worked también exp078, el kernel
  se hace con la que tenga mayor pseudo SWA fold 0.

**Presupuesto:** extracción de embeddings ≈ 0.2 GPU-h · cabezas (control + GRU + semillas extra) ≈ 0.1 GPU-h · all-data
≈ 0.2 GPU-h · RTX PRO 4500 ($0.72/h). Tope del agente $6 (con exp078).

**Riesgos:**
- El backbone congelado sin aug puede dejar todas las cabezas por debajo del padre (aug espacial y fine-tuning conjunto
  fuera): el control lo mide; si el control queda muy abajo, la cabeza nueva difícilmente llega a 0.9037.
- Sobreajuste de la cabeza con embeddings fijos (sin variación de ventanas ni aug entre épocas); la SWA por pseudo lo
  amortigua.
- Fuga del teacher heredada de exp074 (el teacher vio fold 0): afecta igual al control y a la cabeza nueva.
- La GRU cruza las fronteras entre series (Sag fluido → Sag → Cor fluido) sin marcador de slot; añadirlo sería otra variable.
- Estudios con < 64 centros válidos repiten ventanas (como el kernel); la GRU las ve como pasos repetidos.
- Kernel: GRU en fp16 (`model.half()`) en T4; se valida en Kaggle antes de proponerlo.

## 2. PLAN

- Celda 0: `PADRE = "exp074"`, `CORPUS` = caché del volumen (`/workspace/data/cache/raptor96_384`, mismo corpus que
  `/root/corpus96_384` del padre), `HEAD = "gru"`, `BACKBONE_CKPT`, `BACKBONE_CKPT_ALLDATA`, `EMB_DIR`, `K_SEQ = 64`,
  `GRU_HIDDEN = 256`, `EMB_BATCH`, `INFO_SEEDS`, `CRIT_WORKED`, `CRIT_KILLED`.
- Celda 3: además del modelo del padre (sin cambios): extracción y caché de embeddings con el backbone congelado,
  `MILHead` (= cabeza del padre sin backbone) y `SeqGRUHead`.
- Celda 4: entrenamiento de cabezas sobre embeddings (mismo bucle, pérdida, selección y SWA del padre); control +
  cabeza nueva + semillas informativas; en all-data guarda backbone + cabeza en un checkpoint para el kernel.
- Celda 5: tabla cabeza nueva / control / padre / padre con 64 ventanas, decisión con los criterios de arriba; el bench
  de CoAtNet del padre se sustituye por el tiempo de la cabeza.
- Celda 6: `metrics.json` con `control`, `parent_k64`, `info_seeds`, `verdict_auto`.
- Celdas 1, 2 y 7 sin cambios. Smoke `SMOKE = True`, luego `FOLDS = [0]`, en cadena desatendida en el pod (`orq/seqhead/`).

## 3. VERDICT (números de `results/exp077_seqhead_gru/metrics.json`)

**Resultado:** worked / didnt / killed · **Fase alcanzada:** screen / all-data

| Métrica | Padre | Control | Este | Δ vs padre | Δ vs control |
|---|---|---|---|---|---|
| pseudo AUC fold 0 (SWA) | | | | | |
| gold AUC fold 0 (SWA) | | | | | |
| GPU-h | | | | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
