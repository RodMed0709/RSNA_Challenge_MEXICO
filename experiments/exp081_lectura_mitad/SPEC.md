# exp081_lectura_mitad

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp080_seqhead_ctrl_mil |
| Fecha | 2026-10-18 |
| Rama | exp081_lectura_mitad (desde exp080_seqhead_ctrl_mil 278dd0f: exp074/exp080 aún no están en main) |
| Estado | spec |

Idea: presupuesto de lectura/ventanas a la mitad (13/11/9 cortes, 32 ventanas) con la cabeza MIL reentrenada de exp080.
Experimento de eficiencia autorizado por Rodrigo el 18-oct ("dale prueba todas"); propuesta 2 del informe eff2.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si leo la mitad de los cortes de cada serie (cada 2.º corte de los picks del padre: 13/11/9 en vez de 26/22/18)
y evalúo con 32 ventanas en vez de 64, el kernel ahorra ~58 s de runtime en el test oculto (≈ 0.00083 de AUC al tipo de
cambio 0.001 AUC = 70 s), y la cabeza MIL reentrenada sobre esos embeddings pierde menos que eso, así que la Efficiency mejora.

**Única variable cambiada: presupuesto de lectura/ventanas** (se declara como una sola variable porque leer la mitad de cortes
sin bajar las ventanas no ahorra GPU, y bajar ventanas sin leer menos no ahorra lectura; el ahorro medido es el del par):

| Parámetro o celda | Padre (exp080) | Este experimento |
|---|---|---|
| Cortes por slot usado (celda 0 `READ_STRIDE`, celda 3 `StudyWindows`) | 26 / 22 / 18 (66) | cada 2.º: 13 / 11 / 9 (33) |
| Ventanas por estudio (`K_SEQ`, = `K_EVAL` del kernel) | 64 | 32 |

Todo lo demás igual: backbone exp074 congelado (`swa_f0.pt` en fold 0, `swa_alldata.pt` en all-data), cabeza MIL del padre
reentrenada con el protocolo de exp080 (hiperparámetros, épocas, SWA top-3 por pseudo), split `split_v1_seed42`, semilla 42,
etiquetas (0.5 teacher + 0.5 consenso), corpus `raptor96_384`.

Equivalencia con el kernel: el kernel lee `pick_indices(n, 2k)[::2]` (los mismos cortes que `corpus[slot][::2]`) y K_EVAL 32
con el mismo `_eval_centers`. Diferencia declarada: la normalización 2–98 % del kernel se calcula sobre los 13/11/9 cortes
leídos; la del corpus de entrenamiento sobre los 26/22/18 (el corpus no se reconstruye).

**Ahorro medido (antes de entrenar, bench Kaggle T4×2, 1300 estudios de train, pipeline fast idéntico salvo la variable):**
bucle principal `exp074-bench-half32` 182 s vs media de las 4 corridas fast de 64 ventanas (252, 230, 225, 253) = 240 s
⇒ **ahorro = 58 s** (varianza entre máquinas ±15 %). Informativo: con la cabeza de exp074 SIN reentrenar, gold del bench
0.9197 (half32) vs 0.9221 (64 ventanas).

**Efecto esperado:** Δ pseudo fold 0 entre −0.003 y 0; runtime −58 s.

**Criterios (no cambian después de correr):**
- `worked` si SWA pseudo f0 > 0.90184 − (58/70)·0.001 = **0.90101** (es decir Δ pseudo vs exp080 > −0.00083: la Efficiency
  estimada mejora). worked → all-data (épocas SWA de fold 0) + kernel fast + CANDIDATE.json en orq/eff2.
- `killed` si SWA gold f0 < **0.9052** (exp080 0.9252 − 0.02).
- Si no, `didnt`. Gold nunca selecciona nada.

**Presupuesto:** ≈ 0.15 GPU-h (embeddings fold 0 + all-data + cabezas) en RTX PRO 4500 ($0.72/h); tope $4.

**Riesgos:** la pérdida de AUC puede ser menor que el ruido del fold 0 (semillas de cabeza en exp080: ±0.0006), así que el
criterio se decide con una sola corrida; desajuste de normalización corpus/kernel (arriba); fuga del teacher heredada de exp074;
el ahorro de runtime es una estimación con estudios de train.

## 2. PLAN

- Celdas que cambian: 0 (parámetros: `K_SEQ` 32, `READ_STRIDE` 2, `CRIT_WORKED`/`CRIT_KILLED` de esta SPEC), 3 (`StudyWindows`:
  volumen = cortes `SUB_IDX`; tag de la caché de embeddings con el stride), 4 (con `HEAD = "mil"` el control es la misma corrida),
  5 (regla de verdict de esta SPEC), 6 (texto del criterio). Parche reproducible: `orq/eff2/e81/patch_e81.py`.
- Cadena en el pod: `SMOKE = True` → `FOLDS = [0]` → si worked: `ALL_DATA = True` con las épocas SWA de fold 0 →
  `full_mil_alldata.pt` → `kaggle_export/exp081_lectura_mitad/`.
- Kernel si worked: `exp081-half224-fast` = `exp080-ctrl224-fast` con picks `[::2]`, slots k//2 y K_EVAL 32 + checkpoint nuevo.

## 3. VERDICT (números de `results/exp081_lectura_mitad/metrics.json`)

**Resultado:** worked / didnt / killed · **Fase alcanzada:** screen / 5 folds

| Métrica | Padre | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | | | |
| gold AUC (screen: fold 0 contra fold 0 del padre) | | | |
| OOF pseudo 5 folds | | | |
| GPU-h | | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
