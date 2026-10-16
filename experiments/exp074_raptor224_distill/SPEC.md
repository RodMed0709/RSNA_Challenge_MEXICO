# exp074_raptor224_distill

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp057_raptor_eficiente |
| Fecha | 2026-10-16 |
| Rama | exp074_raptor224_distill |
| Estado | spec |

Idea: objetivo = 0.5 teacher (R7+nartaa384) + 0.5 consenso exp031

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** el modelo rápido (Raptor 224, 3 slots, sin TTA) aprende mejor imitando al ensamble 0.957 (R7 compact +
Raptor nartaa 384) que solo con nuestras etiquetas LLM de consenso: el teacher ve la imagen y da objetivos más finos y
mejor alineados con la definición del test que el informe.

**Única variable cambiada:** los objetivos de entrenamiento.

| Parámetro o celda | Padre (exp057) | Este experimento |
|---|---|---|
| Objetivo de entrenamiento (celda 2) | consenso exp031 | 0.5 × prob. teacher + 0.5 × consenso exp031 |

Teacher = media en espacio de probabilidad 0.5/0.5 de R7 compact (heliosli `rsna-knee-compact-r7cnxfsss`, 4 folds
públicos promediados como en su kernel) y Raptor nartaa 384 (`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`, con la
misma TTA que el slot Raptor del kernel `r7-raptor-repro`). El teacher se corre una vez sobre los estudios de
entrenamiento en un pod; sus probabilidades no entran a git.

Todo lo demás igual a exp057: split `split_v1_seed42`, semilla 42, 224 px, 3 slots (Sag fluid, Sag, Cor fluid), épocas,
SWA, aug espacial de exp045. Se entrena en folds 1–4 y se evalúa en fold 0 (pseudo L2 y gold contra los objetivos de
siempre, no contra el teacher).

**Efecto esperado:** Δ pseudo fold 0 ≈ +0.005 a +0.010; mismo costo de entrenamiento e inferencia que exp057.

**Criterios (fijados por Rodrigo, no cambian después de correr):**
- `worked` si pseudo SWA fold 0 ≥ 0.8939 (exp057 0.8889 + 0.005).
- `killed` si gold < 0.8896 (exp057 0.9096 − 0.02), diga lo que diga pseudo.
- Si no: `didnt`. Gold nunca elige época, checkpoint ni pesos.
- Si `worked`: all-data con los mismos objetivos + kernel = candidato a final B (eficiencia), medido en LB contra 0.945
  (exp041); no hay 5 folds antes del cierre.

**Presupuesto:** teacher ≈ 2–3 GPU-h (R7 manda) · screen fold 0 ≈ 1.2 GPU-h · all-data ≈ 1.5 GPU-h · RTX PRO 4500.

**Riesgos:**
- **Fuga del teacher:** el teacher vio el fold 0 en su propio entrenamiento (R7 entrenó con ~3/4 de casi todo el pool
  no-gold; nartaa con todo). Sus probabilidades en fold 0 son in-sample, así que la pseudo fold 0 del alumno puede salir
  inflada; el veto gold y el LB son la verificación real.
- **Licencias:** destila pesos de heliosli (R7, licencia "other") y de nartaa ("other", supervisión OAI). Antes de elegir el
  alumno como final de premio (ganadores CC BY-NC 4.0) hay que revisar las dos licencias. Código y pesos de heliosli nunca
  a git.
- Calibración: R7 da probabilidades casi binarias (1e-6 … 0.99999 en los 3 estudios visibles) y nartaa 384 da valores
  moderados (0.15 … 0.95); la media en probabilidad (fijada así) hace que R7 domine la forma del objetivo. El 0.957 los
  mezcla por rangos; no se cambia aquí (sería otra variable).
- Fidelidad del teacher: se corren los dos scripts del kernel `r7-raptor-repro` sin cambios de modelo sobre una raíz
  `/kaggle/input` simulada (R7 lee los DICOM con su propio decodificador, sin caché `px384_v4`). Smoke = los 3 estudios
  visibles de test, comparados con las salidas de Kaggle de ese kernel.

## 2. PLAN

- Celda 0: `EXP`, `PADRE`, `TEACHER_CSV` (ruta en el volumen), `TEACHER_W = 0.5`.
- Celda 2: carga `teacher.csv` (StudyInstanceUID + 12 probabilidades), revisa que cubra todos los estudios de train y
  arma el objetivo 0.5 × teacher + 0.5 × consenso. Ninguna otra celda cambia.
- Antes del notebook (fuera del repo, en el pod): inferencia del teacher sobre los 4407 estudios → `teacher_r7.csv`,
  `teacher_n384.csv`, `teacher.csv` en `results/exp074_raptor224_distill/` (no se suben).
- Smoke con `SMOKE = True`, luego `FOLDS = [0]` con `run_notebook.sh`.

## 3. VERDICT (números de `results/exp074_raptor224_distill/metrics.json`)

**Resultado:** worked / didnt / killed · **Fase alcanzada:** screen / 5 folds

| Métrica | Padre | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | | | |
| gold AUC (screen: fold 0 contra fold 0 del padre) | | | |
| OOF pseudo 5 folds | | | |
| GPU-h | | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
