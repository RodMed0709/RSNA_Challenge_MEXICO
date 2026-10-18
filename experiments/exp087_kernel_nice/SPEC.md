# exp087_kernel_nice

| Campo | Valor |
|---|---|
| Autor | rodrigo (agente eff3a) |
| Padre | exp084_kernel_rapido (variante `x`, X1–X4) |
| Fecha | 2026-10-18 |
| Rama | exp087_kernel_nice |
| Estado | didnt |

Idea: los procesos de prep tempranos (X1) con prioridad baja (`os.nice(10)`) para no frenar import/hash/carga del modelo.

## 1. SPEC (antes de tocar el notebook)

**Tipo:** PRODUCTO / EFICIENCIA, predicciones iguales. Sin entrenamiento.

**Motivo (visto en exp084, corridas `xb` v1 y `x` v1, antes de fijar esta SPEC):** con X1 los 6 procesos de prep ocupan los 4 vCPU
mientras el proceso principal importa torch/timm, hace el hash y carga los modelos: `main()` empezó a los 71 s del
notebook (padre: 29 s) y los modelos quedaron listos a los ~88 s, así que la GPU (que pasa a ser el cuello: cola llena,
espera GPU ≈ 0) arrancó ~55 s tarde y el total fue 297 s (`x`: 299 s; padre 272 / 268 s). Además `xb` dio max|Δp| 1.37e-3 y `x` 1.34e-3 (ver abajo).

**Única variable cambiada (contra exp084 `x`):**

| Parámetro | exp084 `x` | Este experimento |
|---|---|---|
| Prioridad de los procesos de prep tempranos | normal | `os.nice(10)` al iniciar cada proceso (X6) |

Todo lo demás igual a exp084 `x` (X1–X4, sin cudnn.benchmark), **con una corrección de X4 hecha antes de correr nada de
exp087**: exp084 `x` v1 dio max|Δp| 1.34e-3 (misma deriva que `xb`, luego no es cudnn.benchmark): la conversión uint8 →
float por división en CUDA es una multiplicación por el recíproco (deriva de ulp que el fp16 amplifica). En exp087 X4 vuelve
a la LUT del padre (`L0[tri.long()]`) sobre el recorte; los valores son los del padre, mismo generador `gen_bench3.py` (variante `xn`), mismos
1300 estudios de train, misma imagen docker, T4×2.

**Criterios (fijos, no cambian después de correr):** contra el padre del kernel, exp080-bench-fastng6 (media de sus
corridas v1 271.97 s y v2 de hoy), métrica `nb_elapsed_to_end`.
- `worked` si en 2 corridas de `xn`: max|Δ prob| ≤ 1e-3 contra raw_probs de exp080-bench-fastng6 v1, sin Traceback ni
  FALLBACK, y media `nb_elapsed_to_end` ≤ media del padre − **30 s**.
- `killed` si falla Δ prob o hay Traceback/FALLBACK.
- Si no, `didnt`. Informativo: comparación con exp084 `x` (efecto aislado de X6).
- Si `worked`: kernel privado `exp087-ctrl224-xfast` (= exp080-ctrl224-fast + X1–X4 + X6), validación COMPLETE sin
  Traceback; criterio LB fijado ahora: LB = 0.949 (mismas predicciones; la ganancia es solo runtime).

**Presupuesto:** 0 $ RunPod; 2 sesiones GPU Kaggle de ~6 min.

## 2. PLAN

- `fastx.py` / `gen_bench3.py` de exp084 con el parámetro `nice` (variante `xn`); copia en esta carpeta.
- Notebook `exp087_kernel_nice.ipynb` = notebook de análisis de exp084 con `DECIDE_RUNS` = corridas `xn`.

## 3. VERDICT (números de `results/exp087_kernel_nice/metrics.json`)

**Resultado:** didnt · **Fase alcanzada:** bench (2 corridas)

| Métrica (bench Kaggle T4×2, 1300 estudios de train) | Padre exp080-bench-fastng6 v1 / v2 | `xn` v1 | `xn` v2 | exp084 `x` (info) |
|---|---|---|---|---|
| `nb_elapsed_to_end` (s) | 272.0 / 268.2 (media 270.1) | 253.9 | 342.5 | 298.6 |
| media `xn` − padre | | **+28.1 s** (criterio ≤ −30 s) | | +28.5 |
| `main_total_s` (s) | 251.7 / 244.2 | 213.1 | 303.1 | 240.0 |
| arranque hasta `main()` (s) | 20.3 / 23.9 | 40.8 | 39.4 | 58.6 |
| prep por estudio, pared / CPU (s) | 1.00 / 0.48 · 0.78 / 0.49 | 1.03 / 0.42 | 1.50 / 0.41 | — |
| espera GPU total (s) | 74.7 / 8.9 | 22.9 | 208.8 | 0.1 |
| max\|Δ prob\| vs padre v1 | 0.0 | **0.0** | **0.0** | 1.34e-3 |

Sin Traceback ni FALLBACK. gold58 del bench 0.9240 (= padre).

**Qué aprendimos:** con la LUT del padre las probabilidades son idénticas (Δ 0.0). `nice 10` recorta el retraso del
arranque (59 → 40 s hasta `main()`) pero no lo elimina (padre 20–24 s): la contención con la prep temprana no es solo de
CPU (probablemente E/S al importar torch/timm mientras 6 procesos leen DICOM). v1 bajó 16 s; v2 cayó en una máquina con
E/S lenta (prep 1.50 s/estudio de pared, GPU esperando 209 s) y subió 72 s: el ruido entre máquinas de Kaggle (±40 s) es
mayor que las ganancias de ingeniería que quedan (~15 s ≈ 0.0002 de AUC en la métrica de eficiencia).

**Siguiente paso sugerido:** no seguir con micro-optimizaciones del runtime de exp080-ctrl224-fast; si acaso, importar
torch/timm antes de crear el pool temprano (solo solapar hash + carga), pero la ganancia esperada (< 10 s) no justifica
sesiones GPU.
