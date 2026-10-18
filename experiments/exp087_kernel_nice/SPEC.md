# exp087_kernel_nice

| Campo | Valor |
|---|---|
| Autor | rodrigo (agente eff3a) |
| Padre | exp084_kernel_rapido (variante `x`, X1–X4) |
| Fecha | 2026-10-18 |
| Rama | exp087_kernel_nice |
| Estado | spec |

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

**Resultado:** pendiente
