# exp084_kernel_rapido

| Campo | Valor |
|---|---|
| Autor | rodrigo (agente eff3a) |
| Padre | exp080_seqhead_ctrl_mil (kernel `exp080-ctrl224-fast`, LB 0.949) |
| Fecha | 2026-10-18 |
| Rama | exp084_kernel_rapido |
| Estado | killed |

Idea: bajar el runtime del kernel exp080-ctrl224-fast sin cambiar sus predicciones (premio de eficiencia).

## 1. SPEC (antes de tocar el notebook)

**Tipo: PRODUCTO / EFICIENCIA (excepción declarada a "una variable", como exp057).** La variable es "ingeniería de
runtime con predicciones iguales": se agrupan cambios que no tocan el modelo ni los datos; la restricción es que las
probabilidades no cambien (≤ 1e-3). No se entrena nada; no hay pseudo/gold que decidir (gold58 del bench solo se reporta).

**Diagnóstico (bench exp080-bench-fastng6, eff2):** bucle main 251.7 s, prep terminó a 250.8 s (cuello = preparación
DICOM en CPU, 6 procesos en 4 vCPU), GPUs esperando ~37 s cada una; arranque del notebook hasta `main()` ≈ 22 s
(import torch/timm, hash del checkpoint, carga de modelos) durante los cuales no se prepara nada.

**Cambios (fastx.py, sobre el cuerpo del kernel exp080-ctrl224-fast):**

| Id | Padre | Este experimento |
|---|---|---|
| X1 | el pool de 6 procesos de prep se crea dentro de `main()`, tras import torch/timm, hash y carga de modelos | primera celda nueva: lee test.csv/test_series.csv, hace fork del pool de 6 procesos (antes de importar torch) y empieza a preparar; `main()` consume ese iterador |
| X2 | cada estudio viaja del proceso de prep al principal como volumen uint8 66×384×384 | solo el recorte central 320×320 que usa el modelo (mismos valores) |
| X3 | cv2 con sus hilos por defecto en cada proceso | `cv2.setNumThreads(1)` en los procesos de prep (misma salida) |
| X4 | ventanas en GPU: LUT de 256 floats + índice int64 sobre el volumen 384, luego recorte | ventanas sobre el volumen ya recortado; uint8 → float32 por división IEEE (mismos valores que la LUT) |
| X5 | `cudnn.benchmark = False` | `True` (forma fija 64×3×224×224, autotune una vez por GPU; puede cambiar el redondeo fp16) |

Test local (CPU) de X2+X4 contra el `gpu_windows` del padre: max|Δ| = 0.0 en 6 volúmenes sintéticos (máscaras normales,
parciales, vacías y en el borde).

**Medición:** bench Kaggle privado (generador `gen_bench3.py` = `eff2/gen_bench.py` con la celda temprana), T4×2, los
mismos 1300 estudios de train (58 gold + 871 fold 0 + 371 al azar, seed 0), misma imagen docker fijada.
Métrica de tiempo = `nb_elapsed_to_end` (desde la primera celda del bench hasta el fin de `main()`; incluye imports, hash,
carga de modelos y el bucle; con X1 el bucle empieza antes de `main()`, así que `main_total_s` deja de ser comparable).
Padre = `exp080-bench-fastng6` (271.97 s ya medido) + 1 corrida nueva del mismo notebook hoy (ruido entre corridas ±15 s).

**Variante que decide:** `xb` (X1–X5), 2 corridas. Informativa: `x` (X1–X4, sin X5), 1–2 corridas (atribución de X5).

**Criterios (fijos, no cambian después de correr):**
- `worked` si (a) max|Δ prob| ≤ 1e-3 en los 1300 estudios contra `raw_probs.csv` de exp080-bench-fastng6 en las dos
  corridas de `xb`, (b) ninguna corrida con Traceback ni FALLBACK, y (c) media de `nb_elapsed_to_end` de `xb` ≤ media del
  padre (sus corridas) − **30 s**.
- `killed` si (a) o (b) fallan.
- Si no, `didnt` (se informa el ahorro medido).
- Si `xb` no cumple (a) pero `x` sí cumple (a) y (c), se reporta y se usa `x` para el kernel (decisión declarada aquí).

**Efecto esperado:** −20 a −35 s (X1 ≈ los ~22 s de arranque solapados; X2–X4 algo de CPU del proceso principal; X5 algo
de GPU, que pasa a ser cuello cuando la prep se adelanta).

**Presupuesto:** 0 $ RunPod; 4–5 sesiones GPU Kaggle de ~6 min, de una en una (las sesiones de la cuenta se comparten).

**Riesgos:** fork antes de importar torch con resultados acumulándose en RAM mientras se carga el modelo (~1 GB como
máximo); la cola de envíos de Kaggle mide el runtime real distinto del bench.

## 2. PLAN

- Notebook `exp084_kernel_rapido.ipynb`: celdas que generan los bench (llaman a `gen_bench3.py` / `fastx.py`, guardados en
  esta carpeta), leen los `bench_report.json` + `raw_probs.csv` descargados y escriben `metrics.json`. No hay
  entrenamiento.
- Orden: `xb` corrida 1 → padre corrida nueva → `xb` corrida 2 → `x` (informativo). Una sesión GPU a la vez.

## 3. VERDICT (números de `results/exp084_kernel_rapido/metrics.json`)

**Resultado:** killed · **Fase alcanzada:** bench (1 corrida de `xb`; la 2.ª no se corrió porque el criterio (a) ya falló)

| Métrica (bench Kaggle T4×2, 1300 estudios de train) | Padre exp080-bench-fastng6 (v1, v2) | `xb` X1–X5 (decide) | `x` X1–X4 (info) |
|---|---|---|---|
| `nb_elapsed_to_end` (s) | 272.0, 268.2 (media 270.1) | 297.3 (Δ +27.3) | 298.6 (Δ +28.5) |
| `main_total_s` (s) | 251.7, 244.2 | 235.6 | 240.0 |
| inicio de `main()` desde la 1.ª celda (s) | ~20–24 | ~62 | ~59 |
| espera GPU total (s) | 74.7, 8.9 | 0.1 | 0.1 |
| max\|Δ prob\| vs padre v1 | 0.0 (v2) | **1.37e-3** (> 1e-3) | 1.34e-3 |
| gold58 AUC del bench | 0.9240 | 0.9240 | — |

Sin Traceback ni FALLBACK en ninguna corrida.

**Qué aprendimos:** (1) X1 sin prioridad es contraproducente: los 6 procesos de prep saturan los 4 vCPU mientras el
proceso principal importa torch/timm, hace el hash y carga el modelo (main empieza ~40 s más tarde); con la prep
adelantada la GPU pasa a ser el cuello (cola llena, espera ≈ 0), así que ese retraso se paga entero. (2) La deriva de
probabilidades no viene de cudnn.benchmark (`x` sin X5 da lo mismo) sino de X4: en CUDA `x / 255` es multiplicación por el
recíproco, no la división IEEE del CPU que generó la LUT; el fp16 la amplifica a ~1e-3.

**Siguiente paso sugerido:** exp087 = `x` con la LUT del padre y procesos de prep con `os.nice(10)`.
