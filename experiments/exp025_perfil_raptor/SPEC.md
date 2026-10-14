# exp025_perfil_raptor

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | — (perfilado de código de terceros: `nartaa/rsna-knee-0949-anatomical-mirror`) |
| Fecha | 2026-10-11 |
| Rama | exp025_perfil_raptor |
| Estado | worked |

Idea: medir por etapa dónde se va el tiempo de inferencia del notebook público Raptor/0.949 (un brazo
CoAtNet-2 + TTA espejo anatómico) para preparar una versión rápida (premio de eficiencia: dentro del
top 10 el orden lo decide el runtime). Alimenta exp026–exp028.

## 1. SPEC

**Hipótesis:** en la inferencia de Raptor/0.949, la lectura+decode DICOM más la normalización por
percentiles dominan más del 60 % del runtime por estudio.

**Qué se mide (no hay variable de entrenamiento; es un experimento de medida):**

| Etapa | Qué incluye |
|---|---|
| `list` | elegir series de los 5 slots + `glob` de los `.dcm` |
| `hdr_read` | lectura de cabeceras de **todos** los archivos de las 5 series (orden por IPP·normal, PixelSpacing) |
| `select` | orden + `linspace` 2–98 % de los cortes por slot |
| `px_read` | I/O de los bytes de pixel de los cortes elegidos |
| `decode` | decode DICOM → array (walker propio en `study_lut`; pydicom en `v2`) |
| `pct_norm` | percentil 2/98 por serie + clip/normalización (en `study_lut`: tabla + gather sobre el recorte) |
| `crop_resize` | recorte 140 mm + `cv2.resize` INTER_AREA a 384 + uint8 |
| `windows` | armado en GPU de las 94 ventanas 2.5D (normal + espejo), recorte central 320 |
| `fwd_plain` / `fwd_mirror` | forward CoAtNet-2 fp16 (autocast), 94 ventanas en trozos de 16, cada vista |

Se perfilan dos lectores, ambos del propio notebook: `study_lut` (el que usa el notebook, módulo
`_fastread` embebido en base64) y `v2` (su lector pydicom literal, `_make_reader` + `build_study`)
como referencia ingenua. Cada lector instrumentado se verifica **byte a byte** contra el builder del
notebook en los 3 primeros estudios.

**Condiciones:** 50 estudios de nuestro train raw DICOM (`/workspace/data/raw/train_series`, mismo
formato que el test; `train_series.csv` hace de `test_series.csv`): 25 gold (fold −1) + 25 de fold 0 de
`split_v1_seed42`, semilla 0. Pod rsna-d (RTX PRO 4500, 112 vCPU, volumen MooseFS). Dos escenarios
CPU: 112 hilos y 4 hilos (`taskset -c 0-3`, `torch.set_num_threads(4)`, `cv2.setNumThreads(4)`) para
aproximar Kaggle (4 vCPU). Además: throughput del pool de preparación del notebook (`Pool(nproc)` fork,
96 procesos con 112 CPU; 4 procesos con 4 CPU) y micro-benchmark de variantes del forward.

**Criterios (fijados antes de correr):**
- Hipótesis confirmada si `list+hdr_read+select+px_read+decode+pct_norm` ≥ 60 % del total por estudio
  (lector del notebook, `study_lut`, escenario 4 hilos). Rechazada si < 60 %.
- Sanity: AUC de las predicciones sobre los gold del sample y pseudo AUC vs L2>0.5 deben ser
  claramente altas (> 0.85); si no, el script no reproduce el notebook y no vale el perfilado.
- Identidad: el volumen de cada lector instrumentado == builder del notebook (byte a byte).

**Excepción a la regla 4 (notebook):** aquí el experimento es un `.py`
(`profile_raptor.py`) porque es inferencia sobre código de terceros: el script importa la celda de
código del notebook público y su `_fastread` embebido (no los copia al repo) y re-cronometra las
mismas llamadas. No entrena nada ni toca `src/knee/`.

**Riesgos:** el volumen de RunPod (MooseFS/FUSE) no es el disco de Kaggle: la latencia de I/O no es
la misma. La GPU es una PRO 4500, no una T4: los tiempos de forward se escalan a T4 por FLOPs
(estimación, no medida). Raptor se entrenó con todos los datos (`alldata`), así que el sanity AUC
sobre train está inflado: solo sirve para detectar que el pipeline está roto.

## 2. PLAN

- `profile_raptor.py --mode serial` (112 y 4 hilos, lectores `lut,v2`, page cache desalojado antes de
  cada estudio con `posix_fadvise(DONTNEED)`; en MooseFS no cambia nada: no hay caché de página).
- `--mode pool` (112 y 4 CPU): throughput real del pool de preparación del notebook.
- `--mode gpubench` (5 estudios): trozo 16/32/47/94, espejo en el mismo batch, sin TTA, channels_last.
- Pesos: `nartaa/rsna-knee-publication-swa-weights-20261007` en
  `/workspace/data/public_assets/nartaa__rsna-knee-publication-swa-weights-20261007/`.
- Resultados: `results/exp025_perfil_raptor/metrics.json` (resumen); JSON crudos solo en el pod.

## 3. VERDICT (números de `results/exp025_perfil_raptor/metrics.json`)

**Resultado:** worked (la medida salió) · **Hipótesis:** rechazada para el lector que usa el notebook.

Identidad byte a byte contra los builders del notebook: 3/3 `study_lut` y 3/3 `v2`, en los dos
escenarios. Sanity (mismas predicciones con 112 y 4 hilos): gold AUC macro **0.929** (n = 25, 12
hallazgos), pseudo AUC vs L2>0.5 **0.931** (n = 50). Inflado porque Raptor se entrenó con todos los
datos; solo prueba que el pipeline reproduce el notebook.

Medias por estudio en ms, serie, sin solapar (los forward son siempre de la PRO 4500):

| Etapa | study_lut 112 h | study_lut 4 h | v2 pydicom 112 h | v2 pydicom 4 h |
|---|---|---|---|---|
| list | 8.8 | 7.2 | 12.7 | 11.5 |
| hdr_read (151 archivos) | 56.9 | 54.2 | 267.6 | 169.6 |
| select | 1.0 | 0.5 | 1.2 | 0.7 |
| px_read (89 únicos, 59 MB) | 181.3 | 179.4 | 173.5 | 168.4 |
| decode | 63.2 | 53.9 | 149.3 | 102.4 |
| pct_norm | 40.3 | 31.8 | 332.8 | 328.6 |
| crop_resize | 61.5 | 49.0 | 56.2 | 43.9 |
| windows (GPU) | 20.3 | 19.2 | 20.3 | 19.2 |
| fwd_plain (GPU) | 195.9 | 196.2 | 195.9 | 196.2 |
| fwd_mirror (GPU) | 200.6 | 200.9 | 200.6 | 200.9 |
| **total** | **829.7** | **792.1** | **1410.1** | **1241.3** |
| lectura+decode+norm | 42.4 % | **41.3 %** | 66.5 % | 62.9 % |

Throughput del pool de preparación del notebook (50 estudios, sin GPU): 4 CPU / 4 procesos →
**0.30 s/estudio**; 112 CPU / 96 procesos → 0.63 s/estudio (más procesos saturan el volumen).

Forward (gpubench, 5 estudios, s/estudio en PRO 4500): TTA 2 pasadas trozo 16 = 0.402 (32: 0.417,
47: 0.432, 94: 0.449); espejo en el mismo batch = 0.405; sin TTA = 0.204; **channels_last trozo 16 =
0.354 (−12 %)**. FLOPs: 5 979 GFLOP por pasada de 94 ventanas (≈ 12 TFLOP por estudio con TTA;
la PRO 4500 logra ≈ 30 TFLOPS efectivos).

**Lectura para Kaggle (estimación, no medida):** con una T4 ≈ 4× más lenta que la PRO 4500 en este
modelo (rango 3–5×), el forward con TTA cuesta ≈ 1.6 s/estudio por T4 (1.2–2.0) → ≈ 0.8 s/estudio de
reloj con las 2 T4, frente a ≈ 0.3 s/estudio del pool de preparación con 4 CPU. El notebook corre
prep y GPU solapados, así que **el cuello de botella en Kaggle es el forward de la GPU** (≈ 17 min
para ~1300 estudios, coherente con los 26.1 min submit→score del autor, que incluyen cola). La
lectura solo pesaría si el disco de Kaggle fuera ≥ 2.5× más lento que el volumen de RunPod: hay que
medirlo en Kaggle.

**Qué aprendimos:** el notebook ya trae la lectura optimizada (`study_lut`: cabeceras por pread de
4 KB, decode propio, percentil exacto por histograma de códigos, una tabla por serie); eso baja la
preparación de 1.24 a 0.79 s por estudio en serie y deja la lectura+normalización en 41 %. Con T4 el
runtime lo manda el forward (TTA = la mitad). El número de CPU apenas cambia la preparación por
estudio (está limitada por I/O y por numpy de un hilo).

**Siguiente paso sugerido:** exp026 = correr este mismo perfilador en un kernel de Kaggle (T4 + disco
real, con `train_series` de la competición) para fijar el factor T4 y la velocidad de I/O; luego
atacar el forward (channels_last, TTA parcial, menos ventanas) en exp027–exp028. Ver la lista
priorizada en `BITACORA.md`.
