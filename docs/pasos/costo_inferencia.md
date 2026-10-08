# Costo de inferencia de punta a punta: decodificación y preprocesado DICOM (CPU)

Fecha: 2026-10-08. Medición para el track de eficiencia y el paper. Todos los números salen de los CSV crudos en
`/workspace/data/_logs/medicion_costo_20261008/` (scripts `medir.py`, `medir_par.py`, `desglose.py` en la misma carpeta).

## Qué se midió y cómo

- **Muestra:** 60 estudios de train, semilla 42, estratificados proporcionalmente por familia de fabricante × idioma
  (16 estratos: Philips/GE/Siemens/Canon-Toshiba × en, es, el, tr, hr, de, bg_ru, nl, fr). 333 series, 11 001 cortes, 7,20 GB
  (`muestra_estudios.csv`).
- **Disco:** la muestra se copió a `/root/tmp_medicion` (disco local del pod) antes de medir: 7,20 GB en 116,7 s
  (1.ª corrida) y 119,4 s (2.ª) desde el volumen de red (~60 MB/s). Ese tiempo **no** entra en las medidas. Al estar recién
  copiados, los ficheros probablemente estaban en page cache: la medida es de CPU, no de disco frío. La lectura pura de bytes
  fue 0,087 s/estudio (5,2 s para los 7,2 GB), despreciable frente al resto. Carpeta borrada al terminar.
- **CPU:** `taskset -c 0-3`, `OMP_NUM_THREADS=4`, `cv2.setNumThreads(1)` por proceso. Sin GPU. El pod tenía en paralelo dos
  entrenamientos (load average 12–21 en 48 núcleos), así que la medida con 4 procesos es ruidosa (ver abajo).
- **Pipelines por serie:**
  - `decode`: `pydicom.dcmread` + `.pixel_array` de todos los cortes (solo decodificación).
  - `v1`: `src/knee/dicom.read_series` (decodificar, float32, rescale, ordenar, normalizar percentiles 1–99 a uint8) +
    resize 256 `INTER_AREA` (igual que `src/knee/cache.py`).
  - `v2`: `read_series` + `geometry` + `resample` (recorte 0,5 mm, ventana 320 px). Estas dos funciones **no están en
    `src/knee/`**: se ejecutaron tal cual desde la celda del notebook `experiments/exp009_preparacion_v2/02_cache_v2.ipynb`.
    Las 333 series se procesaron sin error.
- **No se midió el modelo** (la GPU estaba ocupada por otro entrenamiento). Este informe cubre solo la parte CPU.

## Hallazgo principal: no hay compresión

Las 333 series de la muestra (11 001 ficheros) son **Explicit VR Little Endian sin comprimir** (`1.2.840.10008.1.2.1`).
Los 557 ficheros de los 3 estudios de test disponibles también (`test_transfer_syntax.json`). No hay JPEG2000 ni JPEG
lossless en lo medido. Consecuencia: "decodificar" es casi gratis (~1 ms/corte); el costo está en **nuestro código de
normalización**, no en los decoders.

Desglose de `read_series` en 40 series al azar (1 hilo, `desglose_read_series_40.csv`; 1 465 cortes, 1,1 GB):

| Paso | Tiempo (s) | % |
|---|---|---|
| `dcmread` | 0,77 | 8,0 |
| `pixel_array` | 0,80 | 8,2 |
| float32 + rescale | 0,97 | 9,9 |
| `np.stack` | 0,31 | 3,1 |
| `np.percentile(1, 99)` | 4,27 | 43,8 |
| clip/escala/round → uint8 | 2,62 | 26,9 |

El **70,7 %** de `read_series` es la normalización por percentiles sobre el volumen float32 completo; la decodificación DICOM
(dcmread + pixel_array) es el 16 %.

## Tabla por transfer syntax / plano / slot (1 hilo, media por serie, `por_serie_1hilo.csv`)

Transfer syntax: una sola clase (raw explicit LE): 333 series, 33,0 cortes, 21,6 MB, decode 0,034 s, `read_series` 0,261 s,
v2 0,286 s por serie.

| Plano | n series | cortes | MB | decode (s) | v1 (s) | v2 (s) |
|---|---|---|---|---|---|---|
| Axial | 84 | 40,2 | 30,9 | 0,041 | 0,388 | 0,384 |
| Coronal | 112 | 27,9 | 15,8 | 0,029 | 0,206 | 0,214 |
| Sagital | 135 | 33,0 | 20,8 | 0,033 | 0,272 | 0,287 |
| Oblicuo | 2 | 25,0 | 10,1 | 0,032 | 0,170 | 0,175 |

| Slot | n series | cortes | MB | decode (s) | v1 (s) | v2 (s) |
|---|---|---|---|---|---|---|
| 1 sag fluid FS | 56 | 40,7 | 24,1 | 0,040 | 0,308 | 0,325 |
| 2 cor fluid FS | 56 | 28,2 | 13,4 | 0,028 | 0,183 | 0,191 |
| 3 ax fluid FS | 60 | 33,6 | 19,4 | 0,032 | 0,246 | 0,254 |
| 4 sag PD/T2 | 35 | 29,0 | 23,1 | 0,032 | 0,300 | 0,314 |
| 5 cor T1 | 45 | 27,8 | 18,8 | 0,030 | 0,235 | 0,245 |
| 6 sag T1 | 27 | 27,0 | 12,4 | 0,025 | 0,170 | 0,181 |
| sin slot | 54 | 39,4 | 36,0 | 0,043 | 0,460 | 0,450 |

Las series fuera de slot son las más caras (más MB y más cortes por serie). El costo escala con los MB (matrices de 192 a
1024 px), no con el plano en sí.

## Escenarios por estudio

Definición: (a) todas las series; (b) los 6 slots v2; (c) slots 1–3; (d) 1 slot (slot 1; si falta, el slot más bajo presente).
Los 60 estudios tienen al menos una serie en cada escenario.

Volumen medio por estudio (`por_estudio_1hilo.csv`):

| Escenario | series | cortes | MB |
|---|---|---|---|
| a todas | 5,55 | 183,4 | 119,9 |
| b 6 slots | 4,65 | 147,9 | 87,6 |
| c slots 1–3 | 2,87 | 98,0 | 54,4 |
| d 1 slot | 1,00 | 40,1 | 24,3 |

Segundos por estudio, mediana de las repeticiones con `Pool(1)` (2 reps) y `Pool(4)` (3 reps, rango entre paréntesis)
(`escenarios_pool1_vs_pool4_rep.csv`, `escenarios_4proc.csv`):

| Escenario | decode 1 proc | decode 4 proc | v1 1 proc | v1 4 proc | v2 1 proc | v2 4 proc |
|---|---|---|---|---|---|---|
| a todas | 0,185 | 0,047 (0,038–0,077) | 1,152 | 2,051 (1,812–3,035) | 1,185 | 2,050 (1,873–2,677) |
| b 6 slots | 0,120 | 0,039 (0,038–0,041) | 0,852 | 0,718 (0,448–0,880) | 0,865 | 0,767 (0,309–0,946) |
| c slots 1–3 | 0,070 | 0,020 (0,019–0,024) | 0,528 | 0,198 (0,164–0,502) | 0,560 | 0,382 (0,221–0,494) |
| d 1 slot | 0,030 | 0,010 (0,008–0,010) | 0,230 | 0,298 (0,079–0,331) | 0,239 | 0,117 (0,098–0,190) |

La decodificación pura escala ~4× con 4 procesos. `v1`/`v2` con 4 procesos **no** escalan de forma estable en este pod: a
veces van más lentos que con 1 proceso (escenario a: 2,05 vs 1,19 s/estudio), con 60–70 % de tiempo de sistema en los
núcleos 0–3 durante la medida. Es compatible con contención de memoria/kernel con los dos entrenamientos que corrían en el
nodo (volúmenes float32 de hasta ~0,5 GB por serie), pero **no lo he aislado**. En una medición en la propia sesión
(1 hilo dentro del mismo proceso, tras el pase de decode) v2 dio 1,588 s/estudio en (a), 1,184 en (b), 0,735 en (c) y 0,332
en (d): la variabilidad entre corridas es del orden de ±30 %.

## Extrapolación a ~1 300 estudios de test con 4 vCPU (pipeline v2)

Penalty = RuntimeSeconds / 32 400. Equivalente AUC = segundos / 697 × 0,01. Supone que los estudios de test se parecen a la
muestra de train en número y tamaño de series (no verificado: solo hay 3 estudios de test locales). Tres columnas: 1 núcleo
(Pool(1) × 1 300), 4 vCPU ideal (1 núcleo / 4) y 4 procesos medidos en el pod (mediana Pool(4) × 1 300, cota pesimista).

| Escenario | 1 núcleo (s) | 4 vCPU ideal (s) | 4 proc pod (s) | Penalty ideal / pod | AUC equiv. ideal / pod |
|---|---|---|---|---|---|
| a todas | 1 540 | 385 | 2 665 | 0,0119 / 0,0823 | 0,0055 / 0,0382 |
| b 6 slots | 1 124 | 281 | 997 | 0,0087 / 0,0308 | 0,0040 / 0,0143 |
| c slots 1–3 | 728 | 182 | 497 | 0,0056 / 0,0153 | 0,0026 / 0,0071 |
| d 1 slot | 311 | 78 | 152 | 0,0024 / 0,0047 | 0,0011 / 0,0022 |

Solo decodificación DICOM (sin normalizar ni remuestrear), 4 vCPU: a 61 s, b 51 s, c 26 s, d 13 s medidos
(≤ 0,0009 AUC equiv. en todos los casos). Con v1 los números son casi iguales a v2 (a ideal 374 s, b 277 s, c 172 s,
d 75 s).

## Conclusiones para el diseño

1. **Podar series ahorra CPU casi en proporción a los MB que se dejan de leer**, pero no por "decodificación": los DICOM
   están sin comprimir y la decodificación es ~16 % del costo. Pasar de todas las series a 6 slots ahorra ~27 % (b vs a,
   1 núcleo), a slots 1–3 ~53 % y a 1 slot ~80 %. En AUC equivalente (4 vCPU ideal), ir de (b) a (c) ahorra ~0,0014 y de
   (b) a (d) ~0,0029: un modelo con menos series compensa solo si pierde menos de eso en AUC. La hipótesis del paper debe
   reformularse: lo que se ahorra es **lectura + normalización del volumen**, no decodificación de JPEG.
2. **Antes de podar series, arreglar `read_series`**: `np.percentile` + escala a uint8 son el 71 % del tiempo. Calcular los
   percentiles sobre una submuestra de vóxeles o en uint16 (histograma), y recortar/redimensionar antes de normalizar, debería
   recortar la mayor parte del costo en todos los escenarios. Es una idea para otro experimento (toca `src/knee/`, PR aparte),
   no se ha medido.
3. **El orden de magnitud es pequeño frente al presupuesto si se paraleliza bien**: incluso (a) son ~385 s con 4 vCPU ideales
   (~0,0055 AUC). El riesgo real es la **escalabilidad con 4 procesos**: en el pod v1/v2 no escalaron (hasta 2 665 s en a).
   Hay que medir en el notebook de Kaggle real (CPU 4 vCPU y T4) antes de citar un número en el paper; aquí quedan como cota
   inferior (ideal) y cota pesimista (pod con contención). Falta además sumar el costo del modelo, que no se midió.
