# exp009_preparacion_v2

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp001_preparacion |
| Fecha | 2026-09-26 |
| Rama | paso1-backbones (se separa al abrir el PR) |
| Estado | worked |

Idea: paso 3: metadatos DICOM v2 (plano, secuencia, lateralidad, slots) y cache v2

Experimento de **preparación** (como exp001): no entrena ni tiene AUC. Diseño en `docs/pasos/paso3_entrada_v2.md`.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** las cabeceras DICOM (sin píxeles) bastan para (a) asignar a cada estudio sus series por *slots*
de contenido en vez de la prioridad v1 por plano, (b) saber la lateralidad para espejar a rodilla derecha y
(c) calcular el centro físico y el espaciado que necesita el recorte en mm del cache v2.

**Única variable cambiada:**

| Parámetro o celda | Padre (exp001) | Este experimento |
|---|---|---|
| Metadatos por serie | columnas de `train_series.csv` (plano, líquido, FS) | `meta_v2`: plano por IOP, ponderación por TR/TE/TI, FS en cascada, espaciado, centro LPS, lateralidad, slots |

El cache `v1_256`, `raw/` y el split no se tocan. El cache v2 se construye después, en otro notebook de este
mismo experimento, solo si esta pasada cumple los criterios.

**Salidas:** `/workspace/data/cache/meta_v2/series_meta_v2.csv` (una fila por serie, train + test) y
`study_meta_v2.csv` (una fila por estudio: lado, centro, slots). Resumen en `results/exp009_preparacion_v2/metrics.json`.

**Criterios (no cambian después de correr):**
- Plano por IOP determinado (no oblicuo, IOP legible) en ≥ 98 % de las series.
- Lateralidad determinada en ≥ 90 % de los estudios, con precisión ≥ 95 % contra el lado que dice el informe
  (train, estudios con lado inequívoco en el informe). Precisión por método (tag, texto, geometría); un método con
  < 95 % solo entra como voto débil (cuando el estudio no tiene nada mejor).
- Slot 1 (SAG líquido FS) presente en ≥ 95 % de los estudios.
- 0 errores de lectura que dejen una serie sin metadatos (los fallos van a una columna, no paran la corrida).
- Si no se cumple algo: `didnt` para esa pieza y se documenta; los experimentos 3a/3b usan solo lo que pasó.

**Presupuesto:** 0 GPU-h. CPU: ≤ 1 h de reloj con 16 hilos (solo cabeceras), sin tocar el entrenamiento que
corre en el pod.

**Riesgos:** cabeceras anonimizadas (tags vacíos); ponderación por TR/TE ambigua en gradiente (zona TR 1000–1500);
texto de series en muchos idiomas; regex del informe con falsos positivos (p. ej. "droit" = recto femoral, "leve");
el centro geométrico depende de dónde puso el técnico la bobina.

## 2. PLAN

Notebook nuevo `experiments/exp009_preparacion_v2/01_meta_v2.ipynb` (no hereda celdas de entrenamiento):

- 0 parámetros (`MAX_STUDIES`, `N_THREADS = 16`, `SIDE_MM = 15`, `OUT_DIR`).
- 1 setup · 2 lista de series (train + test) · 3 lectura de cabeceras de todos los archivos (ThreadPool 16).
- 4 derivaciones: plano, ponderación, FS, líquido, espaciado, centro, orden, localizador.
- 5 lado por serie y estudio (tag → texto → geometría → voto del estudio → `unknown`).
- 6 lado del informe (regex multilingüe) y tabla de acuerdo por método.
- 7 slots (6) con desempates y descartes; cobertura por slot e idioma aproximado.
- 8 guardar CSV y `metrics.json` (phase `prep`).
- Smoke con `MAX_STUDIES = 50`; luego completo. Ejecución con papermill; salida en `results/exp009_preparacion_v2/`.

## 3. VERDICT (números de `results/exp009_preparacion_v2/metrics.json`)

**Resultado:** worked (parcial: slot 1 no llega al 95 %) · **Fase alcanzada:** prep completa (train 4,407 estudios + test 3) y cache v2 completo

Corrida: `results/exp009_preparacion_v2/01_meta_v2.ipynb` (papermill). 24,386 series, 819,635 cabeceras,
0 errores de lectura, 33.9 min de reloj con 16 hilos (dentro del presupuesto de 1 h). 0 GPU-h.

| Criterio | Umbral | Resultado | ¿Pasa? |
|---|---|---|---|
| Plano por IOP determinado | ≥ 98 % | 99.29 % (173 oblicuas, acuerdo con csv 100 %) | sí |
| Estudios con lado | ≥ 90 % | 98.62 % (61 `unknown`) | sí |
| Precisión del lado vs informe | ≥ 95 % | 97.04 % (n = 1,890) | sí |
| Slot 1 SAG líquido FS (train) | ≥ 95 % | 94.26 % | **no** |
| Series sin metadatos por error | 0 | 0 | sí |

| Lado por método (estudio) | Cobertura | Precisión vs informe (n) |
|---|---|---|
| tag `Laterality` | 49.95 % | 96.57 % (1,078) |
| texto de la serie | 20.70 % | 98.61 % (575) |
| geometría x_c ±15 mm | 97.17 % | 97.21 % (1,830) |

Los tres pasan 95 %: la geometría entra en la cascada como método fuerte. Tag y geometría coinciden en 98.69 % de
2,142 estudios con ambos.

| Slot | Train | Test (3 estudios) | Respaldo usado (train) |
|---|---|---|---|
| 1 SAG líquido FS | 94.26 % | 2/3 | — |
| 2 COR líquido FS | 96.44 % | 3/3 | — |
| 3 AX líquido FS (resp. AX PD/T2) | 99.98 % | 3/3 | 5.72 % |
| 4 SAG PD/T2 sin FS | 56.84 % | 2/3 | — |
| 5 COR T1 (resp. COR PD sin FS) | 76.45 % | 1/3 | 11.98 % |
| 6 SAG T1 | 47.45 % | 1/3 | — |

Otros: ponderación por TR/TE/TI en 22,855 series, por texto 1,526 (Toshiba y parte de GE no traen TR/TE), sin
determinar 5. FS: ScanOptions 12,279 · STIR 505 · texto 1,272 · csv 10,330; acuerdo con el csv 99.57 %.
En el csv `Fluid_Sensitive` es idéntico a `Fat_Suppression` (100 %): no aporta información de ponderación.
38.17 % de las series tienen InstanceNumber en orden inverso a la posición. Mediana 0.3125 mm/px y 3.5 mm entre cortes.
No hay localizadores en el dataset.

**Cache v2** (`02_cache_v2.ipynb` con `SMOKE = False`, papermill; números en `metrics.json["cache_v2"]`):
`/workspace/data/cache/v2_05mm`, 20,790 de 20,790 series OK, 0 errores (sin `errors.csv`). Train 20,775 series de
4,407 estudios + test 15 de 3. 663,977 cortes, 68.0 GB (`uint8 [cortes, 320, 320]`), 31.3 min de reloj con 8 procesos.
Por slot: 1 → 4,156 · 2 → 4,253 · 3 → 4,409 · 4 → 2,507 · 5 → 3,370 · 6 → 2,092 (+ 3 series de test sin slot).
Centro del estudio en 20,741 series; 49 caen al centro de la imagen. QA visual por slot en `qa_v2/`.

**Qué aprendimos:** las cabeceras bastan para plano, secuencia, centro y lado (97 % contra el informe). El slot 1
falla por protocolo, no por bug: el hospital alemán (260 estudios; slot 1 en 6.5 %) hace sagitales PD/T2 sin FS (el
csv también dice FS = 0). La regex del informe no reconoce `∆` (U+2206) usado como Δ en informes griegos; recalculado
aparte con ese carácter normalizado, la precisión global sube a 97.46 % (n = 1,932) y la del griego de 88 % a 98 %
(no está en `metrics.json`; la regex del notebook no se cambió después de correr).

**Siguiente paso sugerido:** 3a (slots) sobre `v2_05mm` con respaldo del slot 1 → SAG PD/T2 sin FS; el cache ya
está centrado con `study_cx/cy/cz`; el espejado con `study_side` se hace al cargar (sin espejar los `unknown`).
