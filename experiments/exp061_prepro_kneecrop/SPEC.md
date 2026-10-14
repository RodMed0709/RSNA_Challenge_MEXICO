# exp061_prepro_kneecrop

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial |
| Fecha | 2026-10-13 |
| Rama | exp061_prepro_kneecrop (desde exp045_raptor_aug_espacial 607644c, como los sub045_* y exp058–060) |
| Estado | didnt |

Idea: brazo extra de la Fase 1 del estudio de preprocesado (aprobado por Rodrigo 2026-10-13): exp045 con el recorte de 140 mm
centrado en la rodilla de cada serie en vez de en el centro de la matriz. Ficha P-1 de `docs/conocimiento/09_preprocesado.md`;
diagnóstico D-2 (`orq/p7c_prepro/diag/d_px.py`, 300 estudios): |dx| centroide–centro p50 4.0 mm / p90 11.5 mm (sagital 7–8 mm),
12.9 % de las series pierden > 5 % del tejido en horizontal con el recorte central.
Origen: CoPAS (Qiu 2024, Nat Commun, doi:10.1038/s41467-024-51888-4: recorte alrededor de una ROI gruesa); Xie 2025
(eClinicalMedicine, doi:10.1016/j.eclinm.2025.103534: recorte centrado en la máscara sagital propagado a todos los cortes).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si el centro del recorte de 140 mm pasa del centro de la matriz (padre) al centro del tejido de la rodilla de
cada serie, la pseudo L2 fold 0 sube porque se pierde menos rodilla en los bordes (sobre todo en sagital) y la anatomía queda
en una posición más constante entre estudios.

**Única variable cambiada:**

| Parámetro o celda | Padre | Este experimento |
|---|---|---|
| Centro del recorte 140 mm del corpus (celda 0 `CORPUS`) | centro de la matriz (`/root/corpus96_384`, build_corpus96.py) | centro del tejido por serie (`/root/corpus_kneecrop`, `build_corpus_kneecrop.py`) |

Método (`build_corpus_kneecrop.py` = copia de `experiments/exp029_raptor_entreno/build_corpus96.py`; solo cambia el centro):
por serie, máscara de tejido = proyección máxima de los cortes elegidos ya en ventana [0,1] (p2–98), desenfoque 5×5, Otsu,
cierre 7×7, apertura 5×5, mayor componente conexa (idéntico a `diag/d_px.py`); centro = centroide de la máscara, limitado a
±25 mm por eje; un solo centro para todos los cortes de la serie (las ventanas 2.5D siguen alineadas). La ventana de recorte
se ajusta para no salir de la imagen (sin relleno, como el padre). Fallback al centro de la matriz (resultado idéntico byte a
byte al padre, comprobado) si la máscara falla (área < 2 % de la imagen o error). Se reconstruye desde los DICOM crudos.
Todo lo demás igual: 140 mm, 384 px, slots, cortes, percentiles 2–98, uint8, split `split_v1_seed42`, semilla 42, 16 épocas,
etiquetas, aug espacial, SWA top-3, fold 0. Afecta a train, validación y gold (y habría que aplicarlo en la inferencia).

**Efecto esperado:** Δ pseudo fold 0 ≈ +0.000…+0.004. Coste de entrenamiento × 1. Coste de inferencia = solo el cálculo del
centro: se mide en el build con `cv2.setNumThreads(1)` (`build_meta.prepro.ms_per_study_1thread`; prueba en 12 estudios: ~80 ms/estudio).

**Criterios (fijos antes de correr; los mismos de la Fase 1; no cambian después de ver resultados):**
- **Referencia** = media de la pseudo L2 fold 0 (mejor época) de exp045 semilla 42 (0.8898899) y de `sub045_5_seed7` (cuando esté);
  ref gold = media de los gold de esas dos corridas (exp045 s42 = 0.9201799).
- **promising** (candidato a Fase 2): Δ pseudo mejor época ≥ **+0.003** vs referencia **y** gold ≥ ref_gold − 0.02.
- **killed**: gold < ref_gold − 0.02 (diga lo que diga pseudo).
- **didnt**: cualquier otro caso.
- **Ronda 2** (automática): si Δ ≥ **+0.002** con la semilla 42, se corre la semilla 7; la decisión final usa la media de las
  2 semillas vs la referencia con el mismo +0.003 y el mismo veto gold (con la media).
- Para la Fase 2 este brazo solo cuenta si su decisión final está disponible cuando lo estén las de P1 y P2 (regla en ESTADO).
- La decisión automática de la celda 5 (regla SDD estándar +0.005 vs padre) es solo informativa; mandan estos criterios.
- Gold (58) solo veta; el checkpoint lo elige pseudo (`SELECT_ON = "pseudo"`).

**Comprobaciones del corpus antes de entrenar** (`check_corpus_kneecrop.py`, log en `orq/p7c_prepro/kneecrop/`; si falla, no se entrena):
mismo N y orden que `ids.npy` del padre; `masks.npy` idénticas (tolerancia ≤ 5 cortes en todo el corpus); cortes de relleno
a cero; uint8 sin NaN; máscaras recalculadas desde el volumen en una muestra; series en fallback idénticas al padre; series
desplazadas distintas del padre; fallback ≤ 5 %; |desplazamiento| mm por slot (crudo y efectivo) comparado con el
diagnóstico (|dx| crudo p50 de todas las series en [2, 7] mm; diag 4.0); PNG de 6 estudios al azar padre vs nuevo.

**Presupuesto:** build ~25–30 min CPU + smoke + fold 0 ≈ 2.7 GPU-h (+ 2.6 si hay ronda 2) en una RTX PRO 4500 ($0.72/h) con
≥ 200 GB de disco local: ~$2.5–4.5.

**Riesgos:** centroide sesgado por muslo/pantorrilla en sagital/coronal (limitado a ±25 mm y por el borde de la imagen);
FOV ≈ 140 mm deja poco margen (el desplazamiento efectivo es menor que el del centroide); coste de inferencia (~80 ms/estudio).

## 2. PLAN

- `build_corpus_kneecrop.py` (esta carpeta) → `/root/corpus_kneecrop` en el disco local del pod, desde `/workspace/data/raw`;
  escribe `kneecrop_centers.csv` (por serie: slot, fallback, desplazamiento crudo/efectivo en mm, ms) y `build_meta.prepro`.
- `check_corpus_kneecrop.py` (esta carpeta) contra `/root/corpus96_384` (copia de `/workspace/data/cache/raptor96_384`).
- Celdas que cambian: **0** (`CORPUS`, `NOTES`) y **6** (`metrics.json`: añade `corpus.prepro` = `build_meta["prepro"]`, solo
  registro). Ninguna otra (mismas dos celdas que exp058/059/060).
- Smoke (`SMOKE = True`) y fold 0 (`FOLDS = [0]`) con papermill desde `orq/p7c_prepro/worker_kc.sh` (pod propio `rsna-pre-c`).
- Ronda 2 (si Δ ≥ +0.002): mismo notebook con `-p SEED 7` en el worktree `wt_exp061_prepro_kneecrop_seed7`; su metrics.json se
  copia como `results/exp061_prepro_kneecrop/metrics_seed7.json`.

## Etapa 3 — confirmación 5 folds (criterio fijado el 2026-10-13 ~04:00 UTC, antes de ver resultados; aprobado por Rodrigo)

Solo aplica si este experimento resulta el **ganador final** de la cadena p7c (regla en `orq/ESTADO.md`, sección "Fase 2 + Etapa 3"):
mejor brazo `promising` de la Fase 2 (media de 2 semillas); si ninguno, el brazo `promising` de la Fase 1; si nada es
`promising`, la Etapa 3 no corre.
- Se entrenan los folds 1–4 con semilla 42 (el fold 0 s42 ya existe) → OOF pseudo de 5 folds concatenados y gold con el
  ensamble (media de probabilidades) de los 5 modelos `best_f{k}.pt`.
- Preprocesado reajustado por fold donde aprende algo: Nyúl → landmarks ajustados solo con el train de cada fold
  (`nyul_landmarks_f{k}.json`, un corpus por fold); recorte de rodilla y CLAHE no ajustan nada (un solo corpus).
- Referencia: exp045 5 folds — OOF pseudo **0.8844**, gold del ensamble de 5 **0.9244**.
- **worked (confirmado)** si Δ OOF pseudo ≥ **+0.003** vs 0.8844 **y** gold del ensamble ≥ 0.9244 − 0.02; si gold cae más
  de 0.02 → **killed**; si no → **didnt** (se queda en fold 0).
- Si se confirma: dataset privado con los 5 pesos + kernel solo privado + kernel ensamble blend-mine w0.15, con el
  preprocesado inyectado igual que en entrenamiento; se corren hasta COMPLETE y se valida `submission.csv`. No se envía a
  Kaggle desde la cadena: los slugs quedan en ESTADO para las colas del 15/16-oct.

## 3. VERDICT (números de `results/exp061_prepro_kneecrop/metrics.json`)

**didnt** (orquestador, 13-oct 09:00 UTC; números de `metrics.json`; decisión de la cadena p7c `state/phase1.json` coincide).

| | referencia (media exp045 s42 + sub045_5 s7) | exp061_prepro_kneecrop s42 | Δ |
|---|---|---|---|
| pseudo L2 f0 (mejor época, ep 15) | 0.8880 | 0.8840 | -0.0041 |
| gold | 0.9223 | 0.9110 | -0.0113 |
| SWA f0 pseudo / gold | — | 0.8836 / 0.9107 | |

- Δ pseudo -0.0041 < +0.003 (y < +0.002 → sin ronda 2, n = 1 semilla); gold 0.9110 ≥ veto 0.9023 → `didnt`.
- Vs exp045 s42 solo (campo `delta` de metrics.json): pseudo -0.0059, gold -0.0092.
- Coste de preprocesado: 78 ms/estudio (1 hilo). GPU-h 2.51.
- Aprendizaje: recorte centrado en la rodilla no mejora a exp045 en fold 0; no entra como padre de la Fase 2 (padre = exp045).

