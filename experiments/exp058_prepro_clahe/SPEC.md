# exp058_prepro_clahe

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial |
| Fecha | 2026-10-13 |
| Rama | exp058_prepro_clahe (desde exp045_raptor_aug_espacial 607644c, como los sub045_*) |
| Estado | spec |

Idea: Fase 1 del estudio de preprocesado (aprobado por Rodrigo 2026-10-13): exp045 + CLAHE por corte (clip 2.0, tiles 8×8) sobre el corpus uint8 384. Origen: exp049 / sub056_1 (CLAHE sobre exp031 y sobre el combo); Zuiderveld 1994 (Graphics Gems IV).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si el corpus de entrada pasa de uint8 percentil 2–98 por serie (padre) a uint8 con CLAHE por corte (clip 2.0, tiles 8×8) sobre el corpus uint8 384, la pseudo L2
fold 0 sube porque realce de contraste local homogeneiza el contraste entre equipos/protocolos (ojo: exp049 y sub056_1 ya salieron didnt; este brazo lo mide contra la receta actual exp045 con la misma vara que el resto del estudio).

**Única variable cambiada:**

| Parámetro o celda | Padre | Este experimento |
|---|---|---|
| Preprocesado del corpus (celda 0 `CORPUS`) | `/root/corpus96_384` (build_corpus96.py) | `/root/corpus_clahe` = corpus96_384 + CLAHE por corte (clip 2.0, tiles 8×8) sobre el corpus uint8 384 (`build_corpus_clahe.py`) |

Método: CLAHE de OpenCV (`cv2.createCLAHE(2.0, (8, 8))`) a cada corte válido 384 del corpus uint8; cortes no válidos a cero. Mismo constructor que exp049 (`build_corpus_clahe.py`, copiado a esta carpeta, solo añade `prepro` a build_meta.json).

Afecta a train, validación y gold (y obligaría a aplicarlo en la inferencia de Kaggle). Todo lo demás igual a exp045: split
`split_v1_seed42`, semilla 42, 16 épocas, etiquetas, aug espacial, SWA top-3, fold 0.

**Efecto esperado:** Δ pseudo fold 0 ≈ +0.003…+0.005; coste de entrenamiento × 1. Coste de preprocesado en inferencia: se mide al construir (`clahe_ms_per_study`; exp049 no dejó el número registrado).

**Criterios (fijos antes de correr; no cambian después de ver resultados):**
- **Referencia** = media de la pseudo L2 fold 0 (mejor época) de `sub045_5_seed7` (exp045 con semilla 7) y de exp045 semilla 42
  (0.8898899); ídem para gold (exp045 s42 = 0.9201799). Se calcula cuando termine sub045_5_seed7.
- **promising** (candidato a Fase 2 y a segunda semilla): Δ pseudo mejor época ≥ **+0.003** vs referencia **y** gold ≥ ref_gold − 0.02.
- **killed**: gold < ref_gold − 0.02 (diga lo que diga pseudo).
- **didnt**: cualquier otro caso.
- **Ronda 2** (automática si hay presupuesto): si Δ ≥ **+0.002** en ronda 1, se corre la semilla 7 de este brazo; la decisión final usa
  la media de las 2 semillas del brazo vs la referencia con el mismo +0.003 (y el mismo veto gold con la media).
- La decisión automática de la celda 5 (regla SDD estándar vs padre +0.005) es solo informativa; mandan estos criterios.
- Gold (58) solo veta; el checkpoint lo elige pseudo (`SELECT_ON = "pseudo"`). Coste de preprocesado en inferencia (ms/estudio,
  1 hilo) se registra en `metrics.json` (`corpus.prepro`) y aquí.

**Presupuesto:** smoke + screen fold 0 ≈ 2.7 GPU-h (+ 2.6 si hay ronda 2) · GPU barata con ≥ 200 GB de disco local (RTX PRO 4500 / 4090 / 5090).
Construcción del corpus en CPU del pod (disco local, no /workspace).

**Riesgos:** corpus mal construido (se revisa con el smoke y `build_meta.json`); contención de CPU si el constructor corre
a la vez que otro entrenamiento (solo afecta al tiempo); coste de inferencia en Kaggle (límite 9 h).

## 2. PLAN

- Constructor `build_corpus_clahe.py` en esta carpeta → `/root/corpus_clahe` en el disco local del pod, desde `/root/corpus96_384`
  (copia de `/workspace/data/cache/raptor96_384`); escribe `build_meta.json` con `prepro` (método, segundos, ms/estudio 1 hilo).
- Celdas que cambian: **0** (`CORPUS`, `NOTES`) y **6** (`metrics.json`: añade `corpus.prepro` = `build_meta["prepro"]`, solo registro).
  Ninguna otra.
- Smoke (`SMOKE = True`) y después fold 0 (`SMOKE = False`, `FOLDS = [0]`) con papermill, cola `orq/p7c_prepro/worker.sh`.
- Ronda 2 (si Δ ≥ +0.002): mismo notebook con `-p SEED 7` en el worktree `wt_exp058_prepro_clahe_seed7`; su metrics.json se copia como
  `results/exp058_prepro_clahe/metrics_seed7.json`.

- **Nota (2026-10-13):** el fold 0 corre en el mismo pod mientras se construye el corpus N4 (CPU compartida): `gpu_hours` y el
  tiempo por paso de este experimento están inflados y no son comparables con los del padre (no afecta a las AUC).

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

## 3. VERDICT (números de `results/exp058_prepro_clahe/metrics.json`)

Pendiente.
