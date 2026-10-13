# exp059_prepro_nyul

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial |
| Fecha | 2026-10-13 |
| Rama | exp059_prepro_nyul (desde exp045_raptor_aug_espacial 607644c, como los sub045_*) |
| Estado | spec |

Idea: Fase 1 del estudio de preprocesado (aprobado por Rodrigo 2026-10-13): exp045 + estandarización de histograma Nyúl–Udupa por tipo de slot. Origen: Nyúl, Udupa & Zhang 2000, IEEE TMI 19(2) (TorchIO HistogramStandardization).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si el corpus de entrada pasa de uint8 percentil 2–98 por serie (padre) a uint8 con estandarización de histograma Nyúl–Udupa por tipo de slot, la pseudo L2
fold 0 sube porque quita la variabilidad de escala de intensidad entre estudios/equipos que el percentil 2–98 por serie no corrige.

**Única variable cambiada:**

| Parámetro o celda | Padre | Este experimento |
|---|---|---|
| Preprocesado del corpus (celda 0 `CORPUS`) | `/root/corpus96_384` (build_corpus96.py) | `/root/corpus_nyul` = corpus96_384 + estandarización de histograma Nyúl–Udupa por tipo de slot (`build_corpus_nyul.py`) |

Método: Landmarks = percentiles [1, 10, 20, …, 90, 99] del primer plano (vóxel > media) de la pila de cortes válidos de cada slot; escala estándar 0..255 (p1→0, p99→255) promediada por tipo de slot (5 tipos) **solo con estudios TRAIN del fold 0** (fold ∉ {0, −1, −2}: sin validación del fold 0 ni gold). Se aplica a cada estudio y slot como mapa lineal a trozos (segmentos extremos extrapolados, igual que TorchIO) → LUT de 256 sobre el uint8 → recorte [0, 255] → uint8. Landmarks guardados en `experiments/<este>/nyul_landmarks.json` (los necesita la inferencia de Kaggle).

Afecta a train, validación y gold (y obligaría a aplicarlo en la inferencia de Kaggle). Todo lo demás igual a exp045: split
`split_v1_seed42`, semilla 42, 16 épocas, etiquetas, aug espacial, SWA top-3, fold 0.

**Efecto esperado:** Δ pseudo fold 0 ≈ +0.003…+0.005; coste de entrenamiento × 1. Coste de preprocesado en inferencia: ≈ 170 ms/estudio (medido en rsna-cron, 30 estudios, 1 hilo).

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

- Constructor `build_corpus_nyul.py` en esta carpeta → `/root/corpus_nyul` en el disco local del pod, desde `/root/corpus96_384`
  (copia de `/workspace/data/cache/raptor96_384`); escribe `build_meta.json` con `prepro` (método, segundos, ms/estudio 1 hilo).
- Celdas que cambian: **0** (`CORPUS`, `NOTES`) y **6** (`metrics.json`: añade `corpus.prepro` = `build_meta["prepro"]`, solo registro).
  Ninguna otra.
- Smoke (`SMOKE = True`) y después fold 0 (`SMOKE = False`, `FOLDS = [0]`) con papermill, cola `orq/p7c_prepro/worker.sh`.
- Ronda 2 (si Δ ≥ +0.002): mismo notebook con `-p SEED 7` en el worktree `wt_exp059_prepro_nyul_seed7`; su metrics.json se copia como
  `results/exp059_prepro_nyul/metrics_seed7.json`.

- **Corrección de implementación (2026-10-13 ~03:05 UTC, antes de ver ningún resultado; criterios sin cambios):** la v1 del
  constructor usaba la máscara por defecto de TorchIO (vóxel > media de la pila) y mapeaba p1 → 0; todo lo que estaba por debajo
  de la media se extrapolaba a negativo y se recortaba a 0 (auditoría: 42 % de los vóxeles > 0 a cero, rango 29–61 %; borra
  meniscos/ligamentos). v2: landmarks sobre los vóxeles > 0 de los cortes válidos y ancla 0 → 0 (escala estándar = p/p99·255,
  LUT por tramos [0, landmarks propios] → [0, landmarks estándar]); el 0 (aire/relleno) sigue en 0. Aceptación: < 2 % de vóxeles
  > 0 mandados a 0 (prueba en 25 estudios: 0.006 %); el build completo lo registra en `build_meta.prepro.pos_to_zero_frac`.
  Se reconstruye `/root/corpus_nyul`, se re-commitea `nyul_landmarks.json` (version 2), smoke en el corpus nuevo antes del fold 0;
  la ronda 2 (semilla 7) usa el mismo corpus v2.

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

## 3. VERDICT (números de `results/exp059_prepro_nyul/metrics.json`)

Pendiente.
