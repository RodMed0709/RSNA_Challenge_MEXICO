# exp051_raptor_norm

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | exp051_raptor_norm (desde exp031_raptor_etiquetas_consenso) |
| Estado | spec |

Idea: track T3 del paso 7. Los percentiles 2–98 por serie saturan el 2 % superior: el líquido/edema brillante (derrame, contusión, Baker) y la grasa quedan clipados a 255 en secuencias fluid-sensitive. Un rango 0.5–99.5 conserva esa cola alta a costa de algo de contraste medio.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** reconstruir el corpus con percentiles 0.5–99.5 (en vez de 2–98) sobre los cortes elegidos de cada serie sube la pseudo AUC L2 de fold 0 en ≥ +0.003 contra exp031 sin que gold caiga > 0.02.

**Única variable:** percentiles de la normalización de intensidad del corpus (2–98 → 0.5–99.5, mismo ámbito: todos los píxeles de los cortes elegidos de la serie, clip a [0, 1]). Selección de series, cortes, recorte 140 mm, 384 px y uint8 idénticos (`build_corpus_norm.py` = `build_corpus96.py` con `PCT` cambiado; se verifica que con `PCT=(2, 98)` reproduce bit a bit filas de `/root/corpus96_384`). Corpus nuevo en `/root/corpus96_384_p05` (disco local del pod).

**Todo lo demás idéntico al padre (exp031):** etiquetas de consenso (Raptor + L2 + L3 + L4, media por celda),
CoAtNet-2 384 + `RaptorClassifier`, 16 épocas, bs 8 × k 12 (k_eval 24), AdamW / OneCycle / LRs, bf16, grad-ckpt,
semilla 42, `split_v1_seed42` fold 0, selección por pseudo L2 (L2 binarizada > 0.5) en val fold 0, SWA = media de las
3 mejores épocas por pseudo L2. Gold y v1 solo se reportan.

**Métrica principal:** pseudo L2 fold 0 de la mejor época (la del padre: **0.8836**, gold 0.9144). SWA top-3 se reporta
(padre 0.8835).

**Criterios (fijos, paso 7; no cambian después de correr):**
- `worked` (promete): pseudo L2 fold 0 ≥ 0.8836 + 0.003 = **0.8866** **y** gold no cae > 0.02 contra el padre (≥ 0.8944).
- `killed`: pseudo L2 fold 0 ≤ 0.8836 − 0.003 = 0.8806 **o** gold cae > 0.02.
- `didnt`: cualquier otro caso. Ruido de semilla medido (sub031_2): 0.0013.
- Si `worked`: run all-data (`ALL_DATA=True`, `SWA_EPOCHS` = las 3 mejores épocas por pseudo L2 del fold 0) →
  `swa_alldata.pt` para el ensamble.

**Presupuesto:** fold 0 ≈ 2.5 GPU-h; build ≈ 20–40 min de pared (32 procesos, lee DICOM del volumen).

**Riesgos:** un rango más ancho comprime el contraste medio (cartílago, menisco) en menos niveles de gris uint8; z-score por volumen (otra opción del plan) sería otro experimento.

**Nota:** `decision_auto` del notebook usa los umbrales del repo (±0.005); el verdict usa los criterios de arriba.

## 2. PLAN

- `build_corpus_norm.py` (auxiliar en esta carpeta): costo medido antes (orden: 1317 s de pared con 32 procesos para el corpus original, ≈ 11.7 CPU-h de proceso). Regla del orquestador: si el costo > 3 h, se salta y se reporta.
- Notebook = copia del de exp031; cambio: celda 0 (`EXP`, `PADRE`, `CORPUS`, `NOTES`).
- Smoke `SMOKE=True` → run real `FOLDS=[0]` en el primer pod libre de T3 (rsna-e2 tras exp049).

## 3. VERDICT (números de `results/<EXP>/metrics.json`)

Pendiente.
