# exp050_raptor_res384

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | exp050_raptor_res384 (desde exp031_raptor_etiquetas_consenso) |
| Estado | spec |

Idea: track T3 del paso 7. El corpus guarda 384 px (140 mm), pero la receta (como la inferencia nartaa) recorta el centro a 320 px: tira el 30 % del área (bordes del campo de 140 mm: inserciones, Baker posterior, bursas) y entrena el CoAtNet `_384` fuera de su resolución nativa.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** entrenar y evaluar con el corte completo de 384 (sin recorte central) sube la pseudo AUC L2 de fold 0 en ≥ +0.003 contra exp031 sin que gold caiga > 0.02.

**Única variable:** resolución/campo de entrada `RES` 320 → 384 (= `IMG`; el recorte central pasa a ser nulo y el backbone se crea con `img_size=384`). Mismo corpus, mismas ventanas (k 12 / k_eval 24), mismo batch 8 (el benchmark sintético de `raptor_receta.md` mide bs 8 × k 12 × 384² con grad-ckpt en 20.6 GiB: cabe en 32 GB sin tocar batch ni ventanas). Si el smoke diera OOM, se baja a bs 4 con acumulación ×2 y se declara.

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

**Presupuesto:** fold 0 ≈ 3.5–4 GPU-h (~1.44× píxeles; el benchmark de 384 dio 1.66 s/iter) en rsna-c2.

**Riesgos:** más lento en inferencia (~1.4–1.6×) → peor en el premio de eficiencia aunque suba AUC; el all-data costaría ~5–6 GPU-h; los modelos de nartaa (para ensamble) siguen a 320.

**Nota:** `decision_auto` del notebook usa los umbrales del repo (±0.005); el verdict usa los criterios de arriba.

## 2. PLAN

- Notebook = copia del de exp031; cambio: celda 0 (`EXP`, `PADRE`, `RES = 384`, `NOTES`). Ninguna otra celda cambia.
- Smoke `SMOKE=True` (mide VRAM y s/paso) → run real `FOLDS=[0]` en rsna-c2 (tras exp044).
- Costo de inferencia: ratio s/paso 384 vs 320 del log (entrenamiento) y del eval (k_eval 24).

## 3. VERDICT (números de `results/<EXP>/metrics.json`)

Pendiente.
