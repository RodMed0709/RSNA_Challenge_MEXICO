# exp049_raptor_clahe

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | exp049_raptor_clahe (desde exp031_raptor_etiquetas_consenso) |
| Estado | spec |

Idea: track T3 (preprocesado) del paso 7. El corpus Raptor es uint8 normalizado por percentiles 2–98 de cada serie; el contraste local queda plano en tejidos blandos (meniscos, ligamentos, edema óseo) y varía mucho entre sitios. CLAHE ecualiza el contraste local y es barato en inferencia (cv2 en Kaggle).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** aplicar CLAHE por corte (OpenCV, `clipLimit=2.0`, `tileGridSize=(8, 8)`) al uint8 384 del corpus sube la pseudo AUC L2 de fold 0 en ≥ +0.003 contra exp031 sin que gold caiga > 0.02.

**Única variable:** píxeles de entrada. Padre: `/root/corpus96_384` (uint8, percentiles 2–98, recorte 140 mm, 384) → este: `/root/corpus_clahe` = el mismo corpus con `cv2.createCLAHE(2.0, (8, 8)).apply` sobre cada corte válido (máscara > 0) del corte completo de 384 (tiles de 48 px), antes del recorte central 320 y de las ventanas; cortes no válidos siguen a cero; máscara e ids idénticos. Mismo CLAHE en train, val, gold e inferencia (es preprocesado, no augmentation). Se aplica offline (corpus nuevo en el disco local del pod, ~59 GB) para no cargar la CPU del dataloader.

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

**Presupuesto:** fold 0 ≈ 2.5 GPU-h (como el padre) en rsna-e2; construcción del corpus ~15–30 min CPU/disco local. Costo de inferencia: CLAHE de 96 cortes 384² por estudio (se mide en el build).

**Riesgos:** CLAHE amplifica ruido en zonas planas (fondo) y cambia la distribución que vio el preentrenamiento ImageNet; la ganancia ×U(0.9, 1.1) del padre se aplica después del CLAHE (igual que antes, sobre el uint8 de entrada).

**Nota:** `decision_auto` del notebook usa los umbrales del repo (±0.005); el verdict usa los criterios de arriba.

## 2. PLAN

- `build_corpus_clahe.py` (script auxiliar en esta carpeta): lee `/root/corpus96_384`, escribe `/root/corpus_clahe` (16 procesos, `nice`), mide ms/estudio de CLAHE (costo de inferencia por estudio).
- Notebook = copia del de exp031; cambio: celda 0 (`EXP`, `PADRE`, `CORPUS`, `NOTES`). Ninguna otra celda cambia.
- Smoke `SMOKE=True` → run real `FOLDS=[0]` en rsna-e2 (tras el all-data seed 7).

## 3. VERDICT (números de `results/<EXP>/metrics.json`)

Pendiente.
