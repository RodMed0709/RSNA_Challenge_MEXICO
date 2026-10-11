# exp029_raptor_entreno

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | — (reproducción de receta pública; referencia interna: exp021, mejor LB propio 0.836) |
| Fecha | 2026-10-11 |
| Rama | exp029_raptor_entreno |
| Estado | run (fold 0) |

Idea: reproducir el entrenamiento público Raptor en su variante nartaa d96/384 sobre nuestro split, con nuestra
regla de selección (pseudo, nunca gold). Base: `docs/pasos/raptor_receta.md` (rama exp010_dev_informes).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** con la receta pública Raptor/nartaa (corpus denso 96×384, CoAtNet-2 384, atención MIL por hallazgo,
etiquetas blandas públicas de Raptor) entrenada en `split_v1_seed42` fold 0, el modelo llega a
**pseudo AUC ≥ 0.90** (L2 binarizada > 0.5, val fold 0) y el modelo fold 0 enviado a Kaggle a **LB ≥ 0.91**.

**No es un experimento de una variable**: es una reproducción de un pipeline externo completo, sin padre del mismo
pipeline. Cambia de golpe datos (corpus), modelo, etiquetas y bucle de entrenamiento contra exp021. Se juzga con
criterios absolutos, no con Δ contra un padre. Las variantes posteriores (JEV HYBRID, TTA espejo, 224, …) sí serán de
una variable con este experimento como padre.

**Receta (fija):**

| Pieza | Valor |
|---|---|
| Corpus | 5 slots Sag fluid 26 / Sag no-fluid 22 / Cor fluid 18 / Cor 12 / Ax 18 = 96 cortes; `linspace` 2–98 % de la serie; percentiles 2–98 por serie; recorte físico 140 mm; `INTER_AREA` a 384; uint8. Construido con `build_corpus96.py` (= `build_study` verbatim de `nartaa/rsna-knee-0949-anatomical-mirror`) en el disco local del pod |
| Entrada | ventana 2.5D (c−1, c, c+1) → RGB; **recorte central 320** del corte de 384 (igual que la inferencia de nartaa: `INPUT_RES=320`, `INPUT_MODE="crop"`), `img_size=320` en timm; `--norm imagenet` |
| Modelo | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (ImageNet, timm) + `RaptorClassifier` (atención MIL por hallazgo), grad checkpointing |
| Entrenamiento | 16 épocas, bs 8 estudios × k=12 ventanas (k_eval 24), AdamW wd 0.02, LR backbone 3e-5 / cabeza 1e-3, OneCycle pct 0.15, BCE con targets blandos y `pos_weight = clip((1−prev)/prev, 1, 10)`, bf16, clip 3.0, ganancia ×U(0.9,1.1), sin flips, semilla 42 |
| Etiquetas de entrenamiento | **públicas de Raptor**: `dreaddevelopment/rsna-knee-labels/labels_llm_soft.parquet` (4349 no-gold, una sola fuente LLM, CC0) |
| Split | `split_v1_seed42`: train = folds ≠ 0 ∩ etiquetas blandas; val = fold 0 |

**Cambios declarados contra el script público** (`dreaddevelopment/knee-mri-training-the-twelve-finding-model`):
1. **Selección**: el script elige la mejor época y las 3 épocas del SWA por AUC en gold-58. Aquí se elige por
   **pseudo AUC macro contra L2 binarizada (> 0.5) en val fold 0** (L2 = `stevenleehans/llm_labels_v4_blend.csv`).
   SWA = media de pesos de las **3 mejores épocas por pseudo L2**. Gold y v1 (`report_labels_v1.csv`) se reportan
   por época solo como información / veto; **gold nunca elige**.
2. Validación en el fold 0 de nuestro split en lugar de gold.
3. Entrada recortada al centro 320 (el script público redimensiona el corte completo a `res`); se hace así para que
   entrenamiento e inferencia vean el mismo campo que los pesos de nartaa.
4. Port a celdas de notebook; conversión a float, ganancia y normalización hechas en GPU (mismas operaciones).
5. Sin ROI (igual que el valor por defecto del script).

**Criterios (no cambian después de correr):**
- Éxito (`worked`): pseudo L2 fold 0 (mejor época o SWA, se reportan ambas; decide la mejor época) ≥ 0.90 y gold ≥ 0.85.
  Si además LB del modelo fold 0 ≥ 0.91, la receta pasa a ser la base del envío (all-data).
- Kill: gold < 0.80 (claramente por debajo de lo publicado, ~0.915) o GPU-h > 6 en fold 0.
- Intermedio (pseudo < 0.90 sin kill): `didnt`; se revisa el pipeline (corpus/etiquetas) antes de seguir.

**Presupuesto:** fold 0 ≈ 3–3.5 GPU-h (RTX PRO 4500 32 GB, pod rsna-d2) · all-data ≈ 4 GPU-h · corpus ≈ 25 min CPU.

**Riesgos:** memoria a bs 8 (20.6 GiB medido a 384; a 320 menos); dataloader limitado por disco (por eso el corpus
va al disco local de 200 GB, 62 GB); etiquetas de Raptor ≠ L2: la vara interna mide acuerdo con otra fuente LLM;
el LB solo se mide con envío (otro paso).

## 2. PLAN

- `build_corpus96.py` (script auxiliar en esta carpeta, declarado: construye el corpus, no entrena): 4407 estudios →
  `/root/corpus96_384/{vols,masks,ids}.npy` en rsna-d2; copia a `/workspace/data/cache/raptor96_384/` para otros pods.
- Notebook `exp029_raptor_entreno.ipynb`, celdas 0–7 (port del script: `StudyWindows`, `build_backbone`,
  `RaptorClassifier`, bucle, top-K/SWA por pseudo L2).
- Smoke: `SMOKE=True` (64 train / 48 val, 1 época) sobre un corpus pequeño (gold + 150 estudios).
- Run real: `FOLDS=[0]`, `SMOKE=False`.
- Después (otro paso, mismo notebook): `ALL_DATA=True` con `SWA_EPOCHS` fijadas a partir del fold 0 para el envío final.

## 3. VERDICT (números de `results/exp029_raptor_entreno/metrics.json`)

**Resultado:** pendiente · **Fase alcanzada:** screen fold 0 en curso

| Métrica | Referencia | Este |
|---|---|---|
| pseudo L2 fold 0 (mejor época) | ≥ 0.90 (criterio) | |
| pseudo L2 fold 0 (SWA top-3) | | |
| v1 fold 0 | | |
| gold (veto) | publicado ~0.915–0.919 (seleccionado en gold) | |
| GPU-h | ≤ 6 | |

**Qué aprendimos:**

**Siguiente paso sugerido:**
