# exp034_raptor_convnext

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp030_raptor_etiquetas_L2 |
| Fecha | 2026-10-11 |
| Rama | exp034_raptor_convnext |
| Estado | smoke |

Idea: misma receta que exp030 (Raptor d96 + etiquetas L2), pero con un backbone ConvNeXt en lugar de CoAtNet-2, para
tener un segundo modelo que se equivoque distinto y sirva para ensamblar.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** un ConvNeXt (CNN pura, sin atención) entrenado con la receta de exp030 queda cerca en pseudo L2
(≥ exp030 − 0.01) y comete errores lo bastante distintos para aportar al ensamble: correlación de predicciones en val
con exp030 < 0.97, o el promedio por rangos exp030 + exp034 sube la pseudo L2 ≥ +0.003 sobre exp030 solo.

**Única variable cambiada:**

| Parámetro o celda | Padre (exp030) | Este experimento |
|---|---|---|
| `ARCH` (celda 0) | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` | `convnext_base.fb_in22k_ft_in1k_384` |

Ajuste obligado por el backbone (declarado, no es otra variable): `build_backbone` no pasa `img_size` a ConvNeXt
(es totalmente convolucional y timm no acepta ese argumento; la entrada sigue siendo el recorte 320). Lo demás idéntico:
corpus `/root/corpus96_384`, 2.5D, recorte central 320, `RaptorClassifier` (atención MIL por hallazgo; `F_dim` =
`num_features` = 1024 en ambos), etiquetas L2 blandas, 16 épocas, bs 8 × k 12 (k_eval 24), AdamW wd 0.02,
**LR backbone 3e-5 / cabeza 1e-3** (las mismas; son LR típicas de fine-tune de ConvNeXt con AdamW, no hace falta
cambiarlas), OneCycle 0.15, bf16, clip 3.0, grad checkpointing, `drop_path` por defecto (0, igual que el padre), semilla 42,
selección por pseudo L2, SWA top-3.

**Por qué `convnext_base.fb_in22k_ft_in1k_384`** (medido en rsna-d2, 96 imágenes 320², bf16, grad checkpointing,
paso de entrenamiento con AdamW):

| Backbone | Parámetros | Entrenamiento s/paso | Evaluación s/96 img | Pico VRAM |
|---|---|---|---|---|
| coatnet_rmlp_2_rw_384 (padre) | 72.9 M | 1.120 | 0.248 | 14.6 GiB |
| **convnext_base.fb_in22k_ft_in1k_384** | 87.6 M | 0.527 | 0.137 | 6.2 GiB |
| convnext_small.fb_in22k_ft_in1k_384 | 49.5 M | 0.367 | 0.097 | 4.6 GiB |

- Capacidad comparable al padre (88 M vs 73 M) y mismo nivel de preentrenamiento (ImageNet-22k → 1k, ajustado a 384,
  cerca del recorte 320); la variable es la familia (CNN pura vs híbrido conv+atención), no el tamaño.
- Cabe de sobra en tiempo: ~2× más rápido que CoAtNet-2 en entrenamiento e inferencia (bueno también para el premio de
  eficiencia). ConvNeXt-S sería aún más rápido pero mezcla "familia" con "la mitad de capacidad".
- **Kaggle:** pesos `timm/convnext_base.fb_in22k_ft_in1k_384` en Hugging Face, licencia **Apache-2.0**,
  `model.safetensors` 354 MB (fp32). En la inferencia no hace falta el preentrenado: se crea con `pretrained=False` y se
  carga nuestro checkpoint (~350 MB fp32, ~175 MB en fp16) desde un dataset privado de Kaggle; tamaño muy por debajo de los
  límites de datasets. Sin internet no hay problema.

**Criterios (no cambian después de correr)** — pseudo = pseudo AUC macro contra L2 binarizada (> 0.5) en val fold 0,
**mejor época**; exp030: pseudo **0.8795** (ep12), gold 0.9123 (`metrics.json` de exp030).
Correlación = media sobre las 12 etiquetas de la correlación de **Spearman** entre las predicciones de val (mejor época)
de exp034 y de exp030 (`oof_best_f0.csv`, copiadas a `/workspace/users/rodrigo/preds_ref/exp030_raptor_etiquetas_L2/`);
se reporta también Pearson. Ensamble = media de rangos percentiles por etiqueta de ambos modelos. Todo se calcula en la
celda 5 (`DECISION_SPEC`).
- `worked`: pseudo ≥ exp030 − 0.01 (≥ 0.8695) **y** (Spearman medio < 0.97 **o** pseudo del ensamble por rangos ≥
  exp030 + 0.003).
- `killed`: pseudo < exp030 − 0.03 (< 0.8495).
- `didnt`: cualquier otro caso.
- Gold se reporta solo como información y veto (no decide; si cae > 0.02 contra exp030 se anota como riesgo para el envío).

**Presupuesto:** fold 0 ≈ 1.3–1.6 GPU-h (434 pasos × ~0.6 s + evaluación), rsna-d2. Smoke ≤ 10 min.

**Riesgos:** ConvNeXt con LR backbone 3e-5 puede necesitar más épocas o warmup distinto (no se toca: una variable);
una correlación alta (> 0.97) es probable porque ambos aprenden las mismas etiquetas L2 con la misma cabeza; el cuello
de botella puede pasar a ser el dataloader (12 workers) con un backbone 2× más rápido.

## 2. PLAN

- Notebook `exp034_raptor_convnext.ipynb` = copia de exp030; cambios: celda 0 (`EXP`, `PADRE`, `ARCH`, `NOTES`,
  `PARENT_OOF`, `CORR_MAX`, `ENS_GAIN`), celda 3 (`build_backbone` sin `img_size` para ConvNeXt), celda 5 (correlación,
  ensamble por rangos y `DECISION_SPEC`), celda 6 (`decision_spec` y `ensemble_vs_parent` en `metrics.json`).
- Smoke: `SMOKE=True` (64 train / 48 val, 1 época) → `results/.../smoke/`.
- Run real: `FOLDS=[0]`, `SMOKE=False`, rsna-d2, symlink `results/exp030_raptor_etiquetas_L2` en el clon del pod.

## 3. VERDICT (números de `results/exp034_raptor_convnext/metrics.json`)

**Resultado:** pendiente · **Fase alcanzada:**

| Métrica | exp030 (padre) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8795 | | |
| pseudo L2 fold 0 (SWA top-3) | | | |
| Spearman medio val vs exp030 | — | | |
| pseudo ensamble por rangos exp030+exp034 | — | | |
| v1 fold 0 | | | |
| gold (veto) | 0.9123 | | |
| GPU-h | | | |

**Qué aprendimos:**

**Siguiente paso sugerido:**
