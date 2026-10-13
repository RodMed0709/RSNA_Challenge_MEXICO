# exp066_compact_f0

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | — (familia nueva). Registro: `exp002` solo por mecánica de `nuevo_experimento`. Referencia del blend: `exp045_raptor_aug_espacial` (SWA fold 0) |
| Fecha | 2026-10-13 |
| Rama | exp066_compact_f0 |
| Estado | spec |

Idea: reproducir la OTRA mitad del blend-mine 0.954 (la compact ConvNeXt-S de heliosli, brazo `E_lrprior`) entrenada
limpia en nuestro split (folds 1–4; fold 0 fuera; gold fuera), para tener un blend interno limpio = Raptor exp045 f0 +
compact f0. Aprobado por Rodrigo (2026-10-13).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** la compact (ConvNeXt-S 2.5D, vistas centradas en la rodilla, atención por clase) es una familia distinta de
la Raptor CoAtNet; su error es poco correlacionado con el de exp045, así que el blend rank-mean 0.5/0.5 en fold 0 sube la
pseudo AUC ≥ +0.005 sobre exp045 solo (como en Kaggle, donde nartaa + compact da 0.954 contra ~0.95 por separado).

**Receta de referencia (pública, `heliosli/rsna-knee-compact-v1`, bundle `209a32c8b50a6b1c` = el `C_SLUG`
`rsna-knee-compact-r6cnxfsss` que usa nuestro blend-mine; mismos sha256):**

| Pieza | Receta pública (manifest) | exp066 |
|---|---|---|
| Código | `compact_train.py` sha256 14767c7a… (kernel `bliverrigpdoge/velci-kcd25-20`, cell 4) | el mismo, port con **un** cambio: folds de la columna `fold` (split_v1) en vez de `sha1 % K` |
| Brazo | `E_lrprior` = pérdida de razón de verosimilitud `-log(lr·σ(z) + σ(−z))` sobre `L|lr_prior`, sesgo inicial `logit(media L|prior)` | igual |
| Backbone | `convnext_small` init `convnext_small_in22k_ft_in1k.safetensors` | `timm/convnext_small.fb_in22k_ft_in1k` (mismo checkpoint público) |
| Entrada | caché `px384_v4` (384 px, vistas `v4` centradas con anclas SKM-TEA `anatomy.pt`, chequeo de lado `side_cls.pt`), ventanas de 3 cortes, `phys_mm 3.5`, recorte 352 | caché reconstruida desde DICOM con el **camino de inferencia público** (`compact_infer.CStudies`: native → geometría → chequeo de lado → anclas → `px_cache.StudySlices.build`), mismos parámetros |
| Entrenamiento | 16 épocas, batch 2, 176 ventanas/estudio, lr 1e-4, head_lr 1e-3, wd 0.05, warmup 5 % + coseno, clip 5, bf16, modelo de la **última época** (sin selección) | igual (+ `--grad-ckpt`, misma matemática) |
| Arquitectura | `Net` con atención espacial por clase (`spatial`), `extent` (área de defecto OA) y cabeza de estados (`state_spec`) | `spatial` sí; **sin cabeza de estados ni extent** (ver etiquetas) |
| Evaluación | eval stride 1 (manifest), recorte central 352 | igual |
| Folds | 4 folds hash, sin gold | train = folds 1–4 de `split_v1_seed42` (3477), val = fold 0 (871), gold 58 solo veto |

**Etiquetas (declaración).** Los targets de la receta (`probe_targets_opusv11b_rr3_plv21_cdr.csv`: estados categóricos de
informe hechos por un LLM, sus `prior`/`lr_prior` y `state_targets`) **no son públicos**. Solo es público `state_spec.json`
(categorías y vector `prior` por estado; su docstring dice "mean report prior of the state's category over the pool, not
fitted on gold"; no podemos verificarlo). Sustitución: las etiquetas de entrenamiento de nuestra Raptor (consenso exp031 =
media por celda de Raptor `labels_llm_soft`, L2 stevenleehans v4_blend, L3 pilkwang v2, L4 lixin73 gpt56sol; ninguna
elegida con gold), como `L|prior = p` recortado a [0.01, 0.99] y `L|lr_prior = odds(p) / odds(base_c)` con `base_c` = media
de p en train (así el óptimo de la pérdida LR con el sesgo inicial de la receta es p). Celda sin fuente → p = base (lr 1).
Sin estados categóricos no hay cabeza de estados ni supervisión `extent`, así que se quitan. Por tanto exp066 es
"compact con nuestras etiquetas", no la compact exacta.

**Variable (familia nueva):** entrenar la receta compact en nuestro split con nuestras etiquetas. No es un experimento de una
variable contra un padre; la comparación decisoria es el blend con exp045.

**Métricas (fold 0, pseudo = L2 binarizada > 0.5, como exp045):**
- (a) compact f0: pseudo, v1 y gold (gold solo veto). Se reportan siempre.
- (b) pregunta principal: blend rank-mean 0.5/0.5 por hallazgo (rangos dentro del conjunto evaluado) de las predicciones SWA
  fold 0 de exp045 (`results/exp045_raptor_aug_espacial/oof_swa_f0.csv`, `gold_swa_f0.csv`) y las de exp066.

**Criterios (fijos, no cambian después de correr).** exp045 solo (metrics.json, SWA f0): pseudo **0.8895**, gold **0.9201**
(mejor época: pseudo 0.8899, gold 0.9202).
- `worked`: blend pseudo ≥ 0.8895 + 0.005 = **≥ 0.8945** y blend gold ≥ 0.9201 − 0.02 = **≥ 0.9001**.
- `killed`: blend gold < 0.9001, o GPU-h > 1.5 × presupuesto (> 21 GPU-h).
- `didnt`: cualquier otro caso.
La compact sola no decide (es familia nueva); su pseudo/gold se reportan para el paper y para decidir si vale reentrenar 5 folds.

**Si worked:** siguiente experimento = folds 1–4 (o all-data) de la compact para el blend propio en Kaggle (SPEC aparte).

**Presupuesto:** caché ≈ 0.6 h de pod (CPU, 24 procesos) + entrenamiento ≈ 9–12 GPU-h (medido: 1.2 estudios/s → ≈ 48 min/época
× 16) + caché de evaluación y predicción ≈ 0.5 h. Total ≈ 12–14 h de RTX PRO 4500 ($0.72/h ≈ $9–10). Un pod (`rsna-compact`),
watchdog 16 h.

**Riesgos:** disco (caché train ≈ 300 GB en 450 GB: la de evaluación se construye tras borrar la de train); el camino de
inferencia puede no ser byte-idéntico a la caché del autor (`px384_v4` venía de `px_cache.py` sobre su caché nativa; aquí
mismas funciones vía `compact_infer`); sin cabeza de estados la compact puede rendir menos que la pública; el saldo de RunPod
($35, gasto $3.9/h con 5 pods) — el pod se detiene solo.

**Licencia:** código y anclas de heliosli = "no new open-source license is granted": solo para esta competición; **no se sube a
git ni se redistribuye**. El port vive en el disco local del pod; en git solo van nuestro notebook, `build_px_cache.py`
(importa el código de heliosli desde `public_assets`), la SPEC y `metrics.json`.

## 2. PLAN

- Notebook `exp066_compact_f0.ipynb` (nuevo; no copia del padre): celda 0 parámetros (`PHASE` smoke/train/eval) · 1 setup y
  sha256 de las fuentes de heliosli · 2 split, consenso y targets CSV en formato compact (disco local) · 3 port
  (`compact_train_port.py` = original + cambio de folds, en `/root/compact/src`) · 4 entrenamiento (subproceso del port) ·
  5 predicción fold 0 + gold con `Studies`/`Net` del trainer y blend con exp045 · 6 `metrics.json` · 7 verdict.
- `build_px_cache.py` (auxiliar de esta carpeta): caché px384_v4 desde DICOM (train: folds 1–4; eval: fold 0 + gold).
- `worker.sh` (en `orq/compact/`): caché train → smoke (1 época, 48 estudios, caché eval chica) → train 16 épocas → borrar
  caché train → caché eval → eval → commit local de `metrics.json` → detiene el pod. Watchdog 16 h.

## 3. VERDICT (números de `results/exp066_compact_f0/metrics.json`)

**Resultado:** worked / didnt / killed · **Fase alcanzada:** screen

| Métrica | exp045 SWA f0 | compact f0 | blend 0.5/0.5 | Δ blend − exp045 |
|---|---|---|---|---|
| pseudo L2 fold 0 | 0.8895 | | | |
| gold (veto) | 0.9201 | | | |
| GPU-h | 2.51 | | — | — |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
