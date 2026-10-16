# exp075_raptor_ssl_dino

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial (fold 0: SWA pseudo L2 0.8895, gold 0.9201, 2.51 GPU-h) |
| Fecha | 2026-10-16 |
| Rama | exp075_raptor_ssl_dino |
| Estado | run |

Idea: init del encoder CoAtNet-2 = ImageNet + preentreno autosupervisado DINO (sin etiquetas) sobre cortes 2D de los
estudios de train, sin datos externos; después fine-tune fold 0 con la receta exp045 exacta.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** la brecha con nartaa (0.949 LB con init OAI) frente a nuestra receta (0.939–0.943 LB desde ImageNet) es la
inicialización, no la receta (exp071 mostró que init nartaa + receta exp045 da 0.944). Un preentreno continuado
autosupervisado sobre RM de la competición (como hars25: pocas épocas) puede dar un init mejor que ImageNet sin datos
externos.

**Única variable cambiada:**

| Parámetro o celda | Padre (exp045) | Este experimento |
|---|---|---|
| Init del encoder (celdas 0, 3 `build_model`, 4 bloque SSL) | ImageNet `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` | ImageNet + DINO SSL continuado: 3 épocas × 16 ventanas 2.5D aleatorias por estudio (3 cortes → RGB, recorte central 320 del corpus 96×384) sobre los estudios de **train de folds 1–4** (3477), **sin fold 0 ni gold (fold −1)**, sin etiquetas. Se pasa solo el backbone del alumno (carga estricta) |

Detalle DINO (fijado antes de correr): 2 vistas globales 320 por ventana (sin crops locales: las particiones de
CoAtNet dependen del tamaño), aug en GPU (recorte aleatorio área 0.35–1, rot ±15°, volteo, ganancia/contraste 0.7–1.3,
gamma ×/÷1.5, desenfoque p 0.3, ruido p 0.3); cabeza MLP 2048 → 256 → 8192 protos normalizada; profesor EMA coseno
0.996 → 1; centrado 0.9; temperaturas alumno 0.1 / profesor 0.04; capa final congelada el 10 % inicial; AdamW
LR backbone 5e-5 / cabeza 5e-4, wd 0.04, warmup 5 % + coseno; batch 64 ventanas × 2 vistas; bf16; tope duro 150 min.

Todo lo demás igual a exp045 fold 0: split `split_v1_seed42`, semilla 42, etiquetas de consenso (Raptor + L2 + L3 + L4),
aug espacial, 16 épocas, bs 8 × K 12 (K_EVAL 24), AdamW wd 0.02, BB_LR 3e-5 / HEAD_LR 1e-3, OneCycle pct 0.15, clip 3,
SWA = top-3 épocas por pseudo L2. `SELECT_ON = "pseudo"`; gold nunca selecciona.

**Efecto esperado:** Δ pseudo L2 fold 0 +0.000 a +0.010; costo SSL ≈ 1–2.5 GPU-h + fine-tune ≈ 2.5 GPU-h.

**Criterios (no cambian después de correr):**
- `worked` si pseudo L2 fold 0 SWA ≥ 0.8945 (exp045 0.8895 + 0.005).
- `killed` si gold SWA < 0.9001 (exp045 0.9201 − 0.02).
- Si no, `didnt`. Si `worked` → all-data después (no hoy).
- Coste: SSL ≤ 6 GPU-h + fine-tune ≈ 2.5 GPU-h.
- Kaggle (medición pedida por Rodrigo, no decide el verdict): kernel solo `exp075-solo` = copia del kernel exp071-solo con
  un cambio (checkpoint = SWA fold 0 de exp075). Referencias LB: exp031 f0 solo 0.934, exp045 5 folds solo 0.941. Se envía
  hoy (16-oct, último lugar del día) aunque salga peor.

**Presupuesto:** SSL ≤ 6 GPU-h · fine-tune ~2.5 GPU-h · GPU A100 SXM 80 GB (pod `rsna-ssl`, $1.79/h).

**Riesgos:** colapso del DINO (se vigila la entropía del profesor en el log); que 16 épocas de fine-tune borren lo
aprendido; pocas épocas SSL para mover un CoAtNet-2.

## 2. PLAN

- Celdas que cambian: 0 (parámetros SSL_*), 3 (`build_model` carga `SSL_STATE` si existe), 4 (bloque SSL DINO antes del
  fine-tune; el fine-tune es el de exp045 sin tocar), 6 (`metrics.json` guarda bloque `ssl` y suma sus GPU-h; copia el SWA
  fold 0 a `kaggle_export/` para el kernel). Ninguna otra.
- Smoke con `SMOKE = True` (SSL 48 estudios, 1 paso; fine-tune 64/48 1 época): OK 14:17 UTC (pseudo 0.503, sin errores).
- Run: cadena `orq/ssl/chain_gpu.sh` en el pod (papermill `SMOKE False`), luego `orq/ssl/finish_cron.sh` en rsna-cron
  (dataset + kernel + cola + envío).

## 3. VERDICT (números de `results/exp075_raptor_ssl_dino/metrics.json`)

Pendiente.
