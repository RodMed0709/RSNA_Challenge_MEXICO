# exp026_t4_variantes

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp025_perfil_raptor (perfilado en RunPod; aquí se mide en la T4 real de Kaggle) |
| Fecha | 2026-10-11 |
| Rama | exp026_t4_variantes |
| Estado | running |

Idea: medir en un kernel de Kaggle (GPU T4 ×2, 4 vCPU, disco real, sin internet, misma imagen Docker
que el notebook) la inferencia Raptor del notebook público `nartaa/rsna-knee-0949-anatomical-mirror`
por etapa y en reloj de pared, y comparar variantes baratas del forward sobre los mismos estudios.
Alimenta la elección de qué enviar para AUC y qué para el premio de eficiencia.

## 1. SPEC

**Hipótesis:**
- H1: `channels_last` (modelo + ventanas) baja el runtime de reloj ≥ 10 % sin cambiar las predicciones.
- H2: quitar la TTA espejo baja el runtime de reloj ≈ 45 % con un cambio pequeño de AUC.

**Variantes (mismos estudios, mismos pesos, fp16 autocast como el notebook):**

| Variante | Qué cambia contra V0 |
|---|---|
| V0 | tal cual el notebook: 94 ventanas, TTA espejo anatómico (2 pasadas), trozos de 16 |
| V1 | V0 + `channels_last` (modelo y cada trozo de ventanas) |
| V2 | V1 sin TTA (1 pasada de 94 ventanas) |
| V3 | V1 con TTA solo en las ventanas sagitales: pasada normal de 94 + backbone sobre las ~47 ventanas sagitales espejadas; la bolsa espejada = features espejadas en sagitales + features normales en coronal/axial; promedio de las dos sigmoides (≈ 141 ventanas en vez de 188) |
| V4 | V1 con 47 ventanas (`K_EVAL = 47`, cada 2) y TTA |

Además se repite V0 al final (V0b) para medir deriva/ruido de la GPU, y V0 corre una vez en frío
(primera lectura del disco) y otra en caliente.

**Qué se mide:**
- Por etapa, en serie y en frío, sobre 30 estudios aparte (no se reusan): `list`, `hdr_read`, `select`,
  `px_read`, `decode`, `pct_norm`, `crop_resize` (instrumentación de exp025 del lector `study_lut`,
  verificada byte a byte contra el builder del notebook) + `windows`, `fwd_plain`, `fwd_mirror` en una T4.
- Por variante, en reloj de pared: el pipeline del notebook (Pool de preparación con `nproc` del notebook +
  un hilo consumidor por GPU) sobre 150 estudios; s/estudio, tiempo de la preparación, ocupación y espera
  de cada GPU, estudios por GPU, `nvidia-smi` por GPU. Y el mismo consumo desde RAM (solo GPU) para ver el
  techo si la preparación no limitara.
- Uso de las 2 T4: `torch.cuda.device_count()`, estudios procesados por cada GPU y utilización media.

**Estudios:** de `train_series` de la competición (en el input del kernel), `train_series.csv` en lugar de
`test_series.csv`. 150 principales = los 58 gold (`split_v1_seed42`, fold −1) + 92 de fold 0 (semilla 0),
30 de fold 0 para el perfilado serie en frío y 2 de calentamiento. Las etiquetas no van al kernel: las
AUC se calculan fuera con `exp026_studies.csv` (gold de `train.csv`, L2 =
`llm_labels_v4_blend.csv` binarizada > 0.5).

**Métricas de predicción contra V0:** correlación de Pearson de las probabilidades (150 × 12), media y
máximo de |Δp|; AUC macro gold (58) y pseudo AUC vs L2 > 0.5 (150 y los 92 de fold 0); ΔAUC gold con
bootstrap pareado (2000). Las AUC absolutas están infladas (Raptor se entrenó con todo train); vale la
diferencia entre variantes.

**Criterios (fijados antes de correr):**
- H1 confirmada si s/estudio de reloj (pipeline, caliente) de V1 ≤ 0.90 × V0 **y** Pearson ≥ 0.9999,
  media |Δp| < 1e-3 y |ΔAUC gold| < 0.002. Rechazada si no.
- H2 confirmada si V2 ≤ 0.65 × V0 en s/estudio de reloj (≥ 35 % de ahorro) **y** |ΔAUC| ≤ 0.005 en gold y
  en pseudo-150. Si el ahorro sale ≥ 35 % pero el AUC cae más, se reporta "más rápido pero caro".
- Regla de recomendación para eficiencia: 1 min de runtime ≈ 0.00086 de AUC en la métrica; una variante
  le gana a V1 en eficiencia si su ΔAUC estimado (gold y pseudo, contra V1) es mayor que
  −0.00086 × minutos ahorrados extrapolados a 1300 estudios.
- Sanity: AUC gold de V0 > 0.85 (si no, el kernel no reproduce el notebook).

**Excepción a la regla 4 (notebook):** como en exp025, el experimento es un `.py`
(`t4_variants.py`, el arnés que corre en Kaggle) más `make_kernel.py` (arma el notebook del kernel
inyectando la celda de código del notebook público, que no se copia al repo) y `analyze.py` (AUC y
correlaciones fuera de Kaggle). No entrena nada ni toca `src/knee/`.

**Riesgos:** el page cache de Kaggle hace que solo la primera pasada lea el disco en frío; las variantes
se comparan en caliente y el frío se mide con V0 y con el perfilado serie. El orden fijo de las variantes
puede sesgar por temperatura de la T4 (V0b lo controla). 58 gold es poco: el bootstrap da el ruido del ΔAUC.

## 2. PLAN

- `make_kernel.py`: baja (o reusa) `nartaa/rsna-knee-0949-anatomical-mirror`, incrusta su celda de código
  en base64 + `t4_variants.py` + la lista de estudios; kernel privado `medellinrodrigo/exp026-t4-variantes`
  con las mismas fuentes (`nartaa/rsna-knee-publication-swa-weights-20261007` + competición), T4, sin
  internet, misma `docker_image` que el notebook.
- Orden dentro del kernel: carga en 2 GPU → calentamiento → serie en frío (30) → V0 frío (150, guarda los
  volúmenes en RAM) → V0, V1, V2, V3, V4, V0b en caliente (pipeline completo y solo-GPU desde RAM).
- Salidas del kernel: `exp026_results.json`, `preds_<V>.csv`. `analyze.py` → `results/exp026_t4_variantes/metrics.json`.
- Deja en `scratchpad/kaggle_exp026/` los kernels de envío de V1 y de la mejor variante rápida (sin enviar).

## 3. VERDICT

(pendiente)
