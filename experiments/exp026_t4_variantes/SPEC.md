# exp026_t4_variantes

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp025_perfil_raptor (perfilado en RunPod; aquí se mide en la T4 real de Kaggle) |
| Fecha | 2026-10-11 |
| Rama | exp026_t4_variantes |
| Estado | worked |

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

## 3. VERDICT (números de `results/exp026_t4_variantes/metrics.json`)

**Resultado:** worked (la medida salió) · **H1:** rechazada (channels_last apenas ayuda en T4) ·
**H2:** confirmada (sin TTA −50 % de runtime, ΔAUC dentro de ±0.002).

Kernel `medellinrodrigo/exp026-t4-variantes` v1: 2 × Tesla T4, 4 vCPU, 33.7 GB RAM, torch 2.11.0+cu128,
timm 1.0.29 (imagen más reciente de Kaggle, sin fijar), `nproc` = 4. 2 297 s en total. Lector instrumentado
idéntico byte a byte al del notebook (3/3).

**El notebook sí usa las 2 T4:** un hilo consumidor por GPU; V0 repartió 79/71 estudios (cuda:0/cuda:1),
`nvidia-smi` 87.8 % / 87.9 % de utilización media. La cuda:1 fue ~10 % más lenta por estudio en todas las
corridas.

Serie en frío, 30 estudios aparte, ms por estudio (una T4): list 55 · hdr_read 65 · select 1 · px_read 111
(58 MB/estudio, 525 MB/s) · decode 70 · pct_norm 55 · crop_resize 133 · **prep 494** · windows 41 ·
**fwd_plain 1 040 · fwd_mirror 1 048** · total 2 624. La T4 es **5.3× más lenta** que la PRO 4500 en el
forward (1 040 vs 196 ms por pasada).

Pipeline del notebook, 150 estudios (58 gold + 92 fold 0). "steady" = reloj sin el exceso de autoajuste
de cudnn del primer estudio de cada hilo; minutos a 1 300 estudios = steady × 1 300 + ~0.8 min fijos
(imports 14 s, carga de modelos 7 s, autoajuste 18–36 s):

| Var | s/est. reloj | steady disco | steady RAM | autoajuste s | min @1300 | AUC gold | pseudo 150 | Pearson vs V0 | media \|Δp\| | ΔAUC gold vs V0 [IC95] | Δ pseudo 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 (frío) | 1.329 | 1.201 | — | 19 | 26.8 | — | — | — | — | — | — |
| V0 | 1.320 | 1.200 | 1.169 | 18 | 26.8 | 0.9229 | 0.9267 | 1 | 0 | 0 | 0 |
| V1 | 1.336 | 1.162 | 1.141 | 26 | 26.0 | 0.9229 | 0.9267 | 1.00000 | 2.5e-5 | +0.0000 | −0.0000 |
| V2 | 0.761 | **0.594** | 0.565 | 25 | **13.7** | 0.9219 | 0.9282 | 0.99798 | 0.0101 | −0.0010 [−0.0045, +0.0026] | +0.0015 |
| V3 | 1.116 | 0.878 | 0.858 | 36 | 19.9 | 0.9214 | 0.9278 | 0.99862 | 0.0068 | −0.0015 [−0.0045, +0.0017] | +0.0011 |
| V4 | 0.768 | 0.600 | 0.563 | 25 | 13.8 | 0.9196 | 0.9255 | 0.99716 | 0.0103 | −0.0034 [−0.0079, +0.0006] | −0.0012 |
| V0b | 1.331 | 1.201 | 1.175 | 20 | 26.8 | 0.9229 | 0.9267 | 1.00000 | 1.4e-5 | +0.0000 | +0.0000 |

(AUC infladas: Raptor se entrenó con todo train. V0b ≈ V0 en tiempo y predicción: no hay deriva por
temperatura; la diferencia V1–V0 en |Δp| es del mismo tamaño que el ruido de una corrida a otra.)

**Criterios:**
- H1: V1/V0 steady = 0.968 (−3.2 %; en reloj a 150 estudios 1.336 vs 1.320, peor por 8 s más de
  autoajuste) → no llega a ≤ 0.90. Predicciones idénticas (Pearson 1.00000, |Δp| 2.5e-5, ΔAUC 0). **Rechazada**:
  el −12 % de la PRO 4500 (Blackwell) no se traslada a la T4 (Turing). Ahorro a 1 300 estudios ≈ −0.7 min.
- H2: V2/V0 steady = 0.495 (−50.5 %) ≥ 35 % y |ΔAUC| gold 0.0010, pseudo 0.0015 ≤ 0.005. **Confirmada.**
- Sanity: AUC gold V0 0.923 > 0.85. OK.
- Eficiencia contra V1 (1 min ≈ 0.00086 de AUC): V2 ahorra 12.3 min ≈ 0.0106 de AUC equivalente frente a
  ΔAUC −0.0010 (gold) / +0.0015 (pseudo) → V2 gana con mucho margen. V4 cuesta lo mismo que V2 y pierde más
  AUC (dominada). V3 ahorra 6.2 min (≈ 0.0053) con ΔAUC similar a V2 → dominada por V2.

**Disco y CPU:** V0 en frío (primera lectura) = 1.329 s/estudio vs 1.320 en caliente: el disco de Kaggle no
importa mientras manda la GPU. Con V2/V4 (~0.6 s/estudio) las GPU casi no esperan (2.5 s de espera total
por GPU en caliente; RAM 0.565 vs disco 0.594 = 5 %); la prep en frío a ese ritmo no está medida en pool
(la serie da 0.49 s/estudio con 1 proceso; el pool tiene 4).

**Qué aprendimos:** en Kaggle el runtime es el forward de la T4 (≈ 1.04 s por pasada de 94 ventanas);
la TTA espejo cuesta la mitad del runtime y aporta ≈ 0 en nuestros datos (±0.002, dentro del ruido de 58
gold). channels_last no sirve en T4. Reducir ventanas a 47 ahorra lo mismo que quitar la TTA pero pierde más.
cudnn.benchmark re-autoajusta en cada hilo nuevo (18–36 s por corrida, una vez por GPU).

**Recomendación:** para AUC enviar **V1** (idéntico a V0, −0.7 min; mantiene la TTA que el autor validó
en LB). Para eficiencia enviar **V2** (sin TTA, ≈ 13.7 min estimados contra 26.8 de V0, ΔAUC ≈ 0).
Siguiente: V2 + quitar `cudnn.benchmark`/autoajuste por hilo, TensorRT/compile en T4, o fp16 puro sin autocast.
