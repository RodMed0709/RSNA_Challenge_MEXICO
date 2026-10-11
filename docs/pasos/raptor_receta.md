# Receta Raptor: cómo entrenarla nosotros (y la compact ConvNeXt de heliosli)

Fecha: 2026-10-11 (sesión 6). Estado: **terreno preparado, nada entrenado.** Todo lo descargado está en el volumen
(`/workspace/data/public_assets/`, rutas abajo). Este documento es la base para escribir la SPEC de un experimento
`expNNN_raptor_*`; no es una SPEC.

## 0. Resumen en 10 líneas

- **El código de entrenamiento de Raptor ES público**: `dreaddevelopment/knee-mri-training-the-twelve-finding-model`
  (script, 670 líneas). Su gemelo de inferencia es `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model` (0.924).
- **Las etiquetas originales de Raptor también son públicas**: `dreaddevelopment/rsna-knee-labels` →
  `labels_llm_soft.parquet` (el script lo busca por ese nombre). Ya estaba en `/workspace/data/labels/public/dreaddevelopment/`.
- **El corpus preprocesado público** (`knee-raptor-corpus` + `-ext`, 4407 estudios, 44×336×336 uint8) es la geometría
  *vieja* (44 cortes); con él el script corre casi tal cual. **Solo está bajada la parte 2 (`-ext`)**: la parte 1
  (16 GB) llenó la cuota del volumen de red y la borré (sección 6).
- Las versiones buenas (widedense v4 64 cortes = 0.924; nartaa 96 cortes/384 = 0.949) usan corpus más densos que
  **no están publicados**, pero el código de inferencia define exactamente cómo construirlos (sección 4).
- **No público**: las etiquetas propias de nartaa ("blendjev" = JEV + sus etiquetas), su supervisión externa con
  **2399 rodillas OAI** (acceso controlado NDA) y su notebook de entrenamiento. El 0.949 exacto no es reproducible;
  sí lo es la receta sin OAI.
- Compact heliosli: **`compact_train.py` es público** (embebido byte a byte en el notebook velci-kcd25-20, SHA-256
  verificado contra el manifest), pero sus **targets** (`probe_targets_opusv11b_*.csv`, categóricos de un LLM) **no**.
- Todos los pesos públicos cargan con 0 claves faltantes / 0 sobrantes (sección 8).
- Coste medido en RTX PRO 4500 (benchmark sintético): **CoAtNet-2 384 ≈ 3.5 GPU-h por fold** (16 épocas, 5 folds),
  ≈ 4.2 GPU-h el fine-tune all-data. CoAtNet 224 ≈ 1 GPU-h/fold. Compact ConvNeXt-S ≈ 9–12 GPU-h/fold.

## 1. Inventario: de dónde sale cada pieza

| Pieza | Fuente Kaggle | Licencia | En el volumen |
|---|---|---|---|
| Script de entrenamiento Raptor | kernel `dreaddevelopment/knee-mri-training-the-twelve-finding-model` | kernel público (Apache 2.0 por defecto de Kaggle) | `public_assets/dreaddevelopment__knee-mri-training-the-twelve-finding-model/` |
| Inferencia Raptor 0.924 (64 cortes) | kernel `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model` | ídem | `public_assets/dreaddevelopment__knee-mri-twelve-findings-from-a-single-model/` |
| Pesos widedense v4 (0.924) | dataset `dreaddevelopment/raptor-knee-widedense` | CC0 | `public_assets/dreaddevelopment__raptor-knee-widedense/` (`raptor_ft_coatnet_v4_full.pt`, `..._swa.pt`) |
| Etiquetas blandas de Raptor | dataset `dreaddevelopment/rsna-knee-labels` | CC0 | `labels/public/dreaddevelopment/rsna-knee-labels/` (ya estaba; no duplicado) |
| Corpus 44×336 parte 1 (3200 estudios) | dataset `dreaddevelopment/knee-raptor-corpus` | CC0 | `public_assets/dreaddevelopment__knee-raptor-corpus/`: **solo `all_ids.npy` y `all_masks.npy`**. `all_vols.npy` (15.9 GB; zip de 11.9 GB) **NO está**: al descomprimirlo el volumen llegó a su cuota (`Disk quota exceeded`, ~761 GB usados) y lo borré. Ver sección 6. |
| Corpus 44×336 parte 2 (1207 estudios) | dataset `dreaddevelopment/knee-raptor-corpus-ext` | CC0 | `public_assets/dreaddevelopment__knee-raptor-corpus-ext/` (`extra_vols.npy` 6.0 GB, …) |
| Inferencia nartaa 0.949 (96 cortes, mirror) | kernel `nartaa/rsna-knee-0949-anatomical-mirror` | kernel público | `public_assets/nartaa__rsna-knee-0949-anatomical-mirror/` |
| Pesos nartaa 0.949 (384) y 0.945 (224) | dataset `nartaa/rsna-knee-publication-swa-weights-20261007` | "other": uso de investigación/educativo **incluida la competición**; sin derechos sobre datos OAI | `public_assets/nartaa__rsna-knee-publication-swa-weights-20261007/` (ya estaba) |
| Par privado de nartaa | `nartaa/rsna-knee-oai-original384-s42-pair` | **privado (403)** | — |
| Etiquetas stevenleehans (v2, v4_blend) | `stevenleehans/rsna-knee-llm-report-labels` | CC0 | `labels/public/stevenleehans/…` (ya estaba) |
| Etiquetas pilkwang (YES/NO/UNK + conf) | `pilkwang/rsna-knee-llm-labels` | CC0 | `labels/public/pilkwang/…` (ya estaba) |
| Etiquetas lixin73 GPT-5.6-Sol | `lixin73/rsna-knee-llm-report-labels-sol56` | CC0 | `labels/public/lixin73/…` (ya estaba) |
| Etiquetas JEV (HYBRID) | dataset `riadmohamed42/jev-knee-labels` (+ kernel `riadmohamed42/jev-knee-labels-verification`) | CC0 | `public_assets/riadmohamed42__jev-knee-labels/`, `public_assets/riadmohamed42__jev-knee-labels-verification/` |
| Consenso de 6 fuentes de etiquetas | `karttikjangid05/rsna-knee-public-llm-label-sets-compared` | CC0 | `public_assets/karttikjangid05__rsna-knee-public-llm-label-sets-compared/` |
| Compact ConvNeXt-S v1 (4 folds + side_cls) | `heliosli/rsna-knee-compact-v1` | "other"; README: "Feel free to blend my model with yours!" | `public_assets/heliosli__rsna-knee-compact-v1/` |
| Compact r6cnxfsss | `heliosli/rsna-knee-compact-r6cnxfsss` | "other" (mismo formato) | `public_assets/heliosli__rsna-knee-compact-r6cnxfsss/` |
| Código compact (`compact_train.py`, `compact_infer.py`, `px_cache.py`, …) | embebido en kernel `bliverrigpdoge/velci-kcd25-20` (cell 4, dict `SOURCES`) | kernel público | `public_assets/heliosli__compact-src-from-velci/` (SHA-256 = manifest) |
| Código de decodificación/geometría que importa el compact (`training.*`, native cache, anclas SKM-TEA) | `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` (`cloud_code/`) | "other": "no new open-source license is granted" | `public_assets/heliosli__rsna-orthofoundation-fold0-epoch7-20261004/` (`cloud_code/`, `dependencies/`, `anatomy.pt` = pesos de anclas, SHA igual al `anchor_weights_sha256` del compact). Borré `weights.part000-009` (1.2 GB, modelo OrthoFoundation, no hace falta) para liberar cuota |
| Pesos "leg" de bliverrigpdoge (CoAtNet + slot/pos emb) | `bliverrigpdoge/rsna-knee-leg-r9all` (CC0). El pedido `bliverrigpdoge/rsna-knee-kcd25leg` da **403 (privado/borrado)**; este es el público equivalente | CC0 | `public_assets/bliverrigpdoge__rsna-knee-leg-r9all/` |
| Reproducción independiente (44 cortes, Steven+pilkwang 1:1) | kernel `nvap2011/coatnet-train` | kernel público | `public_assets/nvap2011__coatnet-train/` |

Scripts nuestros de esta preparación: `public_assets/_scripts/` (`verify_weights.py`, `verify_leg.py`,
`bench_raptor.py`, `dl_assets.sh`, y sus `.log`).

**Licencias.** Todo lo que se reentrena (código, corpus, etiquetas) es CC0 o kernel público. Los pesos "other"
(nartaa, heliosli) permiten uso en la competición; los usamos solo como referencia/inicialización/comparación, no se
redistribuyen. Los datos OAI no se tocan. Ojo con las reglas de la competición sobre datos externos si algún día
usamos pesos entrenados con OAI (los de nartaa lo están) en un envío.

## 2. Etiquetas blandas

### 2.1 Raptor original (`dreaddevelopment/rsna-knee-labels`)
- Formato: `StudyInstanceUID` + 12 columnas en [0,1]; 4349 filas (todos los no-gold). Son exactamente las que usa el
  script (`_find("labels_llm_soft.parquet")`). El md5 del csv (`96e6723f…`) coincide con el `labels_file` registrado en
  los metadatos del checkpoint de bliverrigpdoge (`d25_rsna_knee_labels_soft.csv`): también lo entrenó con estas.
- Un LLM lee el informe y emite una **probabilidad por hallazgo**, no un sí/no. Valores discretos observados (conteo
  sobre las 52 188 celdas): 0.05 (24 273, negativo o no mencionado), 0.2 (6 191), 0.8 (4 382), 0.95 (3 325),
  0.35 (3 157), 0.1 (3 149), 0.15, 0.45, 0.85, 0.08, 0.07, 0.76, 0.3, 0.82.
  Convención: negación/normal ≈ 0.05; positivo seguro ≈ 0.95; **dudoso ("sugiere", "posible") ≈ 0.8**;
  baja sospecha 0.2–0.45. Media por clase: ACL 0.19, MCL 0.12, MM 0.42, LM 0.22, MOA 0.28, LOA 0.20, PFOA 0.36,
  Effusion 0.33, Synovitis 0.45, Baker 0.17, Contusion 0.24, Fracture 0.12.
- Una sola fuente: no mezcla nada.

### 2.2 nartaa ("blendjev"): no es pública
El nombre del checkpoint (`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`) y los créditos dicen: etiquetas de
stevenleehans + pilkwang + JEV de riadmohamed42, **combinadas con sus propias etiquetas derivadas de informes**, más
supervisión externa enmascarada con 2399 rodillas OAI. La tabla final no se distribuye. La aproximación pública más
cercana es **JEV HYBRID**:

### 2.3 JEV HYBRID (`riadmohamed42/jev-knee-labels/HYBRID_labels_scores.csv`)
1. **Hard-swap por hallazgo** entre dos fuentes: por cada columna se queda con la fuente que puntuó más en gold-58.
   stevenleehans `llm_labels_v4_blend.csv` gana 9 (meniscos, los tres OA, Effusion, Synovitis, Baker, Contusion);
   pilkwang `report_labels_v2.csv` gana ACL, MCL y Fracture. Gold-74: 0.9172.
2. **Ensemble de texto JEV**: 6 lecturas del mismo modelo (prompts v5–v9, una con `steps=4`, una con `think=512`)
   promediadas con pesos por AUC del target. Solo: 0.8962.
3. **Mezcla global por celda**: `score = 0.95·hard_swap + 0.05·ensemble` → 0.9212 en gold-74.
   - `HYBRID_labels_binary.csv`: umbrales F1 por target (`HYBRID_thresholds.json`).
   - **Aviso SDD**: el hard-swap y los pesos se eligieron mirando gold-58; usar estas etiquetas implica que gold ya
     influyó en los targets. Para nosotros gold solo veta, así que esto hay que declararlo en la SPEC.

### 2.4 Convenciones de cada fuente (para mezclar)
- **pilkwang** `report_labels_v2.csv`: por hallazgo `X`, `X__conf`, `X__verdict` (YES/NO/UNK). YES → 0.82 (0.94 si
  enfático, 0.68 si leve); NO → 0.08; **UNK (no se menciona) → 0.28 con conf 0.05**. Sirven para ordenar; no están
  calibradas.
- **stevenleehans** v1/v2: **0.5 = "el informe no lo menciona"** (25 % de las celdas en v1); v2 rellena Synovitis
  indeciso a partir de Effusion. v4_blend = v2 mezclado con lixin73 GPT-5.6-Sol (0.8927 vs gold). Peso natural de
  confianza: `2·|p−0.5|`.
- **karttikjangid05** `consensus_labels.csv`: media de 6 fuentes con `n_sources`; su análisis concluye que en gold-58
  11/12 hallazgos no distinguen la mejor fuente de la mediana.
- nvap2011 (reproducción a 44 cortes) usa pilkwang + Steven v4 1:1.

**Recomendación para nuestra SPEC:** el experimento base usa `labels_llm_soft` de Raptor (reproducción fiel). Cambiar a
JEV HYBRID es **otra variable** → otro experimento.

## 3. Pipeline de datos (entrenamiento)

El script entrena sobre un **corpus precomputado** `(N, D, H, W) uint8` + máscara `(N, D)` (corte válido o no). No
lee DICOM durante el entrenamiento. Hay tres geometrías:

| Corpus | Slots (plano, fluid, nº cortes) | D | Rango de cortes | Lado | ¿Público? |
|---|---|---|---|---|---|
| v1 (`knee-raptor-corpus`) | Sag 12, Sag 10, Cor 8, Cor 6, Ax 8 | 44 | 15–85 % | 336 | **sí** (bajado) |
| v4 widedense (0.924) | Sag fluid 18, Sag no-fluid 14, Cor fluid 12, Cor 8, Ax 12 | 64 | 6–94 % | 336 | no (se reconstruye) |
| nartaa d96 (0.949) | Sag fluid 26, Sag no-fluid 22, Cor fluid 18, Cor 12, Ax 18 | 96 | 2–98 % | 384 nativo | no (se reconstruye) |

Construcción de un estudio (`build_study`, idéntica en las inferencias de dreaddevelopment y nartaa salvo constantes):
1. **Selección de series** por slot desde `train_series.csv` (`Anatomical_Plane`, `Fluid_Sensitive`): primera serie de
   ese plano con `Fluid_Sensitive == fluid` no usada todavía; si no hay, cualquier serie de ese plano no usada; si no
   hay ninguna, el slot queda a ceros (la máscara lo excluye). El axial (`fluid=-1`) no tiene preferencia.
2. **Orden de cortes**: proyección de `ImagePositionPatient` sobre la normal de `ImageOrientationPatient` (si falta,
   `InstanceNumber`).
3. **Cortes**: `n` archivos → `linspace(int(n·lo), int(n·hi)−1, k)` redondeado (lo/hi = 0.06/0.94 en v4, 0.02/0.98 en d96).
4. **Intensidad**: modality LUT; MONOCHROME1 invertido; **percentiles 2–98 calculados sobre todos los cortes elegidos
   de esa serie**, clip a [0,1].
5. **Recorte físico**: cuadrado centrado de **140 mm** (`round(140 / PixelSpacing)` px, acotado al tamaño de la imagen),
   `cv2.resize` INTER_AREA a 336 (v4) o 384 (d96), guardado como uint8.

Entrenamiento (`StudyWindows`):
- **2.5D**: ventana = 3 cortes consecutivos del volumen (c−1, c, c+1) como RGB, /255. Los centros válidos van del primer
  al último corte válido de la máscara **en todo el volumen concatenado** (así una ventana puede cruzar el borde
  entre dos slots; es lo que hace el original).
- Train: `k` centros aleatorios por estudio y época (con repetición si faltan). Eval: `k_eval` centros equiespaciados.
- Redimensiona cada ventana a `res` (bilinear). En d96 nartaa entrena a 384 con el campo completo y en inferencia
  recorta el centro a 320 px (`INPUT_RES=320`, `img_size=320` en timm).
- Normalización: `--norm imagenet` (media/desv. ImageNet). **El valor por defecto del script es `none`, pero las
  inferencias de widedense v4 y de nartaa usan `NORM="imagenet"`**: hay que pasar `--norm imagenet`.
- Aumento: solo ganancia de intensidad ×U(0.9, 1.1). **Sin flips** (la lateralidad es señal).
- ROI opcional (`--roi`, desactivado por defecto; requiere `roi_localize.py` y `roi_boxes.npz`, que **no son públicos**).

## 4. Modelo exacto

- Backbone `timm.create_model("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", pretrained=True, num_classes=0,
  global_pool="avg")` → 1024 dimensiones, 73.1 M de parámetros con la cabeza. Gradient checkpointing activado.
- Cabeza `RaptorClassifier` (MIL con atención por hallazgo):
  `h = LayerNorm(f)`; `a = softmax_K(Linear(1024→256) → Tanh → Dropout(0.2) → Linear(256→12))`;
  `pooled[b,n] = Σ_k a[b,k,n]·h[b,k]`; `logit[n] = <pooled[n], clsW[n]> + clsb[n]`.
- Variante 224 (nartaa, eficiencia 0.945): `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`, crop central 0.8 ("c08").
- Variante bliverrigpdoge "leg": misma cabeza + `slot_emb` (5 slots) + `pos` (posición relativa en el slot) + máscara
  de ventanas.

## 5. Hiperparámetros de entrenamiento

Del script (valores por defecto y la "TYPICAL RUN" de su cabecera) y de los metadatos de los checkpoints:

| Hiperparámetro | Valor |
|---|---|
| Épocas | 16 (`--epochs 16`; widedense: mejor época 12, SWA [12,13,14]; nartaa "t16", SWA [14,11,12]) |
| Batch | 8 estudios (`--bs 8`) × `k=12` ventanas = 96 imágenes 384² por paso |
| Ventanas | `k=12` train, `k_eval=24` validación; inferencia: 42 (64 cortes) / 94 (96 cortes) |
| Optimizador | AdamW, wd 0.02; LR backbone 3e-5, LR cabeza 1e-3 |
| Scheduler | OneCycleLR por paso, pct_start 0.15 |
| Loss | `BCEWithLogitsLoss(pos_weight)` sobre targets blandos; `pos_weight = clip((1−prev)/prev, 1, 10)` con `prev = clip(media de targets, 0.03, 0.7)` |
| Precisión | autocast bf16; clip de gradiente 3.0 |
| Selección | mejor época por **AUC macro en gold-58** (validación = gold) |
| "SWA" | media de pesos de las **3 mejores épocas por gold** (`--topk 3`) → `*_swa.pt` |
| Folds | hook opcional `--folds K --fold i --fold_file json` (solo saca OOF; la selección sigue en gold) |
| Fine-tune all-data | los pesos publicados (`*_full*`, `alldata`) se entrenan con **los 4349 no-gold** (sin fold reservado) |
| Semilla | 42 |
| Inicialización | ImageNet (timm). El SSL "Raptor" previo no se distribuye y no se usó en los pesos publicados |
| TTA (solo nartaa) | media de probabilidades vista normal + espejo anatómico L/R (sagital: invertir el orden de los 3 canales; coronal/axial: flip horizontal) |

**Incompatibilidades con nuestras reglas SDD** (hay que cambiarlas en el notebook, no son opcionales):
1. La selección de época y el topk de SWA usan gold. Nosotros: `SELECT_ON = "pseudo"` → validar en el fold reservado de
   `split_v1_seed42` (pseudo AUC contra L2 binarizada > 0.5), o fijar épocas/últimas-3 sin selección.
2. El notebook all-data no tiene validación: fijar el nº de épocas y las épocas del SWA con lo aprendido en el fold 0.

## 6. ¿Se puede correr en nuestro pod? (GPU 32 GB)

**Sí, con cambios mínimos.** Ruta A (reproducción con corpus público v1 44×336; falta bajar la parte 1, ver punto 8):
1. Exponer las entradas: el script busca bajo `/kaggle/input` (`_find`). Opción sin tocar código: crear en el pod
   `/kaggle/input/` con symlinks a `public_assets/dreaddevelopment__knee-raptor-corpus{,-ext}/` y a
   `/workspace/data/raw/train.csv`. Opción limpia: cambiar `root="/kaggle/input"` por un parámetro.
2. Etiquetas: `--labels` a `labels_llm_soft.parquet` **requiere pyarrow, que el venv no tiene** → `pip install pyarrow`
   en el venv, o cambiar `pd.read_parquet` por `pd.read_csv(...csv)` (1 línea).
3. Salidas: escribe en el directorio del script (`HERE`) → ejecutar una copia dentro de `experiments/expNNN_*/`.
4. `--norm imagenet` (ver 3). `--grad_ckpt` obligatorio a bs 8.
5. Selección por pseudo/fold en lugar de gold (sección 5) y fold file desde `load_split()`.
6. Regla del repo: se entrena en el notebook por celdas → copiar las clases/funciones del script a celdas (no `.py`).
7. I/O: el corpus son memmaps de 22 GB sobre el volumen de red (MFS). Para no saturarlo, copiar
   `all_vols.npy`/`extra_vols.npy` al disco local del pod (`/` tiene 40 GB libres) antes de entrenar, y `--workers 8`.
8. **Falta `all_vols.npy` (parte 1)**. El volumen de red tiene cuota y está casi lleno (~761 GB usados; `data/` = 725 GB).
   El 2026-10-11 hacia las 16:52–16:58 UTC la descompresión lo llenó (`Disk quota exceeded`) hasta que borré el zip y
   la extracción parcial. Para bajarlo sin riesgo: descargar el zip al disco **local** del pod que vaya a entrenar
   (`kaggle datasets download dreaddevelopment/knee-raptor-corpus -p /root/corpus`, 11.9 GB) y descomprimir allí
   (`all_vols.npy` 15.9 GB; pico 28 GB de 40) y copiar también `extra_vols.npy` a local. No ponerlo en el volumen sin
   antes liberar espacio o ampliar la cuota.

Ruta B (geometría buena, 64×336 o 96×384): escribir un notebook `build_corpus` que aplique `build_study` (sección 3)
a los 4407 estudios de `/workspace/data/raw/train_series/` y escriba `vols.npy (N,D,S,S) uint8`, `masks.npy (N,D)`,
`ids.npy` en el orden de `train.csv`. Tamaño: 96×384² × 4407 ≈ **62 GB** (64×336² ≈ 32 GB). **No cabe en el volumen sin liberar espacio** (cuota casi llena,
ver punto 8) ni en el disco local (40 GB) para el 96×384: hay que ampliar la cuota, liberar caches viejas (`cache/v1_256`,
`cache/v2_05mm`), o guardar a menor tamaño (p. ej. 96×320 tras recorte central ≈ 43 GB).
Tiempo: el `extra_meta.json` del corpus público registra 1970 s para 1207 estudios (44 cortes, un proceso); en paralelo
con 32–40 procesos debería bajar a <30 min, limitado por el volumen. Luego el entrenamiento es el mismo script con
`D=96`.

Ruta C (compact heliosli): **no reproducible tal cual**. `compact_train.py` necesita (a) la caché de píxeles
`px384_v4` construida por `px_cache.py` desde una caché nativa (`F:/rsnaknee/cache_native`) que genera el código de
`cloud_code/` con geometría de rodilla y anclas SKM-TEA (`anatomy.pt`), y (b) `--targets probe_targets_opusv11b_rr3_plv21_cdr.csv`
con columnas `prior`/`lr_prior` (categorías de estado del informe; ver `state_spec.json`), **no publicado**. Habría que
sustituir los targets (p. ej. `lr_prior` := etiqueta blanda) y reconstruir la caché: es un proyecto aparte. Config
registrada en el manifest: convnext_small (in22k_ft_in1k), 4 folds, 16 épocas, batch 2, 176 ventanas de entrenamiento,
crop 352 de 384, lr 1e-4, head_lr 1e-3, wd 0.05, `phys_mm 3.5`, vistas `v4`, arm `E_lrprior`, sin gold en
entrenamiento, más un clasificador de lado (`side_cls.pt`).

## 7. Tiempo de entrenamiento (medido)

Benchmark sintético de un paso de entrenamiento completo (forward+backward+AdamW, bf16, sin dataloader) en el pod
rsna-c, RTX PRO 4500 Blackwell (`_scripts/bench_raptor.log`):

| Config | s/iter | estudios/s | VRAM pico |
|---|---|---|---|
| CoAtNet-2 384, bs 8, k 12, grad-ckpt | 1.66 | 4.82 | 20.6 GiB |
| CoAtNet-2 384, bs 8, k 12, sin ckpt | OOM | — | >32 GiB |
| CoAtNet-2 224, bs 8, k 12, sin ckpt | 0.42 | 19.0 | 21.9 GiB |
| ConvNeXt-S 352, bs 2, 176 ventanas, grad-ckpt (aprox. compact) | 1.67 | 1.20 | 17.5 GiB |

Estimación (solo GPU; asume que el dataloader da abasto):
- **Raptor 384, por fold** (5 folds → ~3480 estudios de train, 16 épocas): 3480/4.82 ≈ 12 min/época → **≈ 3.3 h + eval ≈ 3.5 GPU-h**.
- **Fine-tune all-data** (4349 estudios, 16 épocas): ≈ 15 min/época → **≈ 4.2 GPU-h**.
- 5 folds + all-data ≈ 22 GPU-h. Solo screen fold 0 + all-data ≈ 8 GPU-h.
- Raptor 224 (eficiencia): ≈ 0.9–1 GPU-h por fold.
- Compact ConvNeXt-S (4 folds, ~3260 de train, 16 épocas): ≈ 45 min/época → ≈ 12 GPU-h/fold (≈ 8.6 h con `--compile`, que dice 1.4×).
- Referencia del autor: "about three hours on one 4090 for 16 epochs at 384" con el corpus de 3200 estudios. Coincide en orden de magnitud.
- **Ojo con la regla de 2 runs por pod**: a bs 8 una run ocupa 20.6 GiB, así que **cabe solo una CoAtNet-384 por pod**.
  Dos simultáneas exigen bs 4 (más lento por run) o la versión 224.

## 8. Verificación de pesos (pod rsna-c, CPU, torch 2.8.0, timm 1.0.30)

`_scripts/verify_weights.py` y `verify_leg.py`, log en `_scripts/verify_weights.log`. Se construye el modelo con la clase
de su código de inferencia, `load_state_dict(strict=False)`, y un forward con entrada aleatoria:

| Checkpoint | Clase | Faltan | Sobran | Forward | Metadatos |
|---|---|---|---|---|---|
| widedense `raptor_ft_coatnet_v4_full.pt` | RaptorClassifier (CoAtNet-2 384) | 0 | 0 | (1,4,3,384,384)→(1,12) | época 12, gold 0.9167 |
| widedense `raptor_ft_coatnet_v4_full_swa.pt` | ídem | 0 | 0 | ok | SWA [12,13,14], gold 0.9150 |
| nartaa `…_d96_r384_swa.pt` (0.949) | ídem, `img_size=320` | 0 | 0 | (…,320,320)→(1,12) | SWA [14,11,12], gold 0.9190 |
| nartaa `…_d96_c08_r224_swa.pt` (0.945) | ídem, `coatnet_rmlp_2_rw_224` | 0 | 0 | (…,224,224)→(1,12) | SWA [15,13,12], gold 0.9147 |
| bliverrigpdoge `loop_R9_all.pt` | PackInfer (de su `pack_submit.py`) | 0 | 0 | ok | 5430 pasos, labels = Raptor soft |
| heliosli compact-v1 `E_lrprior_s0_f0..f3.pt` | `compact_train.Net` (convnext_small, crop 352, spatial+extent, state_spec) | 0 | 0 | — | 50.1 M params |
| heliosli compact-r6cnxfsss `f0..f3` | ídem | 0 | 0 | — | — |
| `side_cls.pt` (ambos) | `side_common.build_side_net()` strict | — | — | — | All keys matched |

Notas: el CoAtNet `rmlp` de nartaa a 320 requiere crear el backbone con `img_size=320`; sin eso falla el forward
(400 vs 576 en la atención), no la carga. El state_dict de "leg" usa el prefijo `enc.` y no carga en RaptorClassifier;
con su clase PackInfer carga sin errores.

## 9. Qué falta (lo que no es público)

1. Etiquetas propias de nartaa ("blendjev") y la tabla final de targets.
2. Supervisión externa OAI de nartaa (2399 rodillas, acceso controlado NDA) y su caché; su notebook de entrenamiento
   y el par privado `rsna-knee-oai-original384-s42-pair`.
3. Los corpus densos de dreaddevelopment (v4 64×336, v5+ 96/384): solo hay pesos; se reconstruyen con `build_study`.
4. `roi_localize.py` / `roi_boxes.npz` (palanca ROI del script, desactivada por defecto).
5. El checkpoint SSL "Raptor" (no se usó en los pesos publicados).
6. Compact: targets `probe_targets_opusv11b_*` y la caché `px384_v4` (el código para construirla sí está, pero depende
   de rutas locales del autor y de `cloud_code`).
7. `bliverrigpdoge/rsna-knee-kcd25leg`: privado o borrado (403).

## 10. Propuesta de experimentos (para la SPEC; no lanzados)

1. `expNNN_raptor_v1corpus`: script tal cual (corpus público 44×336, `labels_llm_soft`, CoAtNet-2 384, 16 épocas,
   `--norm imagenet`), selección pseudo en fold 0. Sirve de control de que nuestra adaptación reproduce el pipeline
   (≈ 3.5 GPU-h).
2. `expNNN_raptor_d96`: misma receta sobre un corpus 96×384 construido por nosotros (una variable: la geometría del corpus).
3. Solo después: etiquetas JEV HYBRID (una variable), TTA espejo (una variable), all-data + SWA de épocas fijas para el envío.
