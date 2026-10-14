# exp064_raptor_ft_nartaa_aug_light

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp044_raptor_ft_nartaa (all-data: SWA [3,4,5] gold 0.91924; init nartaa gold 0.91907; 1.05 GPU-h; `swa_alldata.pt`) |
| Fecha | 2026-10-13 |
| Rama | exp064_raptor_ft_nartaa_aug_light |
| Estado | spec |

Idea: Track B (desde 0.954): exp044 + mri_artifact_aug light (all-data)

Track B = entrenar a partir del ganador público 0.954 ("blend-mine": rank blend 0.5/0.5 de los pesos SWA CoAtNet-2 Raptor de
nartaa + el modelo "compact" convnext_small E_lrprior), no desde nuestra base. Pedido explícito de Rodrigo (2026-10-13).
El modelo que se toca es el miembro Raptor: se afinan los pesos de nartaa (receta exp044) y el resultado entra en el grupo
raptor del blend-mine como rank mean nartaa + este modelo (estilo exp042b).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si al fine-tune de exp044 (pesos SWA de nartaa → etiquetas de consenso, all-data, 6 épocas) le añado la
augmentación de artefactos de adquisición de RM `mri_artifact_aug` con el preset **light** (por unidad estudio×slot: bias de
bobina, anisotropía in-plane, motion y ghosting en k-space, ruido Rician, gamma; P(unidad con ≥ 1 artefacto) = 41 %), el
modelo resultante generaliza mejor al test oculto (otros escáneres/sitios) y el blend-mine con raptor = rank mean
(nartaa + este modelo) sube en el LB público frente al mismo blend con exp044.

**Única variable cambiada:**

| Parámetro o celda | Padre (exp044) | Este experimento |
|---|---|---|
| `AUG_STRENGTH` (aug de artefactos RM en train, `to_input`, antes de la ganancia y la normalización) | sin aug (equivale a `"off"`) | `"light"` (`PRESETS["light"]` de `orq/p7c_augs/mri_aug.py`, copiado verbatim en la celda 3) |

El plumbing necesario no cambia el cómputo del padre: `SLOT_SIZES = [26, 22, 18, 12, 18]` / `SLOT_OF` (igual que exp045) y
`StudyWindows` devuelve el slot del corte central por ventana; con `"off"` la función devuelve el mismo tensor (identidad
exacta, test 23/23 del README de p7c_augs). Evaluación/inferencia sin aug.

Todo lo demás igual a exp044: init pesos SWA de nartaa (sha256 `7e5315da…`, carga estricta), all-data (4349 no-gold),
6 épocas, LR bb 1e-5 / cabeza 1e-4, SWA épocas fijas [3, 4, 5], corpus d96/384 con recorte central 320, `--norm imagenet`,
K 12 / K_EVAL 24, bs 8, AdamW wd 0.02, OneCycle pct 0.15, BCE blanda con pos_weight, bf16, clip 3.0, semilla 42,
etiquetas de consenso de exp031.

**Por qué light y no strong (decisión fijada aquí):** se afina un modelo ya convergido (nartaa, 0.949 LB) con LR bajos y solo
6 épocas. El preset strong toca el 81 % de las unidades con magnitudes que alcanzan las del harness exp040; con tan pocas
épocas no hay margen para que el modelo se re-adapte, y el riesgo principal es la deriva (alejarse del óptimo de nartaa y
perder lo que aporta al blend). Light (41 % de unidades, magnitudes por debajo del harness exp040) es la perturbación mínima
que todavía regulariza. Lecciones internas: p = 1 fue malo (sub045_1), aug de intensidad fuerte bajó gold (exp046 −0.0105).
Strong queda como posible experimento aparte solo si light funciona.

**Origen / citas:**
- Pérez-García F, Sparks R, Ourselin S. TorchIO: a Python library for efficient loading, preprocessing, augmentation and
  patch-based sampling of medical images in deep learning. Comput Methods Programs Biomed 2021;208:106236.
  doi:10.1016/j.cmpb.2021.106236 (RandomMotion, RandomGhosting, RandomBiasField, RandomAnisotropy, RandomNoise, RandomGamma).
- Zhang L, Wang X, Yang D, et al. Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep
  Stacked Transformation (BigAug). IEEE Trans Med Imaging 2020;39(7):2531–2540. doi:10.1109/TMI.2020.2973595.
- Shaw R et al. MRI k-Space Motion Artefact Augmentation. MIDL 2019, PMLR 102:427–436 (sin DOI).
- Interno: `orq/p7c_augs/README.md` (diseño, tests, presets), `docs/conocimiento/02_augmentations.md` §3.

**Contaminación (explícita; condiciona qué métrica decide):**
- Los pesos de nartaa se entrenaron con **todos** los estudios de train (all-data; incluidos los de nuestro fold 0) y gold-58
  con peso 3 / reutilizado para selección. Para cualquier modelo inicializado con nartaa, la pseudo AUC de fold 0 y la gold
  AUC están **contaminadas**: no miden generalización y no son comparables con modelos desde ImageNet.
- Por eso no se corre screen de fold 0 y las métricas internas son **solo sanidad** (no deciden).
- Pesos con supervisión externa OAI: revisar reglas de datos externos antes de un envío final / premio (igual que exp044).

**Efecto esperado:** Δ LB del blend ≈ 0 a +0.003 (prior baja: aug de artefactos in-distribution rara vez mueve la AUC; el valor
esperado está en robustez a sitio). Coste de entrenamiento ≈ igual al padre (+ < 3 %).

**Criterios (no cambian después de correr):**
- **Decide el LB público** con el kernel `ens-exp064` (blend-mine estilo exp042b: grupo raptor 0.5 = rank mean de nartaa +
  exp064 `swa_alldata.pt`; grupo compact 0.5) contra el **mismo kernel con exp044** (`ens-exp042b`, LB pendiente al escribir
  esta SPEC; se copia de `kaggle_queue/scores.csv`).
  - `worked` si LB(ens-exp064) ≥ LB(ens-exp042b) + 0.003 **y** ≥ 0.954 (blend-mine base).
  - `didnt` si LB(ens-exp064) < LB(ens-exp042b) + 0.003.
  - Nota: si ens-exp042b < 0.954 (afinar nartaa no ayuda en sí), exp064 solo puede ser `worked` si además supera 0.954.
- **Sanidad (no decide; contaminada):** gold del SWA all-data ≥ 0.90 y ≥ gold SWA del padre − 0.02 (0.89924). Si falla →
  `killed` por deriva/olvido y no se construye kernel.
- **Coste:** GPU-h ≤ 1.5 × presupuesto; si no, se para y se escribe el verdict.
- El envío lo decide el orquestador de colas (15/16-oct); este experimento solo deja el kernel validado como candidato.

**Presupuesto:** all-data 1 run ≈ 1.1 GPU-h (padre 1.05) + smoke ≈ 0.05 · RTX PRO 4500 ($0.72/h) · ≈ $1.

**Riesgos:** deriva fuera del óptimo de nartaa (mitigado con light); el LB público tiene ruido (±0.002–0.005) y un solo envío;
aug sobre uint8 ya normalizado p2–98 (no es el espacio crudo); motion/ghost con FFT circular (réplicas débiles en el borde).

## 2. PLAN

- Celda 0 (parámetros, celda 2 del .ipynb): `AUG_STRENGTH = "light"`, `NOTES`.
- Celda 3 (modelo, celda 8 del .ipynb): `SLOT_SIZES`/`SLOT_OF` (+ assert `len == D`); contenido de `mri_aug.py` verbatim;
  `StudyWindows.__getitem__` devuelve además el slot por ventana; `to_input(x_u8, gain, slot=None, train=False)` aplica
  `mri_artifact_aug(x, slot, AUG_STRENGTH)` solo si `train` y `AUG_STRENGTH != "off"`, antes de la ganancia; mide el tiempo
  de la aug (`torch.cuda.synchronize`) en los primeros 20 pasos.
- Celda 4 (entrenamiento, celda 10 del .ipynb): desempaquetar `(x, g, idx, s)`; `to_input(x, g, s, train=True)` en train;
  `predict` ignora el slot (eval sin aug).
- Celda 6 (metrics, celda 14 del .ipynb): `mri_aug_preset`, `aug_strength`, `mri_aug_ms_per_step`.
- Ninguna otra.
- Smoke `SMOKE = True` (64 estudios, 1 época, all-data; imprime ms/paso de la aug). Validación visual del preset light:
  `orq/p7c_augs/mri_aug_grid.png` (misma función, sin cambios). Luego run real all-data en pod
  `rsna-trackb` (worker `orq/trackb/worker_b.sh`, flock, auto-stop).
- Después: dataset privado `rsna-knee-exp064-weights` (mk_dataset.sh) + kernel privado `ens-exp064` (make_ens_generic.py,
  cfg = exp042b cambiando exp044 → exp064); correr a COMPLETE y validar con `check_kernel.sh`. Sin envío.

## 3. VERDICT (números de `results/exp064_raptor_ft_nartaa_aug_light/alldata/metrics.json` y LB)

**Resultado:** pendiente · **Fase alcanzada:** spec

| Métrica | Padre (exp044 / ens-exp042b) | Este (exp064 / ens-exp064) | Δ |
|---|---|---|---|
| LB público blend-mine (decide) | | | |
| gold SWA all-data (sanidad, contaminada) | 0.91924 | | |
| GPU-h | 1.05 | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
