# exp071_raptor_nartaa_init_aug

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial (all-data: SWA [13,14,15], gold 0.9186, 2.81 GPU-h; LB 5 folds 0.941) |
| Fecha | 2026-10-14 |
| Rama | exp071_raptor_nartaa_init_aug |
| Estado | run · all-data hecho, LB pendiente |

Idea: V1 del plan de preentreno externo (`docs/pasos/plan_preentreno_externo.md` §3.2): receta exp045 (aug espacial)
entrenada desde los pesos de nartaa (supervisión externa OAI/MOAKS) en vez de ImageNet, all-data.

Encargo de Rodrigo (14-oct): probar todas las vías de preentreno/datos externos. Antes de reclamar el número se revisó que
no estuviera duplicado: pods vivos (rsna-r44s corre `sub044_1_semilla` = receta exp044 con semilla 7, **sin** aug espacial;
rsna-c67/c68 = compact exp067/068), claims del volumen (último exp070) y ramas de origin (exp064 = exp044 + aug de
artefactos light). Ninguno combina init nartaa + receta completa exp045.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** la brecha de ~0.008–0.010 LB entre nuestra receta Raptor (exp045 5 folds 0.941, exp056 all-data 0.943) y
nartaa (0.949 con TTA) se debe sobre todo a la inicialización con supervisión externa OAI/MOAKS (exp044, afinar nartaa:
blend 0.953 frente a 0.939–0.943 desde ImageNet). Si la receta completa de exp045 (16 épocas, LRs normales, aug espacial)
parte de los pesos de nartaa, el modelo hereda la representación OAI y suma la regularización de exp045, y su LB solo sube.

**Única variable cambiada:**

| Parámetro o celda | Padre (exp045 all-data) | Este experimento |
|---|---|---|
| Inicialización (`INIT_CKPT`, celda 0 y `build_model` en celda 3) | ImageNet (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, timm) | pesos SWA de nartaa `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (`/workspace/data/public_assets/nartaa__rsna-knee-publication-swa-weights-20261007/`), sha256 `7e5315da…` verificado, carga estricta (código de carga = exp044, misma arquitectura) |

Todo lo demás igual a exp045 all-data: 4349 estudios no-gold, etiquetas de consenso (Raptor + L2 + L3 + L4), corpus 96×384
en `/root/corpus96_384`, recorte central 320, aug espacial (espejo L/R p 0.5 + afín rot ±10°, escala 0.9–1.1, tras ±6 %),
16 épocas, bs 8 × K 12 (K_EVAL 24), AdamW wd 0.02, BB_LR 3e-5 / HEAD_LR 1e-3, OneCycle pct 0.15, bf16, clip 3.0, semilla 42,
SWA sobre las épocas fijas [13, 14, 15] (las mismas del all-data de exp045). La celda de sanidad del init sobre gold (exp044) se
copia tal cual (solo imprime).

**Por qué no hay screen de fold 0 (y por qué no aplica el criterio Δ pseudo ≥ +0.005):** los pesos de nartaa se entrenaron con
todos los estudios de train (incluido nuestro fold 0) y seleccionaron con gold-58. Pseudo fold 0 y gold quedan contaminados y
no miden generalización (mismo motivo que exp044/exp064). Las métricas internas son solo sanidad; **decide el LB público**.

**Efecto esperado:** LB solo +0.003 a +0.008 sobre la receta ImageNet; riesgo de que 16 épocas a LR normal borren parte de lo
que aporta OAI (por eso exp044 usó 6 épocas a LR 1e-5; aquí se mantiene la receta exp045 a propósito, es la variable de V1).

**Criterios (no cambian después de correr):**
- **Decide el LB público** del kernel solo `exp071` (inferencia = la de los kernels de exp045/exp056: TTA espejo, K_EVAL 24,
  `swa_alldata.pt`), un envío.
  - `worked` si LB(exp071) ≥ 0.946 (= exp045 5 folds 0.941 + 0.005; Δ < 0.004 es ruido del LB público, ~390 estudios).
  - `didnt` si 0.941 ≤ LB(exp071) < 0.946 o si no supera exp056 all-data (0.943) por ≥ 0.003.
  - `killed` si LB(exp071) < 0.941.
- **Sanidad (no decide; contaminada):** gold del SWA all-data ≥ 0.90 y ≥ gold del init nartaa − 0.02 (se imprime en la celda de
  sanidad). Si falla → `killed` por olvido y no se construye kernel.
- **Coste:** GPU-h ≤ 1.5 × 2.81 (= 4.2 GPU-h equivalentes PRO 4500); si no, se para y se escribe el verdict.
- Si `worked`: entra como candidato al blend (grupo raptor) en un experimento/kernel aparte; aquí no se mezcla nada.

**Presupuesto:** smoke ≈ 0.1 h + all-data ≈ 2 h en RTX 5090 (pod `rsna-ext1`, $1.19/h) ≈ $2.5.

**Licencia / legal:** pesos de terceros `nartaa/rsna-knee-publication-swa-weights-20261007` (Kaggle); términos del 13-oct permiten
fine-tune y redistribución no comercial dentro de la competición y licenciar una solución ganadora CC BY-NC 4.0 con atribución
+ aviso OAI. Los pesos incorporan supervisión OAI/MOAKS (OAI: NIMH Data Archive, términos OAI); para el paper se declara como
init de terceros. Rodrigo firmará los acuerdos de datos (OAI/fastMRI/MRNet) que hagan falta para la publicación.

**Fuentes:** nartaa, `nartaa/rsna-knee-publication-swa-weights-20261007` (descripción: 2,399 rodillas OAI baseline, supervisión
MOAKS enmascarada vía 3DReasonKnee, Sambara et al. 2025, arXiv 2510.20967); interno: exp044 (código de carga), exp045 (receta).

**Riesgos:** olvido catastrófico de lo aprendido en OAI con LR 3e-5 × 16 épocas (la sanidad de gold lo detecta solo en parte);
el LB público tiene ruido ±0.004 y un solo envío; datos externos: vigilar reglas §6 (OAI gratuito con registro).

## 2. PLAN

- Celda 0 (parámetros, celda 2 del .ipynb): `INIT_CKPT`, `INIT_SHA256`, `ALL_DATA = True`, `SWA_EPOCHS = [13, 14, 15]`, `NOTES`.
- Celda 3 (modelo, celda 8 del .ipynb): `_init_state()` + `build_model()` con `pretrained=not INIT_CKPT` y carga estricta (verbatim exp044).
- Celda 4 (entrenamiento, celda 10 del .ipynb): sanidad del init sobre gold antes de entrenar (verbatim exp044, solo imprime).
- Celda 6 (metrics, celda 14 del .ipynb): `init_ckpt`, `init_gold_auc`.
- Ninguna otra. Markdown de cabecera actualizado.
- Smoke `SMOKE = True` (all-data, pocos estudios, 1 época) en `rsna-ext1`; luego `SMOKE = False` con `run_notebook.sh`, auto-stop.
- Después (si la sanidad pasa): dataset privado con `swa_alldata.pt` + kernel solo (como los de exp045/exp056) → cola de envíos.

## 3. VERDICT (números de `results/exp071_raptor_nartaa_init_aug/alldata/metrics.json` y LB)

**Resultado:** pendiente del LB · **Fase alcanzada:** all-data entrenado (14-oct 14:43–16:13 UTC, pod rsna-ext1 RTX 5090), sanidad OK

Pesos: `/workspace/users/rodrigo/wt_exp071_raptor_nartaa_init_aug/results/exp071_raptor_nartaa_init_aug/swa_alldata.pt` (SWA épocas 13–15).

| Métrica | Padre (exp045 all-data) | Este | Δ |
|---|---|---|---|
| LB público solo (decide) | 0.941 (5 folds) | pendiente | |
| gold SWA all-data (sanidad, contaminada) | 0.9186 | 0.9173 | −0.0013 |
| gold del init nartaa (sanidad) | — | 0.9191 | |
| GPU-h | 2.81 (RTX PRO 4500) | 1.49 (RTX 5090) | |

Sanidad: gold 0.9173 ≥ 0.90 y ≥ 0.9191 − 0.02 → pasa; se puede construir el kernel solo (TTA espejo, K_EVAL 24) y enviarlo.

**Qué aprendimos (2-4 líneas):** pendiente del LB.

**Siguiente paso sugerido:** dataset privado con `swa_alldata.pt` + kernel solo como los de exp045/exp056 → cola de envíos; aplicar criterio.
