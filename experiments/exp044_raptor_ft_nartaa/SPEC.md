# exp044_raptor_ft_nartaa

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso (fold 0: pseudo L2 0.8836, gold 0.9144; SWA [15,13,14] 0.8835 / 0.9151; tiene `swa_alldata.pt`) |
| Fecha | 2026-10-12 |
| Rama | exp044_raptor_ft_nartaa |
| Estado | run (all-data) |

Idea: partir de los pesos SWA públicos de nartaa (0.949 LB) en lugar de ImageNet y afinarlos con nuestras etiquetas de
consenso de exp031, en modo all-data, para enviar a Kaggle.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** inicializar el modelo de exp031 con los pesos SWA de nartaa (`nartaa/rsna-knee-publication-swa-weights-20261007`,
`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`, sha256 `7e5315da…`) y afinarlo 6 épocas con las etiquetas de
consenso da un **LB mayor** que el modelo all-data de exp031 entrenado desde ImageNet, porque arranca de un modelo que ya
llega a 0.949 (con TTA) y que vio supervisión que nosotros no tenemos (etiquetas propias de nartaa + OAI).

**Única variable cambiada (con su fine-tune fijado aquí, antes de correr):**

| Parámetro | Padre (exp031) | Este experimento |
|---|---|---|
| Inicialización | ImageNet (timm) | pesos SWA de nartaa (carga estricta) |
| Épocas | 16 | 6 |
| LR backbone / cabeza | 3e-5 / 1e-3 | 1e-5 / 1e-4 |
| SWA | top-3 por pseudo L2 en fold 0 (all-data: épocas fijas) | épocas fijas [3, 4, 5] (las 3 últimas) |

El fine-tune corto con LR bajos es inseparable de la inicialización (16 épocas a 1e-3 en la cabeza borrarían lo
aprendido), así que se trata como una sola variable: "afinar nartaa" frente a "entrenar desde ImageNet". Todo lo demás
igual al padre: etiquetas de consenso (Raptor + L2 + L3 + L4, media por celda), corpus d96/384, recorte 320,
`--norm imagenet`, bs 8 × k=12, AdamW wd 0.02, OneCycle pct 0.15, BCE blanda con pos_weight, bf16, clip 3.0, semilla 42.

**SWA:** la orden pedía "3 mejores por pseudo L2", pero en all-data no hay fold de validación (y la pseudo de cualquier
fold está contaminada, ver abajo), así que se fija de antemano en las 3 últimas épocas [3, 4, 5] (0-index). Se guarda
también `last_alldata.pt`.

**Compatibilidad verificada:** misma arquitectura que nuestro notebook (CoAtNet-2 rmlp 384 con `img_size=320` +
`RaptorClassifier` con atención MIL por hallazgo, mismas claves `backbone.* / norm / att / clsW / clsb`), mismo orden
de los 12 hallazgos (se comprueba contra `ck["lab"]`), mismo corpus y slots (`build_study` de nartaa; sus pesos dan
gold 0.9219 sobre nuestro corpus, BITACORA de exp029). La carga es `strict=True` y se verifica el sha256 del manifest.
Antes de entrenar el notebook evalúa el modelo inicial en gold como sanidad de la carga.

**Contaminación (explícita):**
- nartaa entrenó con **todos** los estudios no-gold ("alldata"): su modelo ya vio fold 0 (y todos los folds). Cualquier
  pseudo AUC sobre nuestros folds queda **inflada y no es comparable** con exp031. Por eso no se corre fold 0.
- Gold: los 58 gold están en el `train.csv` de Kaggle (son los únicos con etiquetas estructuradas). Según el README de
  nartaa, gold-58 **se reutilizó para seleccionar** (épocas del SWA, variantes); no consta que entrenara con ellos (el
  script Raptor los excluye del entrenamiento), pero gold está contaminado por selección. Gold solo sirve de sanidad.
- Pesos con supervisión externa OAI (2399 rodillas, NDA). La licencia de los pesos permite usarlos en la competición; hay
  que revisar las reglas de datos externos de la competición antes de usar este modelo en un envío final.

**Criterios (no cambian después de correr):**
- Decide el **LB público de Kaggle** (mismo kernel de inferencia que el envío all-data de exp031, sin TTA, k 94 o el que use
  ese envío): `worked` si LB(exp044) ≥ LB(exp031 all-data) + 0.005; `didnt` si no.
- Sanidad (no decide): el gold del modelo final no debe caer > 0.02 frente al gold del modelo inicial de nartaa medido en
  el notebook; si cae, se sospecha de olvido catastrófico y se reporta.
- Presupuesto: ≤ 1.5 GPU-h (6 épocas × ~11 min sobre 4349 estudios ≈ 1.1 h).

**Riesgos:** olvido de lo aprendido por nartaa si las LR son altas; el LB público es ruidoso (±0.005); las etiquetas de
consenso pueden ser peores que las de nartaa (su "blendjev") y arrastrar el modelo hacia abajo.

## 2. PLAN

- Celda 0: `ALL_DATA=True`, `SWA_EPOCHS=[3,4,5]`, `EPOCHS=6`, `BB_LR=1e-5`, `HEAD_LR=1e-4`, `INIT_CKPT`, `INIT_SHA256`.
- Celda 3: `build_model` carga los pesos de nartaa (sha256, arch, orden de hallazgos, `strict=True`).
- Celda 4: sanidad (gold del modelo inicial) antes del bucle; el bucle no cambia.
- Celda 6: `init_ckpt` e `init_gold_auc` en `metrics.json` (va a `results/<EXP>/alldata/metrics.json`).
- Smoke `SMOKE=True` (64 estudios, 1 época, all-data), luego run real en rsna-c2.
- Después (otro paso): kernel de inferencia con `swa_alldata.pt` y envío.

## 3. VERDICT

**Resultado:** pendiente (LB) · **Fase alcanzada:** all-data en curso
