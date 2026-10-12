# exp056_raptor_combo

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial (abuelo: exp031_raptor_etiquetas_consenso) |
| Fecha | 2026-10-12 |
| Rama | exp056_raptor_combo (desde exp045_raptor_aug_espacial, 72de635) |
| Estado | combo = exp045 (ningún otro componente worked) · folds 1–4 corriendo |

Idea: paso 7, track T6. Apilar sobre exp045 (aug espacial, único worked hasta 06:30 UTC) los demás cambios de la
primera tanda del paso 7 que salgan `worked`, y entrenar los 5 folds para tener OOF y un ensamble de 5 folds.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** los cambios que por separado mejoran a exp031 en ≥ +0.003 de pseudo L2 son en buena parte
complementarios (regularización espacial, mezcla de estudios, ruido de etiqueta, intensidad suave, resolución); juntos
sobre exp045 no empeoran a exp045 en fold 0 y dan la mejor receta propia para el ensamble de 5 folds.

**Única variable (contra exp045):** el bloque "componentes worked de la primera tanda" (tal como pide el plan del
paso 7: el combo es una variable; si choca, el leave-one-out sub056_k aísla el componente).

**Regla de inclusión (fijada antes de ver los verdicts):** entra un componente de {exp048 mixup, exp050 res384,
exp052 etiqueta estocástica, sub046_1 intensidad suave} si su verdict de fold 0 es `worked`, es decir pseudo L2 fold 0
(mejor época) ≥ exp031 + 0.003 = **0.8866** y gold sin caer > 0.02. Excepción exp050 (384 sube la inferencia ~1.45×):
solo entra si su Δ es claramente mayor, **≥ +0.005 (pseudo ≥ 0.8886)**; si entra se anota el coste de inferencia.
exp049 CLAHE (+0.0028, `didnt` al borde) no entra; queda como candidato a sub056_k de adición.
Los parámetros de cada componente se copian tal cual de su experimento (sin re-afinar).

Si ningún componente más entra, el combo = exp045 tal cual: no se repite el fold 0 (se usa el de exp045) y se va
directo a los folds 1–4.

**Métrica principal:** pseudo L2 fold 0 de la mejor época (como el resto del paso 7); SWA top-3 se reporta.

**Criterios (fijos, no cambian después de correr):**
- `worked` (el combo vale): pseudo L2 fold 0 ≥ exp045 − 0.001 = **≥ 0.8889** y gold ≥ 0.9202 − 0.02 = **≥ 0.9002**.
- `killed`: gold < 0.9002, o pseudo L2 fold 0 ≤ exp045 − 0.005 = ≤ 0.8849.
- `didnt`: cualquier otro caso → leave-one-out `sub056_k` (quitar un componente cada uno) para hallar el que choca.
(Padre exp045, `results/exp045_raptor_aug_espacial/metrics.json`: pseudo L2 0.8899 (ep 15), SWA 0.8895, gold 0.9202.)

**Si vale:** folds 1–4 en paralelo (un fold por pod, `-y 'FOLDS: [k]'`), OOF de 5 folds (`oof_swa_fK.csv`), pseudo L2
OOF y ensamble de los 5 `swa_fK.pt` para Kaggle.

**Presupuesto:** fold 0 ≈ 2.5 GPU-h (×1.45 si entra 384); folds 1–4 ≈ 4 × 2.5 GPU-h.

**Riesgos:** las augs pueden sumar demasiada regularización para 16 épocas (exp045 seguía subiendo en ep 15); mixup
con etiquetas de ruido estocástico duplica el ruido del objetivo.

## 2. PLAN

- Notebook `exp056_raptor_combo.ipynb` = copia del de exp045; cambios:
  - celda 0: `EXP`, `PADRE`, interruptores `USE_MIXUP` (+ `MIXUP_P` 0.5, `MIXUP_ALPHA` 0.4 de exp048),
    `LABEL_MODE` ("consensus" | "stochastic", exp052), `USE_INT_AUG` (+ parámetros suaves de sub046_1), `RES`.
  - celda 3: `intensity_aug` de exp046 (copia literal); `to_input` aplica /255 → espejo+afín → ganancia → intensidad
    (si `USE_INT_AUG`) → normalización. Solo con `train=True`.
  - celda 4: objetivo estocástico de exp052 (si `LABEL_MODE == "stochastic"`), mixup de exp048 sobre la entrada ya
    aumentada (si `USE_MIXUP`); el `metrics.json` de un run de fold k ≠ 0 va a `results/<EXP>/fold<k>/`.
  - Con todos los interruptores apagados y `RES = 320` el notebook es exp045 (mismo consumo de RNG).
- Valores de los interruptores, aplicando la regla de inclusión a los verdicts de fold 0 (pseudo L2 mejor época,
  de cada `metrics.json`; umbral 0.8866, 0.8886 para exp050):

  | componente | pseudo L2 f0 | Δ vs exp031 | gold | ¿entra? |
  |---|---|---|---|---|
  | exp048 mixup | 0.8845 | +0.0009 | 0.9067 | no |
  | exp050 res384 | 0.8856 | +0.0020 | 0.9172 | no (además ×1.46 coste) |
  | exp052 etiqueta estocástica | 0.8851 | +0.0015 | 0.9150 | no |
  | sub046_1 intensidad suave | 0.8850 | +0.0014 | 0.9131 | no |

  Ninguno entra → **combo = exp045 tal cual** (interruptores apagados, RES 320). Según la SPEC no se repite el fold 0
  (se usa el de exp045: pseudo L2 0.8899, SWA 0.8895, gold 0.9202) y se va directo a los folds 1–4.
  Los folds 1–4 se corren con el notebook de exp045 sin cambios (rama exp045, commit 5b401f2: el notebook es idéntico
  al de 72de635), un worktree del volumen por fold (`wt_exp045_fK`) para no pisar resultados.
- Pods: fold 1 rsna-f (06:51), fold 2 rsna-c2 (07:12), fold 3 rsna-i (07:30), fold 4 rsna-h (tras el all-data de exp045).

## 3. VERDICT

Pendiente.
