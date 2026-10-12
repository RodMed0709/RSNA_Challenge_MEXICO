# exp056_raptor_combo

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padres | exp045_raptor_aug_espacial (aug) + exp054_etiquetas_L5 (etiquetas); abuelo exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | exp056_raptor_combo (desde exp045_raptor_aug_espacial, 72de635) |
| Estado | SPEC confirmada · fold 0 en cola (rsna-d2) |

Idea: paso 7, track T6. Apilar sobre exp045 los cambios de las tandas del paso 7 que salgan `worked` y entrenar los
5 folds para tener OOF y un ensamble de 5 folds.

## 1. SPEC (antes de tocar el notebook)

**Regla de inclusión (fijada a las 06:35, antes de ver los verdicts):** entra un componente si su verdict de fold 0 es
`worked`: pseudo L2 fold 0 (mejor época) ≥ exp031 + 0.003 = **0.8866** y gold sin caer > 0.02. exp050 (384, inferencia
~1.45×) solo si Δ ≥ +0.005. Parámetros copiados tal cual de su experimento.

Aplicación (números de cada `metrics.json`, fold 0, mejor época):

| componente | pseudo L2 f0 | Δ vs exp031 (0.8836) | gold | ¿entra? |
|---|---|---|---|---|
| exp045 aug espacial | 0.8899 | +0.0063 | 0.9202 | base |
| exp048 mixup | 0.8845 | +0.0009 | 0.9067 | no |
| exp050 res384 | 0.8856 | +0.0020 | 0.9172 | no (además ×1.46 coste) |
| exp052 etiqueta estocástica | 0.8851 | +0.0015 | 0.9150 | no |
| sub046_1 intensidad suave | 0.8850 | +0.0014 | 0.9131 | no |
| exp054 L5 como 5.ª fuente | 0.8896 | +0.0060 | 0.9190 | **sí** |

exp047, exp049 (+0.0028, al borde) y exp051/exp053 (si salen después) no entran en esta versión.

**Hipótesis:** la aug espacial de exp045 (regularización de entrada) y el consenso de 5 fuentes con L5 de exp054
(etiquetas menos ruidosas) son ortogonales; juntos no empeoran al mejor de los dos padres en fold 0 y dan la mejor
receta propia para el ensamble de 5 folds.

**Única variable (contra exp045):** objetivo de entrenamiento = consenso de 5 fuentes (Raptor, L2, L3, L4, L5) en vez
de 4; es exactamente el cambio de exp054 (celdas 0 y 2) sobre el notebook de exp045. Visto desde exp054, la variable es
la aug espacial de exp045. Nada más cambia.

**Métrica principal:** pseudo L2 fold 0 de la mejor época (sigue binarizando solo L2 > 0.5; con 5 fuentes L2 pesa 1/5
en el objetivo en vez de 1/4, así que el sesgo a favor de la vara baja un poco). SWA top-3 se reporta.

**Criterios (fijos, no cambian después de correr):**
- `worked` (el combo vale): pseudo L2 fold 0 ≥ max(exp045 0.8899, exp054 0.8896) − 0.001 = **≥ 0.8889** y
  gold ≥ 0.9202 − 0.02 = **≥ 0.9002**.
- `killed`: gold < 0.9002, o pseudo L2 fold 0 ≤ 0.8899 − 0.005 = ≤ 0.8849.
- `didnt`: cualquier otro caso.
- Si no es `worked`: se paran los folds 1–4 del combo y el entregable son los 5 folds de exp045 (respaldo).

**Plan de folds (pedido por el orquestador por saldo y tiempo):** los folds 1–4 del combo arrancan en cuanto terminan
los de exp045 (~10:20 UTC), sin esperar al fold 0 (los dos cambios ya están validados por separado); el fold 0 confirma
(~11:40) y si no vale se paran.

**Entregables:** `oof_swa_fK.csv` y `swa_fK.pt` de los 5 folds, pseudo L2 OOF de 5 folds, gold del ensamble de 5 folds.
Lo mismo para exp045 (respaldo).

**Presupuesto:** 5 × ≈ 2.5 GPU-h.

## 2. PLAN

- Notebook `exp056_raptor_combo.ipynb` = copia del de exp045 con los cambios de exp054 copiados literal:
  - celda 0: `EXP`, `PADRE`, `L5_CSV = /workspace/data/labels/ours/L5_20261011.csv`, `NOTES`.
  - celda 2: `SOURCES` añade `"L5"`; diagnósticos de cobertura/acuerdo con 5 fuentes.
  - celda 7: un run de un solo fold k ≠ 0 guarda su `metrics.json` en `results/<EXP>/fold<k>/` (no pisa el de fold 0).
  - Comprobado: `diff` de las celdas de código contra exp054 = solo los cambios de exp045 + lo anterior.
- Smoke `SMOKE=True` en rsna-d2; fold 0 en rsna-d2 (cuando se libere de sub045_2).
- Folds 1–4 (`-y 'FOLDS: [k]'`) en rsna-f, rsna-c2, rsna-i, rsna-h tras los folds de exp045, un fold por pod, todos en
  el worktree del volumen `wt_exp056_raptor_combo` (nombres por fold: `history_fK`, `oof_swa_fK`, `swa_fK.pt`).
- exp045 folds 1–4 (respaldo, notebook de exp045 sin cambios): worktrees `wt_exp045_fK`; F 06:51, C2 07:12, I 07:30, H 07:45.

## 3. VERDICT (números de `results/exp056_raptor_combo/metrics.json`, fase screen, timestamp 2026-10-12T11:41:42Z)

**Resultado: `didnt`** (escrito por el orquestador en la nube, 2026-10-12 ~13:15 UTC).

| | pseudo L2 f0 | v1 | gold (solo veto) |
|---|---|---|---|
| mejor época (ep 15) | **0.8857** | 0.8407 | 0.9198 |
| SWA top-3 (ep 15, 14, 13) | 0.8854 | 0.8407 | 0.9212 |
| exp045 (padre, mejor época) | 0.8899 | | 0.9201 |

- Δ pseudo vs exp045 = −0.0042 (`delta.pseudo_f0` de metrics.json); Δ gold = −0.0003.
- Criterios fijos: `worked` exige ≥ 0.8889 → no; `killed` si ≤ 0.8849 o gold < 0.9002 → no. Queda **`didnt`**.
- Combinar aug espacial (exp045) + L5 (exp054) no suma: cada uno daba ≈ +0.006 sobre exp031 y juntos quedan por debajo
  de cada uno por separado (0.8857 vs 0.8899 / 0.8896).
- Según la SPEC, folds 1–4 del combo parados (pods F, C2, H, I detenidos 11:45 UTC; última época completa: f1 ep13,
  f2 ep11, f3 ep9, f4 ep7). Entregable = los 5 folds de exp045 (OOF 0.8844, gold ens 5 folds 0.9244; `oof5_exp045.json`).
- GPU-h fold 0: 2.51.
