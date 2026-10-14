# exp069_compact_r7_ft

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp068_compact_r7_receta (trainer R7 v9.8 + flags públicos, desde ImageNet) |
| Fecha | 2026-10-14 |
| Rama | exp069_compact_r7_ft (desde exp068_compact_r7_receta) |
| Estado | spec → run encadenado tras exp067 en `rsna-c67` |

Idea: análogo de exp044 (Raptor afinado desde los pesos de nartaa) para la compact: partir de los pesos públicos de R7
(`heliosli/rsna-knee-compact-r7cnxfsss`, `E_lrprior_s0_f0.pt`, sha256 `ca8a76e0…`, licencia "other": uso en la competición)
y afinarlos con nuestras etiquetas de consenso, all-data.

## 1. SPEC

**Hipótesis:** R7 afinado con nuestras etiquetas conserva lo que R7 aprendió de sus etiquetas privadas (opusv11b + estados +
fastMRI+) y añade la señal de nuestro consenso; en el blend Kaggle (R7 público + Raptor + exp069) sube el LB.

**Única variable (inseparable, como en exp044):** init = R7 f0 en lugar de ImageNet, con fine-tune corto:

| Parámetro | Padre (exp068) | Este |
|---|---|---|
| Init | ConvNeXt-S in22k-ft-in1k | R7 `E_lrprior_s0_f0.pt` (carga no estricta: sin cabeza de estados, extent ni aux, que necesitan targets privados) |
| Épocas | 16 | 5 |
| LR enc / cabeza | 1e-4 / 1e-3 | 3e-5 / 3e-4 (×0.3, el `--warm-lr-scale` sugerido por el trainer R7) |
| Datos | folds 1–4 | all-data folds 0–4 (4348), fold ficticio único |

Resto igual a exp068 (flags R7 públicos, consenso exp031 como `lr_prior`, caché `px384_v4`, última época = EMA).
Port = trainer R7 + (1) folds de la columna `fold` + (2) carga del init tras el `bias0` (registra tensores cargados/omitidos).

**Contaminación:** R7 f0 entrenó con 3/4 del pool de heliosli (casi todos nuestros estudios no-gold); no hay validación
limpia. Gold no se entrenó (manifest `gold_in_training 0`) pero puede estar contaminado por selección de heliosli.

**Criterios (fijos):**
- `killed` si gold final < gold del init (R7 f0 sin estados/extent, medido en el notebook) − 0.02 (olvido), o < 0.8772.
- Si no: decide el LB del kernel de blend con y sin exp069 (pesos fijados antes de enviar): `worked` si Δ LB ≥ +0.002.
- Presupuesto ≤ 6 GPU-h (5 ép × ~65 min sin compile sobre 4348; con compile menos).

**Licencia:** código y pesos de heliosli solo en `public_assets` / disco del pod / export privado; nunca a git.

## 2. PLAN

- Notebook = el de exp067 (all-data) con: celda 0 (EXP/PADRE, `SRC_DIR=src_r7`, `R7_INIT`+sha, `R7_FLAGS`, 5 ép, LR ×0.3,
  `EXPORT_DIR`), celda 1 (sha R7 + init), celda 3 (port: + carga del init), celda 4 (flags, `R7_INIT` en el entorno),
  celda 5 (gold del init y del final, veto doble), celda 6 (desvíos/export).
- Encadenado en `rsna-c67` tras exp067 (`orq/compact2/next_rsna-c67.sh`), reutiliza las cachés `/root/px384_v4_all` y
  `/root/px384_v4_gold`: smoke → train → eval → commit local → export `kaggle_export/exp069_compact_r7_ft/` → podStop.

## 3. VERDICT

(pendiente)
