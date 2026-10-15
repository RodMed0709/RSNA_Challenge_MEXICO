# exp067_compact_alldata

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp066_compact_f0 (worked: blend pseudo 0.8955, +0.0060 vs exp045; compact sola pseudo 0.8755, gold 0.8972) |
| Fecha | 2026-10-14 |
| Rama | exp067_compact_alldata (desde exp066_compact_f0) |
| Estado | run (all-data) |

Idea: la compact de exp066 funcionó en el blend interno; para usarla en Kaggle hace falta un modelo entrenado con todos
los estudios no-gold (como hicimos con exp030/exp031/exp044/exp045 all-data). Es la pieza "propia" que aporta diversidad al
blend R7 + Raptor (R7 usa otras etiquetas: opusv11b + estados).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** entrenar la receta exacta de exp066 con los 4348 estudios de folds 0–4 da un modelo que, añadido al blend
público R7 + Raptor, sube el LB (o al menos no lo baja) por diversidad de etiquetas.

**Única variable:** datos de entrenamiento = folds 0–4 (4348) en lugar de folds 1–4 (3477). Todo lo demás igual a exp066:
port de `compact_train.py` (sha 14767c7a…, un cambio de folds), brazo `E_lrprior`, ConvNeXt-S in22k-ft-in1k, 16 épocas,
bs 2, 176 ventanas, crop 352/384, lr 1e-4 / head 1e-3, wd 0.05, phys 3.5, spatial attn, eval stride 1, modelo de la
última época, targets = consenso exp031 como `lr_prior`, sin estados ni extent, caché `px384_v4` con `compact_infer.CStudies`.
Implementación (no cambia la matemática): `--compile` (torch.compile del encoder; flag del propio trainer, "same maths") para
ganar ~1.4× de velocidad. Si el smoke falla con compile, se corre sin compile (se anota).

**Contaminación:** no hay fold de validación limpio (todo está en train). Gold (58) sigue fuera del entrenamiento.

**Criterios (fijos):**
- `killed` si gold del modelo all-data < 0.8972 − 0.02 = **0.8772** (veto contra la compact de exp066 f0).
- Si no hay veto, decide el **LB público** del kernel de blend (R7 + Raptor + exp067; pesos fijados en la SPEC del envío
  antes de enviar): `worked` si LB ≥ LB(blend R7 + Raptor sin exp067) + 0.002; `didnt` si no.
- Presupuesto: ≤ 20 GPU-h (RTX PRO 4500; medido exp066: 13.74 h para 3477 estudios → ≈ 17 h sin compile, ≈ 12–13 h con).

**Entregable:** `results/exp067_compact_alldata/E_lrprior_s0_alldata.pt` y export para dataset Kaggle en
`/workspace/users/rodrigo/kaggle_export/exp067_compact_alldata/` (`E_lrprior_s0_f0.pt`, `manifest.json` estilo heliosli sin
`state_spec`, `side_cls.pt`, `metrics.json`). Pesos nunca a git.

**Licencia:** código heliosli solo en `public_assets` / disco del pod; nunca a git (igual que exp066).

## 2. PLAN

- Notebook `exp067_compact_alldata.ipynb` = el de exp066 con cambios solo en: celda 0 (EXP, PADRE, cachés `/root/px384_v4_all`
  y `/root/px384_v4_gold`, `RUN_DIR`, `REF_GOLD_COMPACT`, `EXPORT_DIR`, `COMPILE`), celda 2 (train = folds ≥ 0; fold ficticio
  único para el port), celda 4 (salida `alldata`), celda 5 (predicción solo gold + veto), celda 6 (phase `alldata` + export).
- Pod `rsna-c67` (RTX PRO 4500, 450 GB): caché all-data (4349, 40 procs) → caché gold → smoke (1 época, 48 estudios) →
  train 16 ép → eval gold → commit local de `metrics.json` → export → siguiente trabajo de la cadena o `podStop`.
  Worker `orq/compact2/worker_c67.sh`, log en `orq/compact2/status.log`, watchdog de horas.

## 3. VERDICT

(pendiente: decide el LB público del envío de abajo)

**Veto gold (metrics.json):** gold all-data = 0.9258 ≥ 0.8772 → pasa, no `killed`.

### SPEC del envío Kaggle (fijada antes de correr el kernel, 2026-10-15 ~05:00 UTC)

- Kernel `medellinrodrigo/r7-raptor-exp067` = copia de `r7-raptor-repro` (R7 + Raptor nartaa384, LB 0.956) con un solo cambio
  en el slot compact. Dataset privado `medellinrodrigo/rsna-knee-exp067-compact` (los 4 archivos del export).
- **Slot Raptor:** idéntico al repro (`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`, nartaa 384).
- **Grupo compact:** rank mean por hallazgo de R7 (peso **0.75**) y exp067 all-data (peso **0.25**). exp067 corre con el mismo
  worker `compact_infer.py` / `probe_dual.py` y el mismo preprocesado que R7 (views v4, crop 352, phys 3.5, side model),
  `compact_train.py` de R7 (carga strict, 0 claves faltantes/inesperadas, verificado).
- **Final:** 0.5 compact + 0.5 Raptor, rank mean como el repro (incluido el override OA = 0.5).
- Pesos fijos; no se ajustan con gold ni con LB. Una sola variante.
- Criterio (sección 1): `worked` si LB ≥ LB(R7 + Raptor repro = 0.956) + 0.002 = **0.958**; `didnt` si no.
