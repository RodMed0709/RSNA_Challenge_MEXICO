# Bitácora — exp018_cabeza_hallazgo_v2

Padre: `exp012_entrada_v2` · Rama: `exp018_cabeza_hallazgo_v2` · Fase SDD actual: **IMPLEMENT, smoke**.

## 1. Qué es este experimento

La receta y la entrada de exp012 (cache v2, 336 px, 6 slots) con la cabeza por hallazgo de exp013: 12 queries, una por
hallazgo, que atienden sobre los 6 slots. La atención se guarda por slot para la poda de series por hallazgo.

## 2. Registro cronológico

### 2026-10-08 — SPEC y notebook
- **Qué:** SPEC (secciones 1 y 2). Notebook copiado de exp012; cambian solo las celdas 0, 3 y 5 del PLAN.
  - Celda 3: `LabelQueryPool` y `KneeV2LabelQueryModel` (hereda de `KneeV2Model`; `series_mask` True = slot faltante se
    pasa tal cual como `key_padding_mask`).
  - Celda 5: atención de exp013 adaptada a slots: `attn_f{fold}.npz` (con `slot`, `slot_name`, `plane`) y en
    `metrics.json` `slot_mass`, `slot_mass_present`, `slot_coverage`, `plane_mass`, `entropy_norm`, `mean_pairwise_cosine`.
    El `metrics.json` del padre se busca en `results/` del clon y en `/workspace/users/<autor>/results`.
- **Verificación local (CPU, backbone falso):** logits (2, 12), atención (2, 12, 6), atención 0 en slots enmascarados,
  cambiar un slot enmascarado no cambia la salida, `slot_mass` suma 1 por hallazgo.

### 2026-10-08 — Smoke en el pod A
- **Qué:** `run_notebook.sh ... -p SMOKE True` en `/workspace/users/rodrigo/wt_exp018` (GPU compartida con 2 runs).
- **Resultado:** `fold 0 epoch 1: loss 0.6063 pseudo 0.5513 gold 0.5378 (1.7 min)` · decisión `sin_padre` (exp012 aún sin
  `metrics.json`) · `Guardado: .../metrics.json` (renombrado a `metrics_smoke.json`). `attn_f0.npz` con 5,544 filas y
  `slot_mass` por hallazgo en `per_fold.0.attention`. Atención uniforme tras 1 época (coseno 1.000), esperado en smoke.
- En el clon del pod hay un symlink `results/exp012_entrada_v2` → resultados de exp012 (no se commitea).
