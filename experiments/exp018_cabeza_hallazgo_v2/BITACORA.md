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
