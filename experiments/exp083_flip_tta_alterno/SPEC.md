# exp083_flip_tta_alterno

| Campo | Valor |
|---|---|
| Autor | rodrigo (agente eff3a) |
| Padre | exp080_seqhead_ctrl_mil |
| Fecha | 2026-10-18 |
| Rama | exp083_flip_tta_alterno |
| Estado | spec |

Idea: TTA espejo a presupuesto constante: de las 64 ventanas del kernel, las de posición impar van espejadas (32 + 32), mismo costo GPU.

## 1. SPEC (antes de tocar el notebook)

**Tipo:** evaluación pura (no se entrena nada). Backbone exp074 `swa_f0.pt` congelado + cabeza MIL de exp080
`swa_mil_f0.pt` SIN reentrenar, fold 0 de `split_v1_seed42` (870 estudios) y gold (58, solo veto).

**Convención del espejo (verificada en el código de entrenamiento):** exp045 entrenó con espejo L/R anatómico p 0.5 por
estudio (`spatial_aug`): ventana cuyo corte central cae en slots sagitales (0, 1 = cortes 0–47) → invertir el orden de los
3 cortes-canal (`x.flip(2)`); si no (coronal) → volteo del ancho (`x.flip(-1)`); **etiquetas sin permutar** (espejo
anatómico: medial/lateral no cambian). exp057 y exp074 heredan esa aug (exp057 SPEC: "todo lo demás idéntico al padre:
aug espacial de exp045"); exp080 reentrenó solo la cabeza sobre embeddings sin aug. Aquí se aplica exactamente esa
convención, sobre el recorte central 320 antes del reescalado a 224, y el promedio/bolsa va **sin permutar**.

**Hipótesis:** si las 32 ventanas de posición impar (de las 64 ordenadas por corte que usa el kernel exp080-ctrl224-fast)
entran espejadas, la cabeza MIL ve una bolsa con las dos "lateralidades" y la pseudo AUC fold 0 sube ≥ +0.002 sin costo
GPU, porque el backbone se entrenó con espejo (invariancia aprendida) y el promedio de vistas reduce varianza.
Evidencia previa en contra: exp041-mirror224 (64+64, doble costo) y r7-raptor-exp067flip salieron `didnt` en LB.

**Única variable cambiada:**

| Parámetro o celda | Padre (exp080, kernel) | Este experimento |
|---|---|---|
| Ventanas en inferencia | 64 centros (linspace sobre centros válidos de slots 0–2), todas sin espejo | los mismos 64 centros; posiciones impares (1, 3, …, 63) espejadas con la convención de exp045; una sola bolsa de 64 a la cabeza |

Todo lo demás igual: backbone y cabeza de fold 0 (sin reentrenar), recorte 320 → 224 bilineal, bf16 autocast como la
extracción de embeddings de exp080, split, gold.

**Control (mismo pipeline, misma corrida):** 64 ventanas sin espejo = exp080 (debe reproducir 0.9018 / 0.9252 ± 0.0005;
si no, se para y se avisa).

**Informativos (no deciden):** (i) 64 todas espejadas; (ii) fase par espejada (complemento); (iii) 64 + 64 media en
probabilidad (doble costo GPU, ~+185 s en el oculto); (iv) bolsa única de 128; (v) 32 + 32 en dos bolsas (subconjunto par,
media en probabilidad); (vi) 32 sin espejo (referencia de mitad de costo); (vii) IC bootstrap pareado 1000× del Δ pseudo.

**Criterios (fijos, no cambian después de correr):** Δ = variante decidida − control del mismo pipeline, fold 0.
- `worked` si Δ pseudo f0 ≥ **+0.002** y gold no cae > 0.02 → se construye kernel privado (LB criterio ≥ **0.950**,
  padre 0.949).
- `killed` si gold de la variante < gold del control − 0.02.
- Si no, `didnt`.
- 64 + 64 (informativo) solo se consideraría para un kernel si Δ pseudo ≥ **+0.004** (cuesta ~+185 s ≈ 0.0026 de AUC
  en la métrica de eficiencia); aun así no es la variable de este experimento.
- No hay fase 5 folds: exp074 solo tiene backbone de fold 0 y all-data (declarado).

**Efecto esperado:** Δ pseudo +0.000 a +0.003; costo GPU × 1.0.

**Presupuesto:** ~0.2 GPU-h (embeddings de 928 estudios × 128 ventanas), 1 pod RTX PRO 4500 (~$0.72/h), tope $1.

**Riesgos:** la cabeza de exp080 se entrenó sobre embeddings sin espejo (posible desajuste de distribución en la bolsa
mixta); la convención sagital (invertir 3 canales) es un espejo parcial (el orden de las ventanas no se invierte, la
cabeza MIL es invariante al orden).

## 2. PLAN

- Notebook `exp083_flip_tta_alterno.ipynb` copiado de exp080. Celdas que cambian: 0 (parámetros: `HEAD_CKPT`,
  `MIRROR_PHASE`, quitar los de entrenamiento), 3 (modelo: `embed_windows` calcula a la vez embeddings sin espejo y
  espejados), 4 ("entrenamiento" = solo extracción de embeddings, no se entrena), 5 (evaluación de variantes), 6 (métricas).
- Smoke con `SMOKE = True` (48 estudios de val + gold) antes del run real; luego fold 0 completo. Cadena nohup en el pod.

## 3. VERDICT (números de `results/exp083_flip_tta_alterno/metrics.json`)

**Resultado:** pendiente
