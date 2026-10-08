# exp017_atencion_dispersa

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp013_cabeza_hallazgo |
| Fecha | 2026-10-08 |
| Rama | exp017_atencion_dispersa |
| Estado | spec |

Idea: penalización de entropía sobre la atención por hallazgo para que cada hallazgo elija pocas series.

Origen: exp013 subió pseudo +0.011, pero su atención por hallazgo es casi uniforme (entropía normalizada ~0.97, similitud
coseno media entre las 12 distribuciones 0.95). Sin atención concentrada no hay poda de series posible, que es el método
del paper (`docs/10_PLAN_PAPER.md`).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si agrego a la pérdida `λ · H̄`, donde `H̄` es la entropía media (sobre hallazgos y estudios del batch,
solo series presentes) de la atención por hallazgo, con `λ = 0.05`, la atención se concentra (entropía normalizada ≤ 0.6)
sin perder pseudo AUC, porque cada hallazgo tiene 1–2 series informativas.

**Única variable cambiada:**

| Parámetro o celda | Padre (exp013) | Este experimento |
|---|---|---|
| Pérdida | BCE con máscara | BCE con máscara + 0.05 · entropía media de la atención por hallazgo |

Todo lo demás igual que exp013 (cache v1_256, 224 px, 8 épocas, lr, semilla, etiquetas v1, `SELECT_ON = "pseudo"`).

**Efecto esperado:** Δ pseudo fold 0 vs exp013 ∈ [−0.005, +0.005]; entropía normalizada media ≤ 0.6.

**Criterios (no cambian después de correr):**
- Éxito (`worked`): Δ pseudo fold 0 ≥ −0.005 contra exp013 **y** entropía normalizada media ≤ 0.6 (en validación).
  Aquí el objetivo es la propiedad de la atención con AUC no inferior, no subir AUC.
- Kill: Δ pseudo fold 0 ≤ −0.015, o gold < exp013 − 0.02 (fold 0 contra fold 0).
- Si la AUC no cae pero la entropía queda > 0.6: `didnt` (λ insuficiente; otro experimento con otro λ).

**Presupuesto:** screen ~2 GPU-h.

**Riesgos:** colapso de todos los hallazgos a la misma serie (se mira la similitud coseno entre hallazgos y la masa por plano).

## 2. PLAN

- Copia del notebook de exp013. Celdas que cambian: 0 (`ENT_LAMBDA = 0.05`) y 4 (entrenamiento: término de entropía en la
  pérdida). Ninguna otra.
- Smoke con `SMOKE = True` antes del run real.

## 3. VERDICT (números de `results/exp017_atencion_dispersa/metrics.json`)

**Resultado:** worked / didnt / killed · **Fase alcanzada:** screen

| Métrica | Padre (exp013) | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | 0.838 | | |
| gold AUC fold 0 | 0.789 | | |
| Entropía normalizada media (val) | ~0.97 | | |
| Similitud coseno media entre hallazgos | 0.95 | | |
| GPU-h | | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:** experimento de poda (inferencia sin reentrenar, top-k series por hallazgo).
