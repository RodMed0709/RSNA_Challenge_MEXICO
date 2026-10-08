# exp022_entropia_suave

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp013_cabeza_hallazgo |
| Fecha | 2026-10-08 |
| Rama | exp022_entropia_suave |
| Estado | spec |

Idea: la penalización de entropía de exp017 con λ diez veces menor (0.005), buscando concentración parcial sin colapso.

Origen: exp017 (λ = 0.05) colapsó: los 12 hallazgos atienden la misma serie coronal (entropía 0.01, coseno 1.00) y la
pseudo cae −0.046. Antes: exp013 subió pseudo +0.011, pero su atención por hallazgo es casi uniforme (entropía normalizada ~0.97, similitud
coseno media entre las 12 distribuciones 0.95). Sin atención concentrada no hay poda de series posible, que es el método
del paper (`docs/10_PLAN_PAPER.md`).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si agrego a la pérdida `λ · H̄`, donde `H̄` es la entropía media (sobre hallazgos y estudios del batch,
solo series presentes) de la atención por hallazgo, con `λ = 0.005`, la atención se concentra (entropía normalizada ≤ 0.6) sin colapsar a una misma serie
sin perder pseudo AUC, porque cada hallazgo tiene 1–2 series informativas.

**Única variable cambiada:**

| Parámetro o celda | Padre (exp013) | Este experimento |
|---|---|---|
| Pérdida | BCE con máscara | BCE con máscara + 0.005 · entropía media de la atención por hallazgo (exp017 usó 0.05) |

Todo lo demás igual que exp013 (cache v1_256, 224 px, 8 épocas, lr, semilla, etiquetas v1, `SELECT_ON = "pseudo"`).

**Efecto esperado:** Δ pseudo fold 0 vs exp013 ∈ [−0.005, +0.005]; entropía normalizada media ≤ 0.6.

**Criterios (no cambian después de correr):**
- Éxito (`worked`): Δ pseudo fold 0 ≥ −0.005 contra exp013 **y** entropía normalizada media ≤ 0.6 (en validación).
  Aquí el objetivo es la propiedad de la atención con AUC no inferior, no subir AUC.
- Kill: Δ pseudo fold 0 ≤ −0.015, o gold < exp013 − 0.02 (fold 0 contra fold 0).
- Si la AUC no cae pero la entropía queda > 0.6: `didnt` (λ insuficiente).
- Colapso (coseno medio entre hallazgos ≥ 0.99 con entropía ≤ 0.6) cuenta como `didnt` aunque la AUC no caiga: no sirve para poda.

**Presupuesto:** screen ~2 GPU-h.

**Riesgos:** colapso de todos los hallazgos a la misma serie (se mira la similitud coseno entre hallazgos y la masa por plano).

## 2. PLAN

- Copia del notebook de exp017. Celda que cambia: 0 (`EXP`, `ENT_LAMBDA = 0.005`, `NOTES`). Ninguna otra.
- Smoke con `SMOKE = True` antes del run real.

## 3. VERDICT (números de `results/exp022_entropia_suave/metrics.json`)

**Resultado:** worked / didnt / killed · **Fase alcanzada:** screen

| Métrica | Padre (exp013) | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | 0.838 | | |
| gold AUC fold 0 | 0.789 | | |
| Entropía normalizada media (val) | ~0.97 | | |
| Similitud coseno media entre hallazgos | 0.95 | | |
| GPU-h | | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
