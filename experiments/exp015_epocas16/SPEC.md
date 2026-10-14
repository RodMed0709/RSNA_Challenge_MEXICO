# exp015_epocas16

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp002_dinov2s_ref |
| Fecha | 2026-10-08 |
| Rama | exp015_epocas16 |
| Estado | running |

Idea: receta exp002 con 16 épocas en lugar de 8.

Origen: `docs/pasos/analisis_eficiencia_paso1.md` y `docs/EXPERIMENTOS.md`: en exp002 el mejor checkpoint es la última
época en 4/5 folds, y en exp003–exp008 también. La receta está subentrenada; eso contamina cualquier comparación.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si cambio `EPOCHS` de 8 a 16, la pseudo AUC sube, porque la curva de validación de exp002 todavía sube en
la época 8. El scheduler coseno se estira a 16 épocas (misma forma, misma `WARMUP_RATIO`).

**Única variable cambiada:**

| Parámetro o celda | Padre | Este experimento |
|---|---|---|
| `EPOCHS` | 8 | 16 |

Todo lo demás igual: split, semilla, lr, `FREEZE_EPOCHS = 1`, etiquetas v1, cache v1_256, cabeza, `SELECT_ON = "pseudo"`.

**Efecto esperado:** Δ pseudo fold 0 ≈ +0.005 a +0.015; costo × 2.

**Criterios (no cambian después de correr):**
- Éxito → 5 folds: Δ pseudo fold 0 ≥ +0.005.
- Kill: Δ pseudo fold 0 ≤ −0.005, o gold < padre − 0.02, o GPU-h > 1.5 × presupuesto.
- Neutral: `didnt`. Se reporta también la época del mejor checkpoint: si sigue siendo la última, la receta aún está corta.

**Presupuesto:** screen ~3 GPU-h (RTX PRO 4500, compartida con otro run: el tiempo de reloj no es comparable).

**Riesgos:** sobreajuste a etiquetas ruidosas v1; se ve en `history_f0.csv` (pseudo baja tras el pico).

## 2. PLAN

- Copia del notebook de exp002. Celda que cambia: 0 (`EPOCHS`, y `PADRE`). Ninguna otra.
- Smoke con `SMOKE = True` antes del run real.

## 3. VERDICT (números de `results/exp015_epocas16/metrics.json`)

**Resultado:** promising en screen (5 folds pendientes) · **Fase alcanzada:** screen

| Métrica | Padre | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | 0.827 | 0.847 | +0.020 |
| gold AUC (fold 0 contra fold 0 del padre) | 0.777 | 0.797 | +0.020 |
| Época del mejor checkpoint | 8 de 8 | 12 de 16 | |
| GPU-h | | 3.34 (200 min de reloj, GPU compartida con exp012) | 1.11× presupuesto |

**Qué aprendimos (2-4 líneas):** la receta de exp002 estaba subentrenada. Con 16 épocas la pseudo sube +0.020 y gold no cae
(+0.020). La curva sube hasta la época 8 (0.844) y luego queda plana (0.845–0.847 hasta la 16): la ganancia viene sobre todo
de la cola del coseno con lr bajo, no de más pasos. Mejor checkpoint en la época 12 (ya no la última): la receta dejó de estar corta.

**Siguiente paso sugerido:** confirmar con 5 folds dentro de la nueva referencia combinada (entrada v2 + cabeza por hallazgo
+ 16 épocas, o 12 si el costo aprieta), no como run aislado: 5 folds × 3.3 GPU-h no caben en esta sesión.
