# sub046_1_aug_suave

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp046_raptor_aug_intensidad (referencia de decisión: exp031_raptor_etiquetas_consenso) |
| Fecha | 2026-10-12 |
| Rama | sub046_1_aug_suave |
| Estado | spec |

Idea: exp046 (aug de intensidad) quedó en pseudo L2 0.8854 (Δ +0.0018 vs exp031, < +0.003) con gold 0.9039 (−0.0105):
cerca pero por debajo del umbral. Su curva fue más baja que la del padre en las épocas medias (p. ej. ep8 0.8709 vs
0.8724, ep10 0.8787 vs 0.8813): la aug fuerte frena el aprendizaje en 16 épocas. Se prueba la versión suave.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** con la magnitud del bloque de intensidad a la mitad la pseudo L2 fold 0 llega a ≥ exp031 + 0.003, porque
conserva parte de la robustez a intensidad sin frenar la convergencia en 16 épocas.

**Única variable:** magnitud del bloque de intensidad de exp046 (mismas probabilidades, mismo orden, ganancia igual).

| Parámetro | exp046 | sub046_1 |
|---|---|---|
| `AUG_GAMMA` | U(0.7, 1.5) | U(0.85, 1.2) |
| `AUG_CONTRAST` | U(0.8, 1.2) | U(0.9, 1.1) |
| `AUG_BIAS_AMP` (p 0.3) | 0.15 | 0.075 |
| `AUG_BLUR_SIGMA` (p 0.25) | U(0.5, 1.0) | U(0.5, 0.75) |
| `AUG_NOISE_SD` (p 0.5) | 0.05 | 0.025 |

Todo lo demás idéntico a exp046 (= exp031 + aug de intensidad).

**Métrica principal:** pseudo L2 fold 0 de la mejor época; SWA top-3 se reporta.

**Criterios (fijos, contra exp031, que es la referencia del paso 7):**
- `worked`: pseudo L2 fold 0 ≥ **0.8866** y gold ≥ 0.8944.
- `killed`: pseudo L2 fold 0 ≤ 0.8786 o gold < 0.8944.
- `didnt`: cualquier otro caso.
(exp031: pseudo L2 0.8836, gold 0.9144. exp046: pseudo L2 0.8854, gold 0.9039.)
La `decision_auto` del notebook compara contra exp046 (PADRE); el verdict se aplica con estos criterios.

**Presupuesto:** ≈ 2.5 GPU-h fold 0 en rsna-i.

## 2. PLAN

- Notebook `sub046_1_aug_suave.ipynb` = copia del de exp046; cambia solo la celda 0 (`EXP`, `PADRE`, los 5 parámetros
  de magnitud, `NOTES`).
- Sin smoke nuevo (el código es el de exp046, ya probado en smoke y run real); run real `FOLDS=[0]` en rsna-i.

## 3. VERDICT (números de `results/sub046_1_aug_suave/metrics.json`)

**Resultado:** pendiente · **Fase alcanzada:** spec

| Métrica | exp031 | exp046 | Este | Δ vs exp031 |
|---|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8836 | 0.8854 | | |
| pseudo L2 fold 0 (SWA top-3) | 0.8835 | 0.8858 | | |
| v1 fold 0 | 0.8368 | 0.8365 | | |
| gold (veto) | 0.9144 | 0.9039 | | |
| GPU-h | 2.49 | 2.51 | | |

**Qué aprendimos:**

**Siguiente paso sugerido:**
