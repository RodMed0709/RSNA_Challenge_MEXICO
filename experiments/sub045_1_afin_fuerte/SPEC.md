# sub045_1_afin_fuerte

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial |
| Fecha | 2026-10-12 |
| Rama | sub045_1_afin_fuerte |
| Estado | spec |

Idea: ronda 2 sobre exp045 (`worked`, pseudo L2 0.8899). La aug espacial ayudó y la curva seguía subiendo en la época 15:
probar si un afín más fuerte regulariza más.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** un afín más fuerte (rot ±20°, escala 0.8–1.2, traslación ±10 %) sube la pseudo L2 fold 0 en ≥ +0.003
contra exp045, porque amplía la invariancia a posición/tamaño de la rodilla dentro del recorte de 140 mm.

**Única variable:** magnitud del afín.

| Parámetro | exp045 | sub045_1 |
|---|---|---|
| `AUG_ROT_DEG` | 10 | 20 |
| `AUG_SCALE` | U(0.9, 1.1) | U(0.8, 1.2) |
| `AUG_TRANS` | ±6 % | ±10 % |

Espejo L/R (p 0.5, eje del TTA nartaa) y todo lo demás idéntico a exp045.

**Métrica principal:** pseudo L2 fold 0 de la mejor época; SWA top-3 se reporta.

**Criterios (fijos, contra exp045):**
- `worked`: pseudo L2 fold 0 ≥ 0.8899 + 0.003 = **≥ 0.8929** y gold ≥ 0.9202 − 0.02 = **≥ 0.9002**.
- `killed`: pseudo L2 fold 0 ≤ 0.8899 − 0.005 = **≤ 0.8849** o gold < 0.9002.
- `didnt`: cualquier otro caso.
(exp045, `results/exp045_raptor_aug_espacial/metrics.json`: pseudo L2 0.8899 (ep 15), SWA 0.8895, gold 0.9202. Ruido de semilla 0.0013.)
Si promete: avisar al orquestador antes de lanzar all-data (saldo justo).

**Presupuesto:** ≈ 2.5 GPU-h fold 0 en rsna-e2.

**Riesgos:** rotación de 20° y zoom 0.8 sacan anatomía del recorte de 320 (relleno 0); puede frenar la convergencia.

## 2. PLAN

- Notebook `sub045_1_afin_fuerte.ipynb` = copia del de exp045; cambia solo la celda 0 (`EXP`, `PADRE`, 3 parámetros del afín, `NOTES`).
- Sin smoke nuevo (mismo código que exp045, probado en smoke, test unitario y run real); run real `FOLDS=[0]` en rsna-e2.

## 3. VERDICT (números de `results/sub045_1_afin_fuerte/metrics.json`)

**Resultado:** pendiente · **Fase alcanzada:** spec

| Métrica | exp045 | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8899 (ep 15) | | |
| pseudo L2 fold 0 (SWA top-3) | 0.8895 | | |
| v1 fold 0 | 0.8447 | | |
| gold (veto) | 0.9202 | | |
| GPU-h | 2.51 | | |

**Qué aprendimos:**

**Siguiente paso sugerido:**
