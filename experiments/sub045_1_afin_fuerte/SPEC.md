# sub045_1_afin_fuerte

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial |
| Fecha | 2026-10-12 |
| Rama | sub045_1_afin_fuerte |
| Estado | didnt |

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

**Resultado:** `didnt` · **Fase alcanzada:** screen fold 0 (rsna-e2)

| Métrica | exp045 | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8899 (ep 15) | 0.8876 (ep 15) | −0.0023 |
| pseudo L2 fold 0 (SWA top-3) | 0.8895 | 0.8875 (ep 15, 13, 14) | −0.0020 |
| v1 fold 0 | 0.8447 | 0.8409 | −0.0038 |
| gold (veto) | 0.9202 | 0.9180 | −0.0021 |
| GPU-h | 2.51 | 2.48 | −1 % |

Criterio: pseudo 0.8876 < 0.8929 (no promete) y > 0.8849 (no killed); gold 0.9180 ≥ 0.9002 → `didnt`.
Contra exp031 (0.8836) sigue +0.0040: el espejo + afín sigue ayudando, pero más afín no suma.

**Qué aprendimos:** duplicar la magnitud del afín (rot ±20°, escala 0.8–1.2, traslación ±10 %) empeora un poco todo
(pseudo −0.002, v1 −0.004, gold −0.002) y retrasa la curva (ep 5 0.8492 vs 0.8620 de exp045). El afín leve de exp045
(±10°, 0.9–1.1, ±6 %) es el punto bueno; con 16 épocas la deformación fuerte cuesta más de lo que regulariza.

**Siguiente paso sugerido:** mantener la aug espacial de exp045 tal cual en el combo exp056. Si sobra cómputo, el
eje pendiente es la duración (sub045_2, 20 épocas, cancelado por prioridad).
