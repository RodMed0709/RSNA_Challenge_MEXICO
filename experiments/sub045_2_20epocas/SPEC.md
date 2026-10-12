# sub045_2_20epocas

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial |
| Fecha | 2026-10-12 |
| Rama | sub045_2_20epocas |
| Estado | spec |

Idea: ronda 2 sobre exp045 (`worked`, pseudo L2 0.8899). La curva de exp045 seguía subiendo en la época 15
(0.8884 → 0.8896 → 0.8899): con augmentation el modelo sobreajusta menos y puede aprovechar más épocas. exp033 (24 épocas
sin aug) tocó techo en la 14; aquí la aug cambia la situación.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** entrenar 20 épocas en vez de 16 (OneCycle reescalado al nuevo total de pasos, mismo `pct_start`) sube la
pseudo L2 fold 0 en ≥ +0.003 contra exp045.

**Única variable:** `EPOCHS` 16 → 20. `OneCycleLR(total_steps = pasos/época × EPOCHS, pct_start 0.15)` se reescala solo;
SWA top-3 por pseudo L2 como siempre. Todo lo demás idéntico a exp045 (espejo + afín leve).

**Métrica principal:** pseudo L2 fold 0 de la mejor época; SWA top-3 se reporta.

**Criterios (fijos, contra exp045):**
- `worked`: pseudo L2 fold 0 ≥ **0.8929** y gold ≥ **0.9002**.
- `killed`: pseudo L2 fold 0 ≤ **0.8849** o gold < 0.9002, o GPU-h > 1.5 × 2.51 × 20/16.
- `didnt`: cualquier otro caso.
(exp045: pseudo L2 0.8899 (ep 15), SWA 0.8895, gold 0.9202. Ruido de semilla 0.0013.)
Si promete: avisar al orquestador antes de lanzar all-data (saldo justo).

**Presupuesto:** ≈ 3.1 GPU-h fold 0 en rsna-d2 (tras exp054 de otro agente).

## 2. PLAN

- Notebook `sub045_2_20epocas.ipynb` = copia del de exp045; cambia solo la celda 0 (`EXP`, `PADRE`, `EPOCHS = 20`, `NOTES`).
- Sin smoke nuevo (mismo código que exp045); run real `FOLDS=[0]` en rsna-d2 cuando quede libre.

## 3. VERDICT (números de `results/sub045_2_20epocas/metrics.json`)

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
