# exp048_raptor_mixup

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | exp048_raptor_mixup |
| Estado | spec |

Idea (paso 7, track T2 estructura/mixup): la receta Raptor casi no tiene augmentation y exp033 (24 épocas) mostró
sobreajuste al final. Mixup con etiquetas blandas es un regularizador estándar que además suaviza la calibración
(ECE 0.20 en exp040).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** mixup a nivel estudio sube la pseudo AUC L2 en fold 0 en ≥ +0.003 contra exp031 sin que gold caiga
más de 0.02.

**Única variable (solo train, en GPU, en el bucle de la celda 4):** con p **0.5** por batch, λ ~ Beta(**0.4**, 0.4) y
una permutación aleatoria del batch: `x = λ·x + (1−λ)·x[perm]` sobre la bolsa normalizada (B, K, 3, 320, 320) —
ventana k de un estudio con ventana k del otro— e `y = λ·y + (1−λ)·y[perm]` sobre las etiquetas blandas de consenso.
Celda NaN en cualquiera de los dos estudios → NaN → enmascarada en la pérdida (como el padre). Validación, gold e
inferencia idénticas al padre.

**Todo lo demás idéntico al padre:** etiquetas de consenso, corpus `/root/corpus96_384`, recorte 320, `RaptorClassifier`,
16 épocas, bs 8 × K 12, AdamW / OneCycle / LRs, pos_weight, bf16, semilla 42, `split_v1_seed42` fold 0, selección por
pseudo L2 (L2 binarizada > 0.5), SWA top-3.

**Métrica principal:** pseudo L2 fold 0 de la mejor época. SWA top-3 se reporta.

**Criterios (fijos, no cambian después de correr):** padre exp031 pseudo L2 0.8836, gold 0.9144.
- `worked` (promete): pseudo L2 ≥ **0.8866** (+0.003) **y** gold ≥ 0.8944.
- `killed`: pseudo L2 ≤ **0.8786** (−0.005) **o** gold < 0.8944.
- `didnt`: cualquier otro caso.
La `decision_auto` usa ±0.005; el verdict se aplica a mano con estos criterios (ruido de semilla medido 0.0013).

**Presupuesto:** fold 0 ≈ 2.5 GPU-h en rsna-f (mezcla en GPU, coste despreciable). Si promete: sub de intensidad
(p 1.0) y all-data con las 3 mejores épocas del fold 0.

**Riesgos:** con 16 épocas el mixup puede frenar la convergencia; mezclar ventanas no alineadas anatómicamente (slot y
posición distintos con K aleatorio) crea imágenes poco realistas; la atención MIL puede aprender a localizar el
estudio dominante en vez del hallazgo.

## 2. PLAN

- Notebook `exp048_raptor_mixup.ipynb` = copia del de exp031; cambios: celda 0 (`EXP`, `PADRE`, `MIXUP_P`,
  `MIXUP_ALPHA`, `NOTES`), celda 4 (mixup en el paso de train + `mixup_frac` en el historial).
- Smoke: `SMOKE=True` → `metrics_smoke.json`.
- Run real: `FOLDS=[0]`, `SMOKE=False`, en rsna-f (tras sub031_4).

## 3. VERDICT (números de `results/exp048_raptor_mixup/metrics.json`)

**Resultado:** pendiente · **Fase alcanzada:** spec

| Métrica | Padre (exp031) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8836 (ep 15) | | |
| pseudo L2 fold 0 (SWA top-3) | 0.8835 | | |
| v1 fold 0 | 0.8368 | | |
| gold (veto) | 0.9144 | | |
| GPU-h | 2.49 | | |

**Qué aprendimos:**

**Siguiente paso sugerido:**
