# exp048_raptor_mixup

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | exp048_raptor_mixup |
| Estado | didnt |

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

**Resultado:** `didnt` · **Fase alcanzada:** screen fold 0 (rsna-f, 2026-10-12 04:16–06:47 UTC)

| Métrica | Padre (exp031) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8836 (ep 15) | 0.8845 (ep 11) | +0.0009 |
| pseudo L2 fold 0 (SWA top-3) | 0.8835 | 0.8847 (ep 11, 15, 14) | +0.0012 |
| v1 fold 0 | 0.8368 | 0.8374 (SWA 0.8386) | +0.0005 |
| gold (veto) | 0.9144 | 0.9067 (SWA 0.9123) | −0.0077 |
| GPU-h | 2.49 | 2.49 | 0 |

Criterio: worked exige ≥ 0.8866; killed ≤ 0.8786 o gold < 0.8944 → `didnt` (Δ dentro del ruido de semilla 0.0013).
Mixup aplicado en ~48 % de los batches (`mixup_frac` 0.47–0.50). Δ pseudo por hallazgo (mejor época): MCL +0.011,
Contusion +0.010, Baker's +0.007, Medial Meniscus +0.006; ACL −0.016, Synovitis −0.006, PF OA −0.006.

**Qué aprendimos:** mixup de bolsas a nivel estudio (p 0.5, Beta(0.4, 0.4)) no mueve la pseudo L2 con 16 épocas: la
pérdida de train queda más alta (0.778 contra 0.742 al final) pero la curva de val es la del padre; la mejor época se
adelanta a la 11 y gold de esa época es más bajo (0.907), en SWA se recupera (0.912). Sin señal para seguir: ni sub
de intensidad ni all-data.

**Siguiente paso sugerido:** ninguno en esta línea con 16 épocas. Si se retoma, mixup tendría sentido solo con más
épocas (exp033 vio techo en la 14 sin aug) o en feature space (mezclar embeddings de ventanas en vez de píxeles).
