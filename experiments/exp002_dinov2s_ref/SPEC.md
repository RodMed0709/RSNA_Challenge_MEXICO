# exp002_dinov2s_ref

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | — |
| Fecha | 2026-09-24 |
| Estado | worked |

Idea: la referencia del equipo sobre el split canónico. Todos los experimentos siguientes parten de aquí.

## 1. SPEC

**Hipótesis:** DINOv2-small por corte + atención sobre cortes y luego sobre series da una referencia reproducible
contra la cual comparar todo lo demás. No busca ser el mejor modelo.

**Receta:** 224 px, hasta 5 series (prioridad secuencias de líquido, luego sagital/coronal/axial), 16 cortes por serie,
batch 2 × 4 de acumulación, 8 épocas (la primera con backbone congelado), lr cabeza 1e-4 y backbone 1e-5, AdamW 0.01,
warmup 10 % + coseno, BCE con máscara (0.5 = etiqueta suave). Checkpoint elegido por pseudo AUC del fold de validación.

Antes del split se probó la misma idea con folds hechos por cada run: con lr 1e-4 para todo el modelo el backbone se
degradaba al descongelarlo (gold 0.64); con backbone 1e-5 subió a pseudo 0.828 / gold 0.781. Esas pruebas se borraron
porque no usaban el split canónico; su aprendizaje quedó en esta receta.

**Criterios:** es la primera referencia, no hay padre. Se acepta si corre completo y gold ≥ 0.75.

**Presupuesto:** ~1.5 GPU-h por fold (RTX PRO 6000) · 5 folds ≈ 7.5 GPU-h.

## 2. PLAN
Notebook con las 8 celdas fijas. Smoke → fold 0 → 5 folds.

## 3. VERDICT (números de `results/exp002_dinov2s_ref/metrics.json`)

**Resultado:** worked · **Fase alcanzada:** 5 folds

| Métrica | Este |
|---|---|
| pseudo AUC fold 0 | 0.8266 |
| pseudo AUC por fold (0–4) | 0.827 · 0.738 · 0.809 · 0.807 · 0.817 |
| gold AUC (promedio de los 5 modelos) | 0.7884 |
| OOF pseudo 5 folds | 0.8015 |
| GPU-h | 7.2 |

**Qué aprendimos:** la receta corre y es reproducible sobre el split canónico (fold 0 igual que la prueba previa).
Fold 1 colapsa (0.738) al descongelar el backbone. El mejor checkpoint fue la última época en 4 de 5 folds: el modelo
seguía mejorando, así que 8 épocas se quedan cortas. El run terminó pero papermill falló al guardar porque la carpeta
se renombró durante el entrenamiento; `metrics.json` se reconstruyó desde `oof_f*`, `gold_f*` e `history_f*`.

**Siguiente paso sugerido:** paso 1 de `docs/Hoja_de_Ruta_Eficiencia.docx` (barrido de backbones con esta receta).
