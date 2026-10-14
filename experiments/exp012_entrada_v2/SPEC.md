# exp012_entrada_v2

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp002_dinov2s_ref |
| Fecha | 2026-10-08 |
| Rama | exp012_entrada_v2 |
| Estado | running |

Idea: misma receta de exp002, pero leyendo el cache v2 de exp009 en lugar de v1_256.

Contexto en `docs/pasos/paso4_base_solida.md`.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si cambio la entrada de v1_256 (imagen completa, 224 px, hasta 5 series por prioridad simple) a cache v2
(recorte físico alrededor de la rodilla, 0.5 mm/px, rodillas alineadas, 6 slots por tipo de secuencia), la pseudo AUC
sube, porque meniscos y ligamentos ocupan más píxeles y cada slot trae la serie que corresponde.

**Única variable cambiada:** la entrada (cache y lo mínimo para leerlo).

| Parámetro o celda | Padre | Este experimento |
|---|---|---|
| Cache | `v1_256` | `v2_05mm` |
| Tamaño al backbone | 224 | 336 (múltiplo de 14 para DINOv2) |
| Selección de series | hasta 5 por prioridad | 6 slots de exp009; slot faltante enmascarado en la atención |
| Embedding de serie | plano + secuencia | id de slot (equivale: el slot ya define plano y secuencia) |

Todo lo demás igual: split `split_v1_seed42`, semilla, épocas, lr, cortes por serie, etiquetas v1, cabeza, `SELECT_ON = "pseudo"`.

**Efecto esperado:** Δ pseudo fold 0 ≈ +0.03 a +0.06; costo × ~2.3 por resolución.

**Criterios (no cambian después de correr):**
- Éxito → 5 folds: Δ pseudo fold 0 ≥ +0.005.
- Kill: Δ pseudo fold 0 ≤ −0.005, o gold < padre − 0.02 (fold 0 contra fold 0), o GPU-h > 1.5 × presupuesto.
- Neutral: `didnt`.

**Presupuesto:** screen ~4 GPU-h · 5 folds ~20 GPU-h · GPU RTX PRO 4000/4090 o mejor.

**Riesgos:**
- Memoria a 336 px: si no cabe, se baja batch y se sube acumulación de gradiente (no cuenta como segunda variable).
- Slot faltante: máscara, nunca relleno con ceros que el modelo vea como imagen.
- Cobertura: slot 1 94 %, slot 4 57 %, slot 6 47 %. Estudio sin ningún slot → se reporta y se excluye solo si son < 0.5 %.
- Tiempo en Kaggle: hay que replicar el preprocesado v2 en inferencia (pendiente para el envío, no para el screen).

## 2. PLAN

- Copia del notebook de exp002. Celdas que cambian: 0 (parámetros: cache, tamaño), 2 (datos: lector de slots v2),
  3 (modelo: embedding de slot y máscara). Ninguna otra.
- Smoke con `SMOKE = True` antes del run real.

## 3. VERDICT (números de `results/exp012_entrada_v2/metrics.json`)

**Resultado:** promising en screen (5 folds pendientes) · **Fase alcanzada:** screen

| Métrica | Padre | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | 0.827 | 0.850 | +0.023 |
| gold AUC (fold 0 contra fold 0 del padre) | 0.777 | 0.784 | +0.007 |
| OOF pseudo 5 folds | 0.802 | pendiente | |
| GPU-h | | 3.65 (219 min de reloj, GPU compartida con exp015) | 0.91× presupuesto |

**Qué aprendimos (2-4 líneas):** la entrada v2 (recorte físico, 0.5 mm, 6 slots alineados) es la mayor ganancia hasta ahora:
+0.023 pseudo con la misma receta de 8 épocas, y gold no cae (+0.007). El mejor checkpoint es la última época (8/8) y la curva
aún sube (0.843 → 0.849 → 0.850): con v2 la receta también está corta (ver exp015). Costo de entrenamiento ~2× v1 por época.

**Siguiente paso sugerido:** v2 es la entrada de la próxima referencia. Cabeza por hallazgo sobre v2 = exp018 (en curso);
poda de slots en inferencia = exp019. 5 folds se corren con la referencia combinada (v2 + cabeza + más épocas).
