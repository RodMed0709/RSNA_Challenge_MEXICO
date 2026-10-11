# exp033_raptor_24epocas

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp029_raptor_entreno |
| Fecha | 2026-10-11 |
| Rama | exp033_raptor_24epocas |
| Estado | didnt |

Idea: en la receta Raptor/nartaa d96/384 de exp029, entrenar 24 épocas en lugar de 16. Origen: en nuestra línea
más épocas sumó (exp015: 8 → 16 épocas, +0.020 pseudo y +0.020 gold en fold 0; exp020: 8 → 16 sobre la cabeza por
hallazgo, +0.012/+0.013, mejor época 12).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** entrenar la receta de exp029 **24 épocas** en lugar de 16 sube la pseudo AUC L2 en fold 0 en ≥ +0.005
sin que gold caiga más de 0.02.

**Única variable:** `EPOCHS` 16 → 24. Consecuencias directas (no son variables aparte): el OneCycle se estira a
24 épocas (`total_steps = pasos/época × 24`, mismo `PCT_START = 0.15`, mismos LR máximos), y la selección por pseudo L2
y el SWA top-3 por pseudo L2 se hacen sobre 24 épocas en lugar de 16 (misma regla).

**Todo lo demás idéntico al padre:** corpus `/root/corpus96_384` (96×384), recorte central 320, CoAtNet-2 384 +
`RaptorClassifier`, etiquetas blandas de Raptor (`labels_llm_soft`), bs 8 × k 12 (k_eval 24), AdamW / LRs, bf16,
semilla 42, `split_v1_seed42` fold 0, selección por pseudo L2 (L2 binarizada > 0.5) en val fold 0, SWA = media de las 3
mejores épocas por pseudo L2. Gold y v1 solo se reportan.

**Métrica principal:** la del padre: pseudo L2 fold 0 de la **mejor época**. SWA top-3 se reporta.

**Criterios (fijos, no cambian después de correr):**
- `worked`: pseudo L2 fold 0 ≥ pseudo padre + 0.005 **y** gold no cae > 0.02 contra el padre.
- `killed`: pseudo L2 fold 0 ≤ pseudo padre − 0.005 **o** gold cae > 0.02 contra el padre.
- `didnt`: cualquier otro caso.

**Presupuesto:** ~4 GPU-h en fold 0 (padre ≈ 2.5 h para 16 épocas → 24 épocas ≈ 3.8 h) en rsna-g (RTX PRO 4500 32 GB),
corpus en `/root/corpus96_384`. El coste de inferencia no cambia (mismo modelo); el de entrenamiento all-data
subiría ×1.5.

**Riesgos:** sobreajuste a las etiquetas de Raptor con más épocas (la vara es otra fuente, L2); comparación mejor época
de 24 vs de 16 da 8 oportunidades más de selección en val (sesgo optimista pequeño de la selección por máximo);
el padre aún corre, la comparación se completa cuando termine.

## 2. PLAN

- Notebook `exp033_raptor_24epocas.ipynb` = copia del de exp029; cambios: solo celda 0 (`EXP`, `PADRE`, `EPOCHS = 24`,
  `NOTES`).
- Smoke: `SMOKE=True` (64 train / 48 val, 1 época) → `metrics_smoke.json`.
- Run real: `FOLDS=[0]`, `SMOKE=False`, en rsna-g.

## 3. VERDICT (números de `results/exp033_raptor_24epocas/metrics.json`)

**Resultado:** `didnt` · **Fase alcanzada:** screen fold 0

| Métrica | Padre (exp029) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8682 (ep 15 de 16) | 0.8695 (ep 14 de 24) | +0.0013 |
| pseudo L2 fold 0 (SWA top-3) | 0.8679 (ep 15, 14, 13) | 0.8715 (ep 14, 15, 17) | +0.0037 |
| v1 fold 0 | 0.8329 | 0.8302 | −0.0026 |
| gold (veto) | 0.9115 | 0.9195 (SWA 0.9212) | +0.0080 |
| GPU-h | 2.51 | 3.79 | ×1.51 |

Criterio: Δ pseudo +0.0013 está entre −0.005 y +0.005 y gold no cae → **didnt** (`decision_auto` = `neutral`).

**Qué aprendimos:** con 24 épocas la pseudo L2 toca techo en la época 14 (0.8695) y después baja lentamente hasta
0.8651 en la 23; el OneCycle más largo no aporta sobre 16 épocas y cuesta ×1.5 GPU-h. A diferencia de exp015/exp020
(DINOv2, 8 → 16), en la receta Raptor 16 épocas ya están en la meseta. Gold sube +0.008 (solo veto, 58 estudios: no
decide). Mantener 16 épocas.

**Siguiente paso sugerido:** ninguno en esta línea; cerrar.
