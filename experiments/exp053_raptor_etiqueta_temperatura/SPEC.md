# exp053_raptor_etiqueta_temperatura

| Campo | Valor |
|---|---|
| Autor | rodrigo (agente T4a) |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | exp053_raptor_etiqueta_temperatura (desde exp031_raptor_etiquetas_consenso) |
| Estado | en curso |

Idea (paso 7 T4): las etiquetas de consenso son muy blandas (promedio de 4 LLMs con escalas distintas). Afilarlas con
una temperatura las acerca a 0/1 sin cambiar qué lado de 0.5 queda cada celda (la binarización no cambia).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** entrenar con el consenso afilado p' = p^(1/T) / (p^(1/T) + (1−p)^(1/T)), T = 0.7, sube la pseudo L2
fold 0 en ≥ +0.003 contra exp031, sin que gold caiga > 0.02 y sin que la subida sea solo contra L2.

**Única variable:** transformación monótona de las etiquetas de ENTRENAMIENTO: consenso del padre → consenso afilado
(T = 0.7; T = 1 es el padre). Se aplica por celda tras promediar; NaN se conserva; p > 0.5 ↔ p' > 0.5.
`pos_weight` se recalcula con la regla del padre sobre las etiquetas transformadas (consecuencia de la variable, igual
que hizo exp031 con su consenso).

**Todo lo demás idéntico al padre:** corpus, recorte 320, CoAtNet-2 384 + `RaptorClassifier`, 16 épocas, bs 8 × k 12,
optimizador / LRs, bf16, semilla 42, `split_v1_seed42` fold 0, selección por pseudo L2 en val fold 0, SWA top-3.

**Métrica principal:** pseudo L2 fold 0 de la mejor época (padre 0.8836). SWA top-3 se reporta (padre 0.8835).

**Varas adicionales (solo informan, fijas):** pseudo fold 0 contra Raptor, L3, L4 binarizadas (> 0.5) y v1, post-hoc
sobre `oof_best_f0.csv`. Padre (best): Raptor 0.8927 · L3 0.8742 · L4 0.8836 · v1 0.8368. Ruido de semilla
(sub031_2 − exp031): L2 +0.0016, Raptor +0.0034, L3 −0.0006, L4 +0.0019. Varas independientes: Raptor y L3.

**Criterios (fijos):**
- `worked`: Δ pseudo L2 ≥ +0.003 **y** gold no cae > 0.02 **y** Δ medio contra (Raptor, L3) ≥ 0.
- `killed`: Δ pseudo L2 ≤ −0.003 **o** gold cae > 0.02.
- `didnt`: cualquier otro caso.
- **sub053_1 (T = 1.5, suavizar; una variable: solo `LABEL_T`)** se corre si exp053 "queda cerca": −0.003 < Δ pseudo
  L2 < +0.003 (`didnt`). Si exp053 es `worked`, primero su all-data y sub053_1 solo si queda pod. Si es `killed`, no.
- Si `worked`: all-data con las 3 mejores épocas del fold 0 → `swa_alldata.pt` al ensamble.

**Sesgos declarados:** la transformación no cambia la binarización de ninguna fuente, así que no acerca las etiquetas a
L2 binarizada más que el padre; sí cambia la escala del objetivo y por tanto la calibración (ECE) y `pos_weight`.

**Presupuesto:** ~2.5 GPU-h fold 0 (+2.5 sub053_1, +3 all-data), rsna-g.

## 2. PLAN

- Notebook = copia del de exp031; cambios: celda 0 (`EXP`, `PADRE`, `NOTES`, nuevo `LABEL_T = 0.7`) y celda 2
  (afilado del consenso tras construirlo + log de media del objetivo y fracción de celdas en (0.2, 0.8)).
- Smoke `SMOKE=True` → `metrics_smoke.json`; run real `FOLDS=[0]` en rsna-g tras exp052.
- Post-hoc: `multi_vara.py` para las varas Raptor/L3/L4/v1.

## 3. VERDICT

Pendiente.
