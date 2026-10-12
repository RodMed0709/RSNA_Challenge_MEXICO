# sub053_1_temperatura_suave

| Campo | Valor |
|---|---|
| Autor | rodrigo (agente T4a) |
| Padre | exp031_raptor_etiquetas_consenso (sub-experimento de intensidad de exp053) |
| Fecha | 2026-10-12 |
| Rama | sub053_1_temperatura_suave (desde exp053_raptor_etiqueta_temperatura) |
| Estado | abortado (sin verdict; queda como idea) |

Sub-experimento previsto en la SPEC de exp053: se corre porque exp053 (T = 0.7) "quedó cerca" (|Δ pseudo L2| < 0.003).
Prueba la dirección opuesta: suavizar el consenso en vez de afilarlo.

## 1. SPEC

**Hipótesis:** el consenso suavizado p' = p^(1/T) / (p^(1/T) + (1−p)^(1/T)) con **T = 1.5** sube la pseudo L2 fold 0
en ≥ +0.003 contra exp031, sin que gold caiga > 0.02 y sin que la subida sea solo contra L2.

**Única variable:** `LABEL_T` = 1.5 (exp031 equivale a T = 1; exp053 usó 0.7). Notebook idéntico al de exp053 salvo
`EXP`, el título y `LABEL_T`. `PADRE` = exp031 (el `decision_auto` compara contra exp031); exp053 se reporta al lado.
`pos_weight` recalculado con la regla del padre sobre las etiquetas transformadas (como exp053).

**Todo lo demás idéntico a exp031/exp053**; selección por pseudo L2, SWA top-3, gold solo veta.

**Varas adicionales (fijas):** pseudo contra Raptor, L3, L4 binarizadas y v1, post-hoc (`multi_vara.py`).
Padre (best): Raptor 0.8927 · L3 0.8742 · L4 0.8836 · v1 0.8368.

**Criterios (fijos, los de exp053):**
- `worked`: Δ pseudo L2 ≥ +0.003 vs exp031 **y** gold no cae > 0.02 **y** Δ medio contra (Raptor, L3) ≥ 0 → all-data.
- `killed`: Δ pseudo L2 ≤ −0.003 **o** gold cae > 0.02.
- `didnt`: otro caso.

**Presupuesto:** ~2.5 GPU-h, rsna-g.

## 2. PLAN

- Notebook = copia del de exp053 con `LABEL_T = 1.5`; smoke `SMOKE=True`, luego `FOLDS=[0]` en rsna-g tras exp053.

## 3. VERDICT

**Abortado, sin conclusión.** La cola lo lanzó automáticamente a las 09:21 UTC (exp053 quedó en |Δ| < 0.003). El
orquestador ordenó después no lanzar nada más y apagar rsna-g; se mató en la época 7 de 16 (~1.2 GPU-h). Sin
`metrics.json`. Curva parcial (pseudo L2 por época, de `history_f0.csv`): 0.7199 · 0.7689 · 0.8230 · 0.8353 · 0.8520 ·
0.8603 · 0.8660 · 0.8730 (exp031 en las mismas épocas: 0.6955 · 0.7935 · 0.8321 · 0.8357 · 0.8449 · 0.8630 · 0.8664 ·
0.8773). No permite decidir nada. Queda como idea: T = 1.5 (media del objetivo 0.302 → 0.337, celdas en (0.2, 0.8)
20 % → 42 %).
