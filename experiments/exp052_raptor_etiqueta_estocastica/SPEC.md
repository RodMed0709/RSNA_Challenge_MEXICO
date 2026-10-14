# exp052_raptor_etiqueta_estocastica

| Campo | Valor |
|---|---|
| Autor | rodrigo (agente T4a) |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | exp052_raptor_etiqueta_estocastica (desde exp031_raptor_etiquetas_consenso) |
| Estado | didnt |

Idea ("augmentation de texto", paso 7 T4): en vez de entrenar contra el promedio fijo de las 4 fuentes de etiquetas, en
cada época cada estudio recibe como objetivo **una** fuente elegida al azar. El modelo ve el desacuerdo entre LLMs como
ruido de etiqueta (regularización tipo label-noise augmentation); el objetivo esperado sigue siendo el consenso.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** muestrear la fuente de etiqueta por estudio y época (Raptor / L2 / L3 / L4, uniforme entre las que tienen
el estudio) en lugar del promedio por celda sube la pseudo L2 fold 0 en ≥ +0.003 contra exp031, sin que gold caiga
> 0.02 y sin que la subida sea solo contra L2.

**Única variable:** cómo se forma el objetivo de ENTRENAMIENTO. Padre: media por celda de las 4 fuentes (fija) → este:
en cada paso, para cada estudio del batch, `s ~ U{fuentes con ese estudio}` y objetivo = fila de la fuente `s`
(blanda, tal cual). Cada estudio aparece una vez por época (shuffle, drop_last), así que es un sorteo por estudio y
época. Celda ausente en la fuente elegida → valor del consenso (en fold 0 y all-data no ocurre: 4/4 fuentes en todas
las celdas, medido en exp031). RNG propio `np.random.default_rng(SEED + fold)` para no alterar el shuffle ni el
muestreo de ventanas. `pos_weight` = el del padre (regla del padre sobre el consenso de train): el objetivo esperado
es el mismo consenso, así que no cambia.

**Todo lo demás idéntico al padre:** corpus `/root/corpus96_384`, recorte 320, CoAtNet-2 384 + `RaptorClassifier`,
16 épocas, bs 8 × k 12 (k_eval 24), AdamW / OneCycle / LRs, bf16, semilla 42, `split_v1_seed42` fold 0, selección por
pseudo L2 (L2 binarizada > 0.5) en val fold 0, SWA top-3 por pseudo L2. Validación sin cambios.

**Métrica principal:** pseudo L2 fold 0 de la mejor época (como el padre: 0.8836). SWA top-3 se reporta (padre 0.8835).

**Varas adicionales (solo informan, se fijan ahora):** pseudo fold 0 contra Raptor, L3 y L4 binarizadas (> 0.5) y v1,
calculadas post-hoc sobre `oof_best_f0.csv` con el mismo `mean_auc`. Padre (best): Raptor 0.8927 · L3 0.8742 ·
L4 0.8836 · v1 0.8368. Ruido de semilla (sub031_2 − exp031): L2 +0.0016, Raptor +0.0034, L3 −0.0006, L4 +0.0019.
L4 ≈ copia de L2 (acuerdo 99 %), así que las varas independientes son Raptor y L3.

**Criterios (fijos, no cambian después de correr):**
- `worked`: Δ pseudo L2 ≥ +0.003 **y** gold no cae > 0.02 **y** Δ medio contra (Raptor, L3) ≥ 0 (no es solo L2).
- `killed`: Δ pseudo L2 ≤ −0.003 **o** gold cae > 0.02.
- `didnt`: cualquier otro caso (incluye Δ L2 ≥ +0.003 pero Δ medio Raptor/L3 < 0 → sesgo hacia L2).
- Si `worked`: run all-data en el mismo pod con las 3 mejores épocas del fold 0 → `swa_alldata.pt` al ensamble.

**Sesgos declarados:** el objetivo toma L2 (o su casi-copia L4) la mitad de las veces, igual que el peso de L2+L4 en
el consenso del padre: la exposición a L2 no cambia en esperanza. El ruido del objetivo puede ralentizar la
convergencia en 16 épocas (el padre mejora hasta la última).

**Presupuesto:** ~2.5 GPU-h (fold 0) + ~3 GPU-h all-data si promete, rsna-g.

## 2. PLAN

- Notebook = copia del de exp031; cambios: celda 0 (`EXP`, `PADRE`, `NOTES`) y celda 4 (`train_fold`: tensor de
  fuentes por estudio, sorteo por estudio en cada paso, conteo de fuentes usadas por época en `history`).
- Smoke `SMOKE=True` → `metrics_smoke.json`; run real `FOLDS=[0]` en rsna-g tras sub031_1.
- Post-hoc: `multi_vara.py` (scratchpad del agente; lee `oof_{best,swa}_f0.csv`) para las varas Raptor/L3/L4/v1.

## 3. VERDICT (números de `results/exp052_raptor_etiqueta_estocastica/metrics.json`; varas extra con `multi_vara.py` sobre `oof_*_f0.csv`)

**Resultado:** `didnt` · **Fase alcanzada:** screen fold 0

| Métrica (fold 0, val 871) | Padre (exp031) | Este | Δ | ruido semilla |
|---|---|---|---|---|
| pseudo L2 (mejor época) | 0.8836 (ep 15) | **0.8851** (ep 15) | **+0.0014** | +0.0016 |
| pseudo L2 (SWA top-3) | 0.8835 | 0.8847 (ep 15, 14, 13) | +0.0012 | |
| pseudo Raptor bin (best) | 0.8927 | 0.9003 | +0.0076 | +0.0034 |
| pseudo L3 bin (best) | 0.8742 | 0.8720 | −0.0022 | −0.0006 |
| pseudo L4 bin (best) | 0.8836 | 0.8847 | +0.0011 | +0.0019 |
| v1 (best) | 0.8368 | 0.8392 | +0.0024 | +0.0026 |
| gold (veto) | 0.9144 | 0.9150 (SWA 0.9153) | +0.0006 | |
| GPU-h | 2.49 | 2.53 | +0.04 | |

Criterio: Δ pseudo L2 +0.0014 < +0.003 (y > −0.003, gold no cae) → **didnt** (`decision_auto` = `neutral`).
Δ medio (Raptor, L3) = +0.0027, pero el criterio principal no se cumple; no se lanza all-data.

**Qué aprendimos:** el objetivo estocástico converge más lento al principio (ep 1: 0.761 vs 0.793 del padre) y
alcanza al padre hacia la ép. 11; termina dentro del ruido de semilla contra L2/L4/v1. El único movimiento fuera del
ruido es **contra Raptor (+0.008)**: entrenar con la fila cruda de Raptor 1/4 de las veces (en vez de diluida en la
media) acerca el modelo a la "opinión" de Raptor, y baja algo contra L3 (−0.002). Es decir: cada vara premia la
fuente que el objetivo expone más, sin ganancia neta clara. Por hallazgo (pseudo L2): MCL +0.016, Contusion +0.010;
ACL −0.008, Fracture −0.008, Synovitis −0.006. Como regularizador de ruido de etiqueta no aporta en 16 épocas.

**Siguiente paso sugerido:** ninguno directo. Si se reintenta: más épocas (la curva aún sube en la 14–15) o muestreo
por celda en vez de por estudio; prioridad baja.
