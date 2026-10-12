# sub031_3_lr_backbone

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | sub031_3_lr_backbone |
| Estado | run (fold 0) |

Idea: en exp031 (receta Raptor d96 con etiquetas de consenso de 4 fuentes) subir la LR del backbone CoAtNet-2 de 3e-5 a
6e-5; la curva de exp029–exp031 sigue subiendo hasta las últimas épocas (mejor época 15 de 16 en exp031), lo que sugiere
que el backbone se adapta lento.

## 1. SPEC

**Hipótesis:** con `BB_LR = 6e-5` el backbone se adapta más en las mismas 16 épocas y la pseudo L2 de fold 0 sube
≥ +0.003 sobre exp031 sin que gold caiga más de 0.02.

**Única variable cambiada:**

| Parámetro | Padre (exp031) | Este |
|---|---|---|
| `BB_LR` (celda 0) | 3e-5 | 6e-5 |

LR de la cabeza igual (1e-3); OneCycle con el mismo `pct_start` (0.15), así que el pico y todo el calendario del backbone
escalan ×2. Todo lo demás idéntico: corpus `/root/corpus96_384`, etiquetas consenso, CoAtNet-2 384 + MIL por hallazgo,
recorte 320, 16 épocas, bs 8 × k 12, semilla 42, fold 0, selección por pseudo L2, SWA top-3.

**Criterios (no cambian después de correr)** — pseudo L2 fold 0 de la **mejor época**; exp031: **0.8836** (ep15), gold
0.9144 (`metrics_f0_backup.json` de exp031, copiado como `metrics.json` en
`/workspace/users/rodrigo/preds_ref/exp031_raptor_etiquetas_consenso/`, al que apunta el symlink del padre):
- `worked`: pseudo ≥ 0.8866 (exp031 + 0.003) **y** gold no cae > 0.02 (≥ 0.8944).
- `killed`: pseudo ≤ 0.8786 (exp031 − 0.005) **o** gold cae > 0.02.
- `didnt`: cualquier otro caso.
- El ruido de semilla lo mide sub031_2 (en curso): si resulta ≥ 0.003, un `worked` por poco margen se marca como no
  concluyente en "qué aprendimos" (el criterio no cambia).

**Presupuesto:** ~2.5 GPU-h (como exp031), rsna-d2.

**Riesgos:** LR más alta puede desestabilizar el fine-tune con bf16 (clip 3.0 lo limita); ganancia del orden del ruido.

## 2. PLAN

- Copia del notebook de exp031. Celdas que cambian: 0 (`EXP`, `PADRE`, `BB_LR`, `NOTES`) y el título markdown. Ninguna otra.
- **Smoke omitido** (declarado): el notebook es idéntico al de exp031, que ya pasó smoke y run completo; solo cambia un
  parámetro de la celda 0.
- Run real `FOLDS=[0]`, `SMOKE=False` en rsna-d2 con `RSNA_USER=rodrigo`; `results/exp031_raptor_etiquetas_consenso` en el
  clon del pod → symlink a la copia del backup fold 0 (el `metrics.json` original lo sobrescribe el all-data de rsna-f).

## 3. VERDICT (números de `results/sub031_3_lr_backbone/metrics.json`)

**Resultado:** pendiente · **Fase alcanzada:** screen fold 0 en curso

| Métrica | exp031 (padre) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8836 | | |
| pseudo L2 fold 0 (SWA top-3) | 0.8835 | | |
| v1 fold 0 | 0.8368 | | |
| gold (veto) | 0.9144 | | |
| GPU-h | 2.49 | | |

**Qué aprendimos:**

**Siguiente paso sugerido:**
