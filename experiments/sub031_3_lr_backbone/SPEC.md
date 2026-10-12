# sub031_3_lr_backbone

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | sub031_3_lr_backbone |
| Estado | didnt |

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

**Resultado:** didnt · **Fase alcanzada:** screen fold 0 (rsna-d2)

| Métrica | exp031 (padre) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8836 (ep15) | 0.8832 (ep15) | −0.0005 |
| pseudo L2 fold 0 (SWA top-3) | 0.8835 | 0.8840 [15,14,11] | +0.0005 |
| v1 fold 0 (mejor época) | 0.8368 | 0.8372 | +0.0004 |
| gold (veto, mejor época) | 0.9144 | 0.9146 | +0.0003 |
| gold (SWA) | 0.9151 | 0.9148 | −0.0003 |
| GPU-h | 2.49 | 2.50 | — |

Decisión automática de la celda 5: `neutral` → `didnt` (Δ −0.0005: ni ≥ +0.003 ni ≤ −0.005; gold estable).

pseudo L2 por etiqueta (este − padre): MCL +0.023 · Medial OA +0.007 · Lateral Meniscus +0.006 · Medial Meniscus +0.003 ·
Contusion −0.002 · ACL −0.003 · PF OA −0.004 · Effusion −0.004 · Lateral OA −0.006 · Synovitis −0.006 · Fracture −0.008 ·
Baker's −0.013. Curva: ep0 0.7204 (padre 0.6955), luego converge al mismo nivel desde ep11 (0.8824).

**Qué aprendimos:** duplicar la LR del backbone acelera las primeras épocas (+0.025 en ep0) pero termina en el mismo
sitio: Δ −0.0005 (mejor época) / +0.0005 (SWA), muy dentro del ruido de semilla medido en sub031_2 (0.0013). La mejor
época sigue siendo la última, así que el límite no es la LR del backbone sino otra cosa (épocas, datos o etiquetas).

**Siguiente paso sugerido:** mantener `BB_LR = 3e-5`. No vale la pena barrer más LR; si se busca un segundo miembro de
ensamble, este modelo (distinta LR, mismo nivel) puede servir igual que otra semilla.
