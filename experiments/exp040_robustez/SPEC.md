# exp040_robustez

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp029 (modelo Raptor d96/384 fold 0, `best_f0.pt`) |
| Fecha | 2026-10-11 |
| Rama | exp040_robustez |
| Estado | spec |

Idea: análisis de robustez **solo inferencia** del modelo Raptor fold 0 de exp029 frente a perturbaciones de
imagen y de adquisición, subgrupos y calibración. No entrena nada. Material para el paper (sección de robustez) y para
decidir palancas de runtime del track de eficiencia (menos cortes, menos ventanas).

## 1. SPEC (antes de tocar el código)

**Modelo evaluado:** `results/exp029_raptor_entreno/best_f0.pt` (época 15, pseudo L2 0.8682; el SWA top-3 da 0.8679,
así que se usa `best`). Elegido por pseudo L2 (nunca gold). Inferencia idéntica a la validación de exp029: k_eval = 24
ventanas 2.5D, recorte central 320 del corte de 384, normalización ImageNet, bf16.

**Datos:** validación fold 0 de `split_v1_seed42` (871 estudios) + gold (58). Corpus `/workspace/data/cache/raptor96_384`
(memmap; los 929 estudios se cachean en el disco local del pod para no releer el volumen en cada pasada).

**Hipótesis (fijas antes de correr):**
- H1: con ruido gaussiano σ ≤ 5 % del rango, Δ macro pseudo AUC > −0.01 (contra la referencia limpia sin TTA).
- H2: con 50 % de los cortes por slot, Δ macro pseudo AUC > −0.01.
- H3: el slot cuya ausencia más baja la pseudo AUC es el **sagital fluido** (slot 0).
- H4: el modelo está razonablemente calibrado: ECE media sobre los 12 hallazgos < 0.10 (contra L2 binarizada > 0.5).

**Condiciones** (cada una = una pasada de inferencia sobre los 929 estudios; perturbación determinista por estudio):

| Grupo | Condiciones |
|---|---|
| 1 Referencia | limpia (k24); espejo anatómico solo; TTA = media(limpia, espejo) (sin pasada extra) |
| 1b Ventanas (extra, eficiencia) | k_eval = 6, 12, 48, 94 |
| 2 Ruido | gaussiano σ = 0.02, 0.05, 0.10 (rango [0,1]); Rician aprox. √((x+n₁)²+n₂²) σ = 0.05, 0.10 |
| 3 Intensidad | gamma 0.7 y 1.4; contraste ±20 % alrededor de la media del estudio; bias field multiplicativo suave (campo 2D de baja frecuencia, exp(±0.2)) |
| 4 Cortes | 50 % y 25 % de los cortes de cada slot (submuestreo uniforme); desplazamiento ±2 cortes dentro del slot |
| 5 Series faltantes | quitar slot 0 (Sag fluido), 1 (Sag no fluido), 2 (Cor fluido), 3 (Cor), 4 (Ax = plano axial); quitar coronal (2+3); quitar sagital (0+1) |
| 6 Resolución / encuadre | 384→192→384 (área + bilineal); recorte desplazado ±16 px en x y en y |
| 7 Lateralidad | espejo anatómico (sagital: invertir canales; coronal/axial: flip horizontal) = condición de 1 |
| 8 Subgrupos (limpia) | fabricante, campo (T), sexo (DICOM `PatientSex`), edad si existe, idioma del informe (≈ hospital), lado de la rodilla |
| 9 Calibración (limpia) | ECE (10 bins) y Brier por hallazgo contra L2 bin y gold; diagrama de fiabilidad |

Ruido y perturbaciones de intensidad se aplican sobre el volumen en [0,1] solo a los cortes válidos (los cortes vacíos
representan series ausentes y quedan a cero). Una serie faltante se simula igual que la representa el corpus: cortes a
cero y máscara 0.

**Métricas:** macro AUC (y por hallazgo) contra (a) L2 binarizada > 0.5 en val fold 0 (pseudo, la vara principal),
(b) v1 (0/1, sin 0.5) en val fold 0, (c) gold-58. IC95 por bootstrap pareado por estudio (1000 réplicas, semilla 42):
para cada condición se reporta AUC con IC y Δ contra la referencia limpia con IC (mismas réplicas).

**Criterios (no cambian después de correr):** H1–H4 se declaran confirmadas o refutadas con el punto estimado de Δ
(y se informa si el IC95 cruza −0.01). No hay worked/killed de entrenamiento: el verdict es `worked` si todas las
pasadas corren y la referencia limpia reproduce la pseudo L2 de exp029 (|Δ| < 0.002); si no, `didnt` y se revisa el
pipeline.

**Presupuesto:** ~32 pasadas × ~1.5 min ≈ 1 GPU-h (RTX PRO 4500, pod rsna-e) + cache 13 GB.

**Riesgos:** bf16 no determinista (diferencias ~1e-4); gold-58 tiene IC enormes (solo informativo); los subgrupos
pequeños pueden no tener positivos en algún hallazgo (se excluye ese hallazgo del macro del subgrupo y se informa n).

## 2. PLAN

- Scripts auxiliares declarados (solo inferencia y análisis, no entrenan):
  - `infer_robustez.py`: carga `best_f0.pt`, cachea los 929 estudios, corre cada condición y guarda
    `results/exp040_robustez/preds/<cond>.parquet`.
  - `analisis_robustez.py`: bootstrap, tablas csv, figuras, subgrupos, calibración y `metrics.json`.
  - `meta_subgrupos.py`: extrae fabricante/campo (meta_v2), sexo/edad (cabecera DICOM), idioma y lado.
- Smoke: `--smoke` (20 estudios, 3 condiciones).
- Run real en rsna-e (una GPU, sin otros runs).

## 3. VERDICT (números de `results/exp040_robustez/metrics.json`)

**Resultado:** pendiente
