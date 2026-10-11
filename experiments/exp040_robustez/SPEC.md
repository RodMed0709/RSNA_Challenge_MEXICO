# exp040_robustez

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp029 (modelo Raptor d96/384 fold 0, `best_f0.pt`) |
| Fecha | 2026-10-11 |
| Rama | exp040_robustez |
| Estado | worked |

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

**Resultado:** worked (32 pasadas completas; la referencia limpia reproduce exp029: pseudo L2 0.8682 = 0.8682) ·
**Fase alcanzada:** análisis completo (fold 0 + gold) · 0.59 GPU-h de inferencia.

Referencia limpia (k24, sin TTA): pseudo L2 **0.8682** [0.8584, 0.8773] · v1 0.8329 [0.8204, 0.8453] · gold 0.9118 [0.8847, 0.9367].

| Condición | macro pseudo AUC | Δ vs limpia [IC95 pareado] | Δ v1 | Δ gold |
|---|---|---|---|---|
| TTA espejo (media) | 0.8697 | +0.0016 [+0.0002, +0.0029] | 0.0000 | −0.0018 |
| espejo solo (lateralidad) | 0.8683 | +0.0001 [−0.0026, +0.0027] | −0.0018 | −0.0075 |
| 6 / 12 ventanas | 0.8338 / 0.8554 | −0.0343 / −0.0128 | −0.028 / −0.013 | −0.058 / −0.019 |
| 48 / todas (~94) ventanas | 0.8728 / 0.8734 | +0.0047 [+0.0027, +0.0068] / +0.0052 [+0.0029, +0.0075] | +0.002 / +0.003 | 0.000 / 0.000 |
| ruido gauss 2 % / 5 % / 10 % | 0.8669 / 0.8567 / 0.8333 | −0.0012 / −0.0114 [−0.0154, −0.0076] / −0.0349 | −0.001 / −0.009 / −0.030 | 0.000 / −0.012 / −0.040 |
| Rician 5 % / 10 % | 0.8568 / 0.8346 | −0.0114 / −0.0336 | −0.009 / −0.029 | −0.009 / −0.039 |
| gamma 0.7 / 1.4 | 0.8664 / 0.8669 | −0.0018 / −0.0013 | −0.002 / −0.002 | −0.001 / 0.000 |
| contraste +20 % / −20 % | 0.8673 / 0.8665 | −0.0009 / −0.0016 | −0.002 / 0.000 | −0.003 / −0.001 |
| bias field exp(±0.2) | 0.8671 | −0.0010 [−0.0023, +0.0004] | −0.001 | +0.003 |
| 50 % de cortes | 0.8613 | −0.0069 [−0.0114, −0.0028] | −0.006 | −0.015 |
| 25 % de cortes | 0.8339 | −0.0343 [−0.0414, −0.0280] | −0.027 | −0.053 |
| desplazamiento +2 / −2 cortes | 0.8687 / 0.8678 | +0.0005 / −0.0003 | −0.002 / −0.004 | −0.007 / −0.008 |
| sin Sag fluido (slot 0) | 0.8562 | −0.0119 [−0.0161, −0.0079] | −0.010 | −0.016 |
| sin Sag no fluido (1) | 0.8641 | −0.0040 | −0.004 | −0.005 |
| sin Cor fluido (2) | 0.8601 | −0.0081 | −0.003 | −0.014 |
| sin Cor (3) | 0.8678 | −0.0003 | −0.001 | −0.001 |
| sin Axial (4) | 0.8682 | 0.0000 [−0.0038, +0.0038] | 0.000 | −0.009 |
| sin coronal (2+3) | 0.8592 | −0.0089 | −0.005 | −0.016 |
| sin sagital (0+1) | 0.8402 | −0.0280 [−0.0343, −0.0220] | −0.024 | −0.037 |
| 384→192→384 | 0.8609 | −0.0073 [−0.0112, −0.0031] | −0.006 | −0.003 |
| recorte ±16 px (x, y) | 0.8673–0.8684 | −0.0009 … +0.0002 | ≤ 0.001 | ≤ 0.002 |

Hipótesis: **H1 refutada a 5 %** (Δ −0.0114, el IC [−0.0154, −0.0076] queda por debajo de −0.01 en su mayor parte;
a 2 % Δ −0.0012 sí cumple) · **H2 confirmada** (50 % de cortes Δ −0.0069) · **H3 confirmada** (el slot 0, Sag fluido,
es el que más pesa, −0.0119; le sigue Cor fluido −0.0081; Cor no fluido y Axial ≈ 0) · **H4 refutada**: ECE media 0.203
(Brier 0.160) contra L2 bin: sobre-predicción sistemática (media de predicción 0.37–0.50 frente a prevalencias 0.08–0.58),
consecuencia esperable del `pos_weight` hasta 10 en la BCE. Afecta a la calibración, no a la AUC.

Subgrupos (pseudo L2, limpia): sin diferencias claras por fabricante (Siemens 0.872, GE 0.872, Philips 0.858,
Canon/Toshiba 0.857, n=45), campo (1.5 T 0.869 / 3 T 0.869), lado (L 0.867 / R 0.869) ni sexo (F 0.873 / M 0.860);
por idioma del informe (≈ hospital) el croata es el peor (hr 0.817 [0.767, 0.861], n=73) frente a en 0.880. Edad: la
etiqueta DICOM está vacía en todos los estudios (no hay subgrupo de edad).

**Qué aprendimos:** el modelo es robusto a intensidad (gamma, contraste, bias field), encuadre (±16 px) y
desplazamientos de ±2 cortes (|Δ| ≤ 0.002), y sensible a ruido ≥ 5 %, a quitar el plano sagital y a submuestrear al 25 %.
El número de ventanas en inferencia importa más que los cortes: pasar de 24 a 48 ventanas da +0.005 pseudo (todas,
~94, +0.005 con ×4 de cómputo), y bajar a 12 cuesta −0.013. El TTA espejo aporta poco (+0.0016) por el doble de cómputo.
El axial y el coronal no fluido apenas aportan.

**Siguiente paso sugerido:** (1) eficiencia: inferencia a 48 ventanas sin TTA (≈ mismo costo que 24 + TTA y mejor);
probar un corpus con 50 % de cortes (−0.007) para recortar lectura de DICOM; quitar el slot Axial y el Cor no fluido
(≈ 0 pérdida) y medir el ahorro de E/S. (2) recalibración post-hoc (Platt/temperatura por hallazgo en val) si alguna vez
importa la probabilidad (no cambia la AUC). (3) aumento de ruido en entrenamiento si se busca robustez a ruido ≥ 5 %.
