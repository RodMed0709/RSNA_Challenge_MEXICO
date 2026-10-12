# exp054_etiquetas_L5

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-11 |
| Rama | exp054_etiquetas_L5 |
| Estado | worked |

Idea (paso 7, track T4 "augmentation de texto"): el modelo solo ve imágenes; el texto entra por las etiquetas. Fabricar
una fuente propia de etiquetas blandas, **L5**, re-leyendo los 4407 informes de train con un LLM y **k = 3 prompts
distintos** (extracción directa, razonamiento breve, paráfrasis/traducción al inglés), promediados por celda; y sumarla
como 5.ª fuente al consenso de exp031.

Dos fases. **Fase 1** (CPU + API, sin GPU): generar L5 y validarla sin gold. **Fase 2** (GPU): exp054 = exp031 con
consenso de 5 fuentes. La fase 2 solo corre si L5 pasa el chequeo de salud de la fase 1.

## 1. SPEC (antes de tocar el notebook)

### Fase 1 — generar L5

| Punto | Decisión (fija) |
|---|---|
| Modelo | `gpt-4.1-mini` (temperatura 0, `seed` 54, salida JSON con esquema estricto). Barato (US$0.40 / 1.60 por M tokens in/out) y de otra familia que L4 (GPT-5.6-Sol), para que la fuente sea distinta y no otra copia de L2/L4. |
| Prompts (k = 3) | **A_extract**: estado por hallazgo `present` / `present_low_grade` / `uncertain` / `absent` / `not_mentioned` → 0.95 / 0.25 / 0.45 / 0.01 / 0.05. **B_reason**: notas breves (≤ 60 palabras) citando/traduciendo lo clave y luego probabilidad 0–100 por hallazgo con guía de calibración. **C_paraphrase**: paráfrasis en inglés (≤ 80 palabras) del informe y luego probabilidad 0–100 usando solo la paráfrasis. Los tres usan las definiciones oficiales (discusión 733343, mismo glosario de exp010/exp011). |
| L5 | media por celda de las 3 lecturas (probabilidad en [0, 1]); 12 columnas en el orden y con los nombres de las otras fuentes. |
| Dónde vive | `/workspace/data/labels/ours/L5_20261011.csv` (+ las 3 lecturas sueltas `L5_20261011__<prompt>.csv`). Dato: no va a git. El código generador va en esta carpeta (`generar_L5.py`, CPU/API, no entrena). |
| Caché | jsonl por prompt en disco, reanudable. |
| Presupuesto | ≤ US$15 de API. Piloto de 20 informes antes de lanzar todo (medido: US$0.0385 → proyección US$8.5). Corte automático a US$14. |

**Validación de L5, sin gold** (criterios fijos antes de ver la corrida completa; del piloto solo se miró el costo y un
acuerdo grueso sobre 20 informes, y los prompts no se tocaron después):

- Métricas: AUC de L5 (blanda) contra cada referencia binarizada > 0.5, por hallazgo y macro de 12; κ de Cohen con L5
  binarizada > 0.5. Referencias: L2, L3, L4, Raptor y el **consenso de las 4** de exp031 (media por celda, > 0.5).
  Se calcula en los estudios no-gold de train.
- L5 **sana** si: cobertura ≥ 99 % de los 4407 estudios con las 12 celdas; macro AUC ≥ 0.90 contra cada una de L2, L3, L4
  y Raptor; macro AUC ≥ 0.93 contra el consenso de las 4; ningún hallazgo con AUC < 0.80 contra el consenso de las 4;
  costo ≤ US$15. (Referencia de exp011: la lectura única de gpt-5.4-mini, L7, tuvo macro AUC 0.907 vs L2 y 0.881 vs L3.)
- Se reporta, no decide: acuerdo entre los 3 prompts (κ), tasa de positivos, media, y macro AUC contra gold (58).
- Si L5 no está sana: se archiva como `didnt` de la fase 1 y la fase 2 no corre.

### Fase 2 — exp054 (entrenamiento)

**Hipótesis:** añadir L5 como 5.ª fuente al promedio por celda sube la pseudo AUC L2 en fold 0 en ≥ +0.003 contra exp031
sin que gold caiga más de 0.02.

**Única variable:** etiquetas de ENTRENAMIENTO = promedio por celda de 5 fuentes (padre: 4).

| Fuente | Archivo | Estado |
|---|---|---|
| Raptor | `labels/public/dreaddevelopment/rsna-knee-labels/labels_llm_soft.parquet` | como el padre |
| L2 | `labels/public/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` | como el padre |
| L3 | `labels/public/pilkwang/rsna-knee-llm-labels/report_labels_v2.csv` | como el padre |
| L4 | `labels/public/lixin73/rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv` | como el padre |
| **L5** | `labels/ours/L5_20261011.csv` | **nueva** |

Mismo manejo que el padre: targets blandos tal cual, media de las fuentes presentes por celda, NaN enmascarado,
`pos_weight = clip((1−prev)/prev, 1, 10)` recalculado con la misma regla (consecuencia de la variable).

**Todo lo demás idéntico al padre:** corpus 96×384 en `/root/corpus96_384`, recorte central 320, CoAtNet-2 384 +
`RaptorClassifier`, 16 épocas, bs 8 × k 12 (k_eval 24), AdamW / OneCycle / LRs, bf16, semilla 42, `split_v1_seed42`
fold 0, selección por pseudo L2 (L2 binarizada > 0.5) en val fold 0, SWA top-3 por pseudo L2. Gold y v1 solo se reportan.

**Métrica principal:** pseudo L2 fold 0 de la mejor época. SWA top-3 se reporta.

**Criterios (fijos, no cambian después de correr):** padre exp031 = pseudo L2 0.8836, gold 0.9144
(`results/exp031_raptor_etiquetas_consenso/metrics.json`); ruido de semilla medido 0.0013 (sub031_2).
- `worked`: pseudo L2 fold 0 ≥ 0.8866 (+0.003) **y** gold ≥ 0.8944.
- `killed`: pseudo L2 fold 0 ≤ 0.8786 (−0.005) **o** gold < 0.8944.
- `didnt`: cualquier otro caso.
La `decision_auto` del notebook usa el umbral estándar ±0.005; el verdict se aplica a mano con estos criterios.
Si promete: run all-data (`-p ALL_DATA True -y 'SWA_EPOCHS: [a, b, c]'`, 3 mejores épocas por pseudo L2 del fold 0).

**Sesgos declarados:**
- La vara pseudo L2 está sesgada hacia etiquetas que contienen L2. Añadir L5 **diluye** L2+L4 (de 2/4 a 2/5), así que
  el sesgo juega en contra: una bajada leve de pseudo no implica peores etiquetas. El LB es el árbitro final.
- L5 usa las definiciones oficiales estrictas (derrame moderado/grande, MCL de alto grado, OA > 50 % de grosor): se
  espera más estricta (media más baja) que L2/L3, como Raptor y JEV. Baja la escala del objetivo y sube `pos_weight`.
- Gold no tocó L5 (ni prompts ni mapeos se ajustaron mirando gold).

**Presupuesto:** fase 1 ≤ US$15 de API, 0 GPU-h. Fase 2: fold 0 ≈ 2.5 GPU-h en el primer pod libre que asigne el
orquestador (RTX PRO 4500 32 GB, corpus en `/root/corpus96_384`).

**Riesgos:** informes en 10 idiomas (la lectura en griego/búlgaro/turco es la más frágil); la paráfrasis puede perder
matices de severidad; el mapeo de A a probabilidades es una elección fija (no afinada).

## 2. PLAN

- Fase 1: `generar_L5.py` (piloto `--n 20 --sample`, luego todo; `--build` escribe el CSV) y validación (AUC/κ contra
  L2/L3/L4/Raptor/consenso; gold solo reporte). CSV a `/workspace/data/labels/ours/`. Números en `BITACORA.md` y
  en `L5_validacion.json` de esta carpeta (solo números agregados; `results/` solo admite `metrics.json`).
  `validar_L5.py` hace la validación.
- Fase 2: notebook `exp054_etiquetas_L5.ipynb` = copia del de exp031; cambios: celda 0 (`EXP`, `PADRE`, `L5_CSV`,
  `NOTES`), celda 2 (`SOURCES` + L5; logs de nº de fuentes y acuerdo con L5). Mismo diff que sub031_1 con JEV.
- Smoke: `SMOKE=True` (64 train / 48 val, 1 época) → `metrics_smoke.json`. Run real: `FOLDS=[0]`, `SMOKE=False`.

## 3. VERDICT (números de `results/exp054_etiquetas_L5/metrics.json`)

**Resultado:** `worked` · **Fase alcanzada:** screen fold 0 (un fold; 5 folds no corridos); all-data pendiente de pod

### Fase 1 — L5 (números de `L5_validacion.json`)

L5 **sana** (pasa los 4 chequeos + costo). Costo real US$8.78 (A 2.71 · B 2.85 · C 3.22; 13.8 M tokens in, 2.0 M out;
0 errores, 0 reintentos fallidos). Cobertura 4407/4407 con las 12 celdas. Pool no-gold = 4349 estudios.

| Referencia (binarizada > 0.5) | macro AUC de L5 | macro κ (L5 > 0.5) |
|---|---|---|
| L2 | 0.9925 | 0.661 |
| L3 | 0.9895 | 0.585 |
| L4 | 0.9917 | 0.672 |
| Raptor | 0.9567 | 0.733 |
| consenso de las 4 | 0.9920 (mín. por hallazgo: Contusion 0.975) | 0.664 |

- Por prompt, macro AUC vs consenso de las 4: A_extract 0.969 · B_reason 0.986 · C_paraphrase 0.983. κ entre prompts:
  A–B 0.659 · A–C 0.690 · B–C 0.871 (las 3 lecturas sí difieren; A es la más distinta).
- L5 es más estricta, como se esperaba con las definiciones oficiales: media 0.182 (consenso 0.301, Raptor 0.257);
  positivos > 0.5: MCL 4.4 %, Lateral OA 8.0 %, Synovitis 9.1 %, Effusion 25.8 %. Ordena casi igual que las demás
  (AUC ≈ 0.99) pero corta más bajo (κ 0.59–0.73); concuerda más con Raptor en el corte que con L2/L3.
- Gold (58, solo reporte, no seleccionó nada): L5 macro AUC 0.889 · consenso de las 4 0.895 · L2 0.893 · L3 0.870 ·
  L4 0.835 (Raptor no cubre gold). Más floja en Synovitis 0.775, Contusion 0.805, Lateral OA 0.819.

### Fase 2 — exp054 (fold 0, rsna-d2, 06:32–09:04 UTC)

| Métrica | Padre (exp031) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8836 (ep 15) | **0.8896** (ep 15) | **+0.0060** |
| pseudo L2 fold 0 (SWA top-3) | 0.8835 (ep 15, 13, 14) | 0.8895 (ep 15, 14, 13) | +0.0059 |
| v1 fold 0 | 0.8368 | 0.8424 | +0.0056 |
| gold (veto) | 0.9144 | 0.9190 (SWA 0.9177) | +0.0047 |
| GPU-h | 2.49 | 2.50 | +0.01 |

Criterio: pseudo 0.8896 ≥ 0.8866 y gold 0.9190 ≥ 0.8944 → **worked** (`decision_auto` = `promising`).
Δ +0.0060 = 4.6 × el ruido de semilla medido (0.0013).

**Qué aprendimos:** una fuente propia y barata (US$8.78, gpt-4.1-mini, 3 prompts) sube el consenso +0.006 en pseudo
L2 aunque **diluye** L2+L4 en el promedio (el sesgo de la vara jugaba en contra), y v1 (+0.006) y gold (+0.005)
suben en la misma dirección. Por hallazgo (pseudo, mejor época): Contusion +0.019, MCL +0.017, Medial Meniscus +0.012,
Lateral Meniscus +0.008, Medial OA +0.006, Lateral OA +0.005, ACL/PF OA/Effusion/Fracture +0.002–0.003;
Synovitis −0.003 y Baker's −0.003. L5 es más estricta (media 0.182) y baja la media del objetivo de 0.301 a 0.278.
La diversidad de prompts/modelo parece contar más que la calidad individual: L5 sola no es mejor que L2 en gold.

**Siguiente paso sugerido:** all-data con `SWA_EPOCHS: [15, 14, 13]` (lanzador listo en `/root/run054_alldata.sh`
de rsna-d2, que quedó ocupado por sub045_2) → pesos al ensamble; sumar L5 al combo exp056; probar L5 con más
lecturas (k = 5, otro modelo barato) como sub054_k.
