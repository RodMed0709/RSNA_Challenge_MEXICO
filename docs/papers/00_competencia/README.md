# 00 — Inteligencia de la competencia (RSNA Knee Abnormality Detection)

Fuente principal: API de Kaggle (`/api/v1/competitions/rsna-knee-abnormality-detection/pages`, leaderboard y
foro), consultada el **2026-09-26**. Todo lo citado entre comillas es textual de esas páginas.
Competencia: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

## 1. Regla de eficiencia exacta (textual)

Página *Efficiency Prize Evaluation*
(https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation):

> "We compute a submission's **efficiency score** by:
> Efficiency = AUC / (Benchmark − max AUC) + RuntimeSeconds / 32400
>
> where AUC is the submission's score on the main competition metric, Benchmark is the score of the benchmark
> `sample_submission.csv`, max AUC is the maximum AUC of all submissions on the Private Leaderboard, and
> RuntimeSeconds is the number of seconds it takes for the submission to be evaluated. **The objective is to
> minimize the efficiency score.**"

Elegibilidad (textual):

> "To be eligible for an Efficiency Prize, a submission:
> - Must be among the submissions selected by a team for the Leaderboard Prize, or else among those submissions
>   automatically selected under the conditions described in the My Submissions tab.
> - Must be ranked on the Private Leaderboard higher than the `sample_submission.csv` benchmark.
>
> All submissions meeting these conditions will be considered for the Efficiency Prize. A submission may be
> eligible for both the Leaderboard Prize and the Efficiency Prize."

> "During the training period … you may see a leaderboard for the *public* test data in the following notebook,
> updated daily: Efficiency Leaderboard (https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)
> … During the training period, this leaderboard will show only the rank of each team, but not the complete score."
> "More details may be posted via discussion forum updates."

Code Requirements (textual): "CPU Notebook <= 9 hours run-time · GPU Notebook <= 9 hours run-time · Internet
access disabled · Freely & publicly available external data is allowed, including pre-trained models ·
Submission file must be named `submission.csv`".

### Qué significa la fórmula en números
- `Benchmark` = AUC de `sample_submission.csv` (todo 0.5) → **0.5** (se deduce; la página no da el número).
- `max AUC` ≈ 0.96 (público hoy 0.959; el privado será otro). Denominador ≈ −0.46, así que
  Efficiency ≈ −AUC/0.46 + t/32400.
- **Tipo de cambio: 0.001 de AUC ≈ 70 s de runtime** (32400·0.001/0.459 = 70.6 s).
  1 min ≈ 0.00085 AUC · 5 min ≈ 0.004 · 30 min ≈ 0.025 · 1 h ≈ 0.051 · 9 h ≈ 0.46.
- `RuntimeSeconds` = "seconds it takes for the submission to be evaluated". Según el #1 de eficiencia
  (Scott Willis, foro 740474): "that's the time from hitting submit to getting a score". Es decir, **reloj de pared
  del rerun oculto sobre ~1300 estudios**, con arranque del contenedor incluido. No verificado por el host.

### Hardware — punto crítico
- **La regla NO restringe a CPU.** En 8 de 9 competencias previas con Efficiency Prize que revisé (ver §3), la
  página decía literalmente "Must not have a GPU enabled. **The Efficiency Prize is CPU Only.**" Esa frase **no
  aparece** en RSNA Knee. Conclusión: aquí cuenta cualquier notebook (CPU o GPU T4×2) y se mide solo el tiempo.
  El host no lo ha aclarado explícitamente en el foro (no encontré post del host sobre esto); riesgo bajo pero real.
- Entorno GPU de Kaggle usado por la comunidad: 2×T4 de 16 GB (no 32 GB unificados); P100 en retiro (foro 739000).
  Specs de CPU/RAM de los VMs de Kaggle no están en las páginas de la competencia (no verificado aquí).

### Datos operativos
- Test ≈ **1300 estudios**; "Series typically contain 20–45 slices (median 30)"; transfer syntaxes mixtos
  (Explicit/Implicit VR LE, JPEG Lossless, JPEG 2000). Public LB = **30 %** del test (`leaderboardPercentage: 30`),
  privado = 70 %.
- 5 envíos/día, 2 finales. Sin informes en test.

### Premios y fechas
- Total 77 000 USD. Main LB: 9000 / 7000 / 6500 / 6000 / 5500 / 5000×5.
  **Efficiency: 1.º 7000, 2.º 6000, 3.º 5000.** Se puede ganar en ambos tracks.
- Obligaciones extra del host: vídeo corto, código y pesos abiertos en el foro, modelo final publicado
  (ej. modelo en Kaggle Models). Licencia ganadora **CC-BY-NC 4.0**.
- Fechas: inicio 30-jul-2026 · entry y merge deadline **15-oct-2026** · envío final **22-oct-2026** ·
  obligaciones de ganadores 5-nov-2026 (23:59 UTC).
- Nota: los enlaces internos de la página de eficiencia apuntan a `rsna-knee-abnormalities-detection` (errata del
  host; el slug real es `rsna-knee-abnormality-detection`).

## 2. Implicaciones de diseño

1. **El track de eficiencia está dominado por AUC, no por tamaño.** Top-10 de eficiencia hoy: AUC pública
   mínima 0.943, mediana 0.95. El #1 de eficiencia (0.958, ~5–6 min) es también co-#1 del LB principal. Con un
   suelo de runtime de ~3–5 min (leer DICOM), bajar de 5 a 2 min solo "compra" ~0.0025 AUC. **Un modelo con
   AUC < ~0.945 no gana eficiencia aunque corra en 1 min.** Nuestra referencia (gold 0.78) está lejísimos.
2. **GPU permitida y el runtime es de pared** → usar la T4 (fp16, un modelo por GPU, sin DataParallel) es casi
   seguro más eficiente que CPU. CPU-only solo tiene sentido si el modelo es diminuto y la lectura DICOM domina.
3. **El cuello de botella es I/O + decodificación DICOM (60–70 % del tiempo del #1).** Receta pública
   (Prateek Grover, foro 743374, de ~40 a <10 min): leer cada DICOM una sola vez con 16 hilos (59 → 372 MB/s),
   parsear cabeceras sin pydicom (0.307 → 0.058 ms/archivo), `np.frombuffer` sobre PixelData (1.22 → 0.33 ms/corte),
   percentiles sobre submuestra, fp16 puro (2.08× vs autocast en T4), uint8 a GPU y normalizar en device.
   Leer menos cortes/series (subsampling) es la otra palanca.
4. **Presupuesto objetivo: 1 modelo, 1 fold, backbone pequeño (ResNet/EfficientNet pequeño), 224–288 px,
   runtime ≤ 10 min, AUC pública ≥ 0.95.** El #1 lo hace con "a smaller resnet", single fold, 0.949 → 0.958.
   Varios reportan que resolución >288 y backbones grandes no ayudan (foro 735304).
5. **La palanca de AUC son las etiquetas, no la arquitectura.** Los fuertes reportan: LLM local para extraer
   etiquetas (Gemma 4, ~0.89 vs gold), soft labels (no redondear), y mezclar etiqueta del informe con predicción
   OOF del modelo (teacher → pseudo-labels → student). Destilar un ensemble grande en un modelo pequeño es
   exactamente cómo se ganó el precedente más parecido (Feedback Prize Effectiveness, §3).
6. Detalles de datos que afectan velocidad y AUC: InstanceNumber va contra la posición en >1/3 de series (ordenar
   por posición); lateralidad falta en ~50 %; 3–14 series por estudio (máscara de series faltantes obligatoria);
   `Fluid_Sensitive` == `Fat_Suppression` en las 24 371 series de train (foro 737312, 741310).
7. **Robustez del rerun**: un fallo en el rerun oculto = sin puntuación (foro 742122). Envolver cada estudio en
   try/except con fallback; nunca exceder memoria de T4.

## 3. Antecedentes de efficiency prizes en Kaggle

Verificado leyendo la página *Efficiency Prize Evaluation* de cada competencia vía API:

| Competencia (año) | Fórmula | ¿CPU only? |
|---|---|---|
| Feedback Prize – Predicting Effective Arguments (2022) | LogLoss/(log3 − min LogLoss) + t/32400 | **No** (GPU permitida) |
| Feedback Prize – English Language Learning (2022) | MCRMSE/(Base − min) + t/32400 | Sí |
| Learning Equality – Curriculum Recommendations (2023) | F2/(Benchmark − max F2) + t/32400 | Sí |
| Predict Student Performance from Game Play (2023) | F1/(Benchmark − max F1) + t/32400 | Sí (VM 2 CPU, 8 GB, sin GPU) |
| CommonLit – Evaluate Student Summaries (2023) | MCRMSE/(Base − min) + t/32400 | Sí |
| Linking Writing Processes (2023) | RMSE/(Base − min) + t/32400 | Sí |
| PII Detection (2024) | F5/(Benchmark − max F5) + t/32400 | Sí |
| Learning Agency Lab – AES 2.0 (2024) | Kappa/(Base − max) + t/32400 | Sí |
| Eedi – Mining Misconceptions (2024) | MAP@25/(Base − max) + t/32400 | Sí |
| **RSNA Knee (2026)** | AUC/(Benchmark − max AUC) + t/32400 | **No** |

- **Ningún challenge RSNA previo tuvo efficiency track**: RSNA 2023 Abdominal Trauma, RSNA 2024 Lumbar Spine y
  RSNA 2025 Intracranial Aneurysm no tienen página de eficiencia (verificado vía API). RSNA Knee es el primero.
- **Precedente más parecido (GPU permitida): Feedback Prize Effectiveness, 1.º eficiencia, Team Hydrogen**
  (https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347537): "a single deberta-v3-large
  that got a 0.557 score on the Private LB in **5 minutes and 40 seconds** … this model alone would get a top-3
  position on the Private LB". Método: pseudo-etiquetas soft del ensemble grande (4 rondas) → entrenar **un solo
  modelo sin etiquetas originales** = destilación; perdió solo ~0.003 vs su mejor ensemble. Optimización de
  inferencia mínima (orden por longitud + padding dinámico). "a single base model runs in less than two minutes".
- CPU-only (LLM – Detect AI Generated Text, 1.º eficiencia, https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/471898):
  TF-IDF + NB/SGD/LGBM; la clave fue curar datos, no el modelo.
- Patrón general: en estas fórmulas los ganadores de eficiencia están cerca del top del LB principal y corren en
  minutos; ganan por **destilación / modelo único fuerte**, no por modelos débiles ultrarrápidos.
- No encontré writeups en PDF/arXiv de ganadores de efficiency prizes; son posts de Kaggle (enlaces arriba).

## 4. Leaderboard y discusiones (snapshot 2026-09-26)

**LB público principal** (4386 equipos): #1 0.959 (rhoskeri, Wassim Dobbi) · #3 0.958 · #10 0.956 · #20 0.955 ·
#50 0.952 · #100 0.946 · #200–#500 0.943 · #1000 0.941 · #2000 0.927. 1306 equipos ≥ 0.94 (≈200 forks
exactos en 0.941 del notebook "Bend the Knee to the Dinosaurs", foro 742050).

**LB de eficiencia** (notebook de ryanholbrook, 4258 equipos, solo rango + AUC pública, sin runtime):

| Rank eff. | Equipo | AUC pública | Nota |
|---|---|---|---|
| 1 | Scott Willis | 0.958 | ~5–6 min, ResNet pequeño, 1 fold (foro) |
| 2 | Avinash Niroula | 0.948 | |
| 3 | Brandon Low | 0.954 | |
| 4 | tomoon33 | 0.954 | |
| 5 | UCAS_fish_wang | 0.951 | |
| 8 | TomKroumov | 0.956 | (#5 main) |
| 234 | rhoskeri | 0.958 | #1 main, lento |

Cotas que deduje de los rangos (con ±0.0005 de redondeo): rhoskeri corre ≥ ~73 min más que el más rápido de su
cadena; TomKroumov ≥ ~14 min; takoi & charmq (0.956, eff #111) ≥ ~46 min. Top-10 eficiencia: AUC mín 0.943,
top-25 mín 0.936, top-100 mín 0.916.

**Discusiones útiles**
- 740474 "9h runtime question": runtime solo cuenta inferencia; Scott Willis "My best is around 5 minutes";
  lectura = "60 or 70% of my submission time". Otro equipo bajó de 9 h a 23 min.
- 735304 "Best single-model score": Scott Willis 0.949 LB single fold/single model, ~0.930 en gold, "around 5
  minutes"; CoAtNet 224 single fold 0.950; Qwen 3.5-2B+LoRA 0.950 (~3.5 h); ResNet pequeño 224 single fold 0.936;
  Chris Deotte: con pseudo-labels "every single model can match our best ensemble CV". Consenso: labels > arquitectura;
  224/288 px es el punto dulce.
- 743374: receta de lectura rápida (ver §2.3).
- 733932: LLM vs regex para etiquetas 0.878 vs 0.814 en gold; 25.4 % de celdas "no mencionado".
- 741310: Synovitis silenciada en ~50 % de positivos expertos; Fracture 7 % en informes vs 31 % en gold.
- 733965 (host): LLM comerciales permitidos para extraer etiquetas. 734109 (host): KneeCoT **no** permitido;
  datasets con licencia ND (MRNet, KneeMRI) probablemente problemáticos para pesos publicados (743416, 737950, sin
  resolución final del host a la fecha).
- 742327: riesgo de shake-up; ranking muy apiñado (0.001 ≈ 7–9 puestos).

## No verificado / abierto
- Que `Benchmark` = 0.5 exactamente (lógico para AUC con todo 0.5, pero no dicho).
- Si `RuntimeSeconds` incluye cola/arranque del contenedor (solo testimonio de un participante).
- Si el host aceptará GPU en el track (la regla no lo prohíbe; sin aclaración explícita).
- Specs exactas de CPU/RAM de los VMs de scoring.
- Runtime real de cualquier equipo salvo auto-reportes del foro.
