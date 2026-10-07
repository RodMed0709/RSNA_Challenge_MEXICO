# Novedad, autoría y protocolo de evaluación — paper RSNA Knee (eficiencia)

Fecha: 2026-10-06. Fuentes verificadas hoy (PubMed E-utilities, arXiv API, API de Kaggle, rsna.org). Lo que no pude
verificar va marcado **[no verificado]**.

---

## A) Autoría en los papers oficiales de RSNA, premio de eficiencia y fórmula

### A.1 Papers oficiales de resultados: los firman los organizadores, no los equipos

| Reto | Paper de resultados | Autores | ¿Equipos como coautores? | Qué analiza |
|---|---|---|---|---|
| Cervical spine 2022 | Lee GR et al., *Performance of the Winning Algorithms of the RSNA 2022 Cervical Spine Fracture Detection Challenge*, Radiol AI 2024;6(1):e230256. https://pubmed.ncbi.nlm.nih.gov/38169426/ | 9, todos organizadores (Flanders, Richards, Kitamura, Colak, Lin, Ball, Talbott, Prevedello) | **No** | Los **8 mejores** (= los premiados). 883 equipos / 1108 participantes |
| Cervical spine 2022 (seguimiento) | Hu Z et al., *Assessing the Performance of Models from the 2022 RSNA Cervical Spine Fracture Detection Competition at a Level I Trauma Center*, Radiol AI 2024;6(6):e230550. https://pubmed.ncbi.nlm.nih.gov/39298563/ | 27 | **Sí**: Ha Q, Hanley D, Seferbekov S, Chen H, Singer P, Henkel C, Pfeiffer P, Pan I, Sheoran H (autores de los **7** modelos top) | Validación externa clínica de 7 modelos premiados |
| Cervical spine 2022 (seguimiento) | Chen YA et al., AJR 2025 (7 modelos premiados, fracturas omitidas por radiólogos). https://pubmed.ncbi.nlm.nih.gov/39772578/ | 12, organizadores/Toronto | No | 7 modelos premiados |
| Cervical (seguimiento) | Harper JP et al., AJNR 2025 (validación externa de 1 modelo ganador). https://pubmed.ncbi.nlm.nih.gov/39993795/ | 6, incluye a Pan I (participante) | Parcial | 1 modelo |
| Abdominal trauma 2023 | Hermans S, Hu Z et al., *RSNA 2023 Abdominal Trauma AI Challenge: Review and Outcomes*, Radiol AI 2025;7(1):e240334. https://pubmed.ncbi.nlm.nih.gov/39503604/ | 18, todos organizadores | **No** | Los **8 modelos premiados** |
| Mammography 2023 | Chen Y et al., *Performance of Algorithms Submitted in the 2023 RSNA Screening Mammography Breast Cancer Detection AI Challenge*, **Radiology** 2025;316(2):e241447. https://pubmed.ncbi.nlm.nih.gov/40793948/ | 14, organizadores/lectores | **No** | **Todos** los 1537 algoritmos evaluables + foco en el **top 10** |
| Lumbar 2024 | Solo paper de dataset: Richards TJ et al., *The RSNA LumbarDISC Dataset*, Radiol AI 2026 (mar). https://pubmed.ncbi.nlm.nih.gov/41532835/ | 34 nominales + grupos colectivos (contribuidores, anotadores, "Competition Data Notebook and Curation Group") | No (los ganadores no aparecen) | No encontré paper de resultados de ganadores a 2026-10-06 (búsqueda PubMed 2025–2026 vacía). Ganadores (9 premiados): https://www.rsna.org/news/2024/november/2024-ai-challenge-winners |

Papers de dataset (cervical https://pubmed.ncbi.nlm.nih.gov/37795143/, RATIC https://pubmed.ncbi.nlm.nih.gov/39441109/,
mamografía https://pubmed.ncbi.nlm.nih.gov/41563075/): autores = organizadores, sitios contribuyentes y grupos colectivos de
anotadores. Ningún participante de Kaggle.

**Conclusión A.1.** El patrón de RSNA es: (i) paper de dataset firmado por organizadores y sitios; (ii) paper de
resultados firmado por organizadores que **analiza** los modelos premiados (top 8 en cervical y abdominal = todos los
premiados; todos + top 10 en mamografía) **sin** hacer coautores a los equipos; (iii) a veces, un paper de seguimiento
(validación externa) donde los autores de los modelos sí firman (Hu 2024: autores de los 7 modelos top). No hay
precedente de "top 5/top 10 como coautores" en el paper principal. El camino realista para nosotros es **un paper
propio** (precedente: Pan I, *Deep Learning for Pulmonary Embolism Detection: Tackling the RSNA 2020 AI Challenge*,
Radiol AI 2021, autor único participante, https://pubmed.ncbi.nlm.nih.gov/34617031/).

### A.2 Premios de eficiencia: primera vez en RSNA; sin precedente de publicación

- RSNA: "including **for the first time** awards for the most efficient models"; ganadores reconocidos en el AI Theater
  de RSNA 2026 (29-nov a 3-dic); modelos ganadores con licencia abierta. **No** se menciona plan de publicación.
  https://www.rsna.org/news/2026/august/ai-challenge-knee-mri · https://www.rsna.org/media/press/2026/2669 ·
  https://www.auntminnie.com/imaging-informatics/artificial-intelligence/news/15831808/rsna-launches-2026-knee-abnormality-detection-ai-challenge
- Premios (página Prizes, vía API Kaggle): principal 10 puestos ($9,000 → $5,000); eficiencia 3 puestos ($7,000, $6,000,
  $5,000). Obligaciones extra del host: vídeo, enlace a código y pesos en el foro, modelo final público. Licencia
  ganadora CC-BY-NC 4.0; datos bajo licencia MIRA (http://rsna.org/mira-license). Timeline: requisitos de ganadores
  hasta el 5-nov-2026.
- Otros Kaggle con Efficiency Prize y **la misma plantilla** de fórmula (verificado por API): Feedback Prize ELL (2022,
  MCRMSE), Predict Student Performance (2023, F1), Learning Agency Lab AES 2 (2024, Kappa). Todos educativos/NLP;
  no encontré ningún reto médico previo con efficiency track. Como RSNA nunca tuvo uno, **no hay precedente de si los
  ganadores de eficiencia entran al paper oficial**. Lo más probable por el patrón A.1 [inferencia]: el paper de
  resultados del reto de rodilla analizará el top 10 principal y quizá los 3 de eficiencia, firmado por organizadores.

### A.3 Fórmula exacta del Efficiency Score (transcrita literal de la página, API Kaggle, hoy)

> We compute a submission's **efficiency score** by:
> `\[ \text{Efficiency} = \frac{ \text{AUC} }{ \text{Benchmark} - \max\text{AUC} } + \frac{ \text{RuntimeSeconds} }{ 32400 } \]`
> where AUC is the submission's score on the main competition metric, Benchmark is the score of the benchmark
> `sample_submission.csv`, max AUC is the maximum AUC of all submissions on the Private Leaderboard, and
> RuntimeSeconds is the number of seconds it takes for the submission to be evaluated. The objective is to minimize
> the efficiency score.

Elegibilidad (misma página): debe ser una de las submissions seleccionadas para el leaderboard principal (o la
seleccionada automáticamente) y quedar en el private LB por encima de `sample_submission.csv`. LB de eficiencia
público (solo rangos): https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb
Página: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/efficiency-prize-evaluation

**Cómo leerla.** `sample_submission.csv` pone 0.5 en todo → Benchmark = 0.5 (AUC de una constante). Con max AUC ≈ 0.965,
el denominador es 0.5 − 0.965 = −0.465, así que el primer término es −AUC/0.465 (más AUC → más negativo → mejor) y el
segundo suma 1 punto por cada 9 h (32,400 s) de reloj de pared del rerun completo sobre el test oculto (~1300 estudios,
incluye leer y decodificar DICOM).

**Tipo de cambio.** ΔE = 0 ⇔ ΔAUC/0.465 = Δt/32400 ⇔ **Δt = 32400 · ΔAUC / (max AUC − 0.5)**.
- 0.01 AUC = 32400 · 0.01 / 0.465 = **697 s ≈ 11.6 min** (≈ 0.54 s por estudio sobre 1300 estudios).
- Sensibilidad a max AUC (desconocido hasta el cierre): 0.955 → 712 s; 0.965 → 697 s; 0.975 → 682 s.
- 0.001 AUC ≈ 70 s (coincide con RANKING.md).

**Ejemplo numérico** (max AUC = 0.965):
- Modelo A: AUC 0.950, 60 min (3600 s) → E = −0.950/0.465 + 3600/32400 = −2.0430 + 0.1111 = **−1.9319**.
- Modelo B: AUC 0.935, 10 min (600 s) → E = −2.0108 + 0.0185 = **−1.9922** → **B gana** (menor E).
- A necesitaría bajar a < 2,445 s (≈ 41 min) para empatar con B: 0.015 AUC de ventaja compran solo 1,045 s.

---

## B) Novedad de los ángulos candidatos

Búsquedas: PubMed (E-utilities) y arXiv API 2020–2026, más el corpus de `docs/papers` (RANKING.md).

### (1) Curva AUC–cómputo en 12 hallazgos con hardware real — **novedad media-alta** (en rodilla), media en general

Lo más cercano:
- Tsai, **ELNet** (MIDL 2020, https://arxiv.org/abs/2005.02706): único antecedente de rodilla centrado en eficiencia,
  pero solo cuenta parámetros, 3 tareas MRNet, sin latencia ni curva.
- Harvey et al., **MIL benchmark 3D neuroimaging** (arXiv 2026, https://arxiv.org/abs/2604.26807): compara
  media/ABMIL/3D CNN/ViT con foco en coste, pero de **entrenamiento** (25× más rápido), cerebro, tarea única.
- Qiu, **CoPAS** (Nat Commun 2024, https://pubmed.ncbi.nlm.nih.gov/39223149/), Vuskov (Eur Radiol 2026,
  https://pubmed.ncbi.nlm.nih.gov/41107495/), Xie (eClinicalMedicine 2025, https://pubmed.ncbi.nlm.nih.gov/41079026/):
  rodilla multi-hallazgo (12/23/9) **sin** FLOPs, latencia ni curva coste-AUC.
- Marco metodológico: Dehghani, *The Efficiency Misnomer* (ICLR 2022, https://arxiv.org/abs/2110.12894).

Búsqueda PubMed "knee MRI + FLOPs/inference time/lightweight" (17 hits): solo modelos "lightweight" de ACL/menisco de
tarea única (p. ej. Jeon JBHI 2021; MV2SwimNet PLoS One 2025), ninguno con frontera de Pareto multi-hallazgo ni con
runtime end-to-end incluyendo DICOM. Un revisor dirá "es medición, no método": por sí solo es débil para CMIG, fuerte
como **marco** del paper y como contribución secundaria. Ventaja única: la métrica oficial de eficiencia de Kaggle permite
anclarlo a un benchmark externo con runtime medido por terceros.

### (2) Etiquetas débiles de informes vía LLM, validadas con 58 gold — **novedad baja-media**

Lo más cercano:
- Park et al., *Automated Spinal MRI Labelling from Reports Using a LLM* (arXiv 2024, https://arxiv.org/abs/2410.17235):
  LLM open-source extrae 5 condiciones de informes de RM de columna y entrena clasificadores de imagen ≈ etiquetas
  manuales. Es casi exactamente el ángulo (2) en otra articulación.
- Al Mohamad et al., Acad Radiol 2025 (https://pubmed.ncbi.nlm.nih.gov/39765434/): Mixtral → etiquetas → CNN (tobillo,
  RX, AUC 0.926). Bigolin Lanfredi, **LEAVS** (arXiv 2025, https://arxiv.org/abs/2503.13330): LLM labeler para TC abdominal.
- Rodilla específicamente: **GreenRFM** (arXiv 2026, https://arxiv.org/abs/2603.06467) usa etiquetas de informes por
  consenso de LLM en una cohorte de rodilla (AH-Knee). Antecedentes pre-LLM: Wood et al. Eur Radiol 2022 (cabeza, RM,
  https://pubmed.ncbi.nlm.nih.gov/34286375/), Zaman Radiol AI 2022 (RM cardiaca).
- Además, en este reto todo el mundo lo hace (RANKING §3.3: repo RoroDev2023 ya usa soft labels LLM).

Lo que sí sería nuevo: **informes en 12 idiomas, 16+ sitios** y una cuantificación explícita del sesgo de evaluar con
pseudo-etiquetas (pseudo-AUC vs gold-AUC por hallazgo; cf. Chavoshi 2025, en `docs/papers/06`). Útil como parte del
método, no como titular.

### (3) "Label-query attention" que decide qué plano/serie mirar → poda de series en inferencia — **novedad media-alta**

Lo más cercano:
- **CoPAS** (Qiu 2024): atención co-plano entre secuencias para 12 hallazgos de rodilla; no hay poda ni presupuesto.
- **OrthoDiffusion** (arXiv 2026, https://arxiv.org/abs/2602.20752): módulo MPAE de fusión por plano específica de
  paciente y etiqueta; backbone pesado, sin análisis de coste.
- **AnyMC3D** (Liu 2025, arXiv; PDF en `docs/papers/03`) y **ML-Decoder**/CSRA: query por etiqueta/vista, sin poda.
- Selección de cortes con presupuesto: CardioLens (arXiv 2026, https://arxiv.org/abs/2606.00123) compara selección
  aleatoria/clínica/data-driven bajo presupuestos de cortes, pero para MLLMs y sin ganancias de coste medidas;
  DEFORMISE (https://arxiv.org/abs/2407.17324) selección de cortes en demencia. Token pruning genérico: EViT, ToMe.

No encontré ningún trabajo que use la atención por hallazgo para **podar series/cortes en inferencia con presupuesto
por hallazgo** y medir el ahorro real en RM multi-serie (búsquedas arXiv "early exit"+volumétrico y PubMed "sequence/series
selection"+clasificación: 0–2 hits no pertinentes). Riesgo: Harvey 2026 muestra que la atención de MIL suele ser poco
fiable y que la media empata → la ablación "media vs atención vs atención+poda" es obligatoria, y la novedad depende de
que la poda dé un punto de Pareto mejor que simplemente bajar resolución/cortes de forma uniforme.

### (4) Validación externa parcial en MRNet — **novedad baja** (pero valor alto para CMIG)

- Bien et al., MRNet (PLoS Med 2018, https://pubmed.ncbi.nlm.nih.gov/30481176/); Tran et al., ACL multi-continental
  external validation (Eur Radiol 2022, https://pubmed.ncbi.nlm.nih.gov/35726103/); Hung JMRI 2023
  (https://pubmed.ncbi.nlm.nih.gov/35648374/). MRNet como test externo es rutinario.
- Sirve como evidencia de generalización (lo pide cualquier revisor de journal), no como novedad. Ojo: etiquetas no
  mapean 1:1 (MRNet: anormal/ACL/menisco sin lado), solo 3 series (sag T2, cor T1, ax PD), licencia de investigación de
  Stanford (verificar compatibilidad con publicar resultados de un modelo entrenado con datos MIRA; no hay conflicto
  aparente si solo se **evalúa**). No usar cifras de MRNet que re-parten el train (RANKING §4).

### Combinaciones propuestas

**CMIG (journal, la más defendible):** (3) como método + (1) como marco de evaluación + (2) como componente de datos
+ (4) como validación externa.
> *Finding-conditioned series and slice budgeting for efficient multi-label knee MRI: accuracy–compute Pareto analysis
> on 12 findings with multilingual LLM-derived labels.*
Contribuciones: (a) cabeza label-query que produce una matriz hallazgo×serie y permite podar series/cortes por
hallazgo; (b) frontera AUC–(s/estudio, GFLOPs) en GPU T4 y CPU, end-to-end, contra baselines (media, CoPAS-like,
ensemble tipo notebook público, destilado); (c) etiquetas LLM multilingües con sesgo pseudo-vs-gold cuantificado;
(d) evaluación triple: OOF 5-fold, 58 gold, LB privado de Kaggle (runtime oficial) + MRNet.

**ISBI 2027 (4 páginas, la más vendible):** (3)+(1) solamente.
> *Where to look: label-query attention enables per-finding series pruning for efficient knee MRI screening.*
Una figura de Pareto (AUC vs s/estudio, con el punto de nuestra submission en el LB privado/eficiencia de Kaggle), una
tabla de ablación (media / atención / atención+poda / poda uniforme), un mapa hallazgo×serie interpretable
(p. ej. ACL→sagital, MCL→coronal). (2) y (4) en una frase/limitaciones. Nota: el deadline ISBI (26-oct) cae 4 días
después del cierre; el LB privado se publica al cierre, así que el número de Kaggle puede entrar.

---

## C) Protocolo de evaluación que exigirá un revisor (test de Kaggle sin etiquetas)

### C.1 Tres niveles de evidencia, cada uno con su papel declarado

| Nivel | n | Qué demuestra | Cómo reportarlo | Límite a declarar |
|---|---|---|---|---|
| OOF 5-fold, pseudo-etiquetas | ~4,349 | Comparaciones internas entre variantes (ablaciones, Pareto) con potencia | macro AUC + AUC por hallazgo, IC 95 % por bootstrap a nivel estudio (≥2,000 réplicas); media±DE entre folds | Sesgo por ruido de etiqueta dependiente de prevalencia (Chavoshi 2025); split estratificado por vendor/idioma **no agrupado por sitio** → posible optimismo por memorizar escáner. Añadir análisis de sensibilidad "leave-one-language/vendor-out" (como split_v2, no reemplaza a v1) |
| Gold experto | 58 | Validez frente a referencia experta; veto | AUC por hallazgo y macro con IC bootstrap; indicar nº de positivos por hallazgo; nunca usado para seleccionar (declararlo) | Muy poca potencia: con AUC 0.80, 15 pos/43 neg, SE ≈ 0.074 (Hanley–McNeil) → IC ±0.15 por hallazgo. Diferencias < ~0.05 en macro no son detectables. Hallazgos con < 5 positivos: reportar pero no interpretar |
| Kaggle LB público (~30 %, ~390) / privado (~70 %, ~910) | ~1,300 | Evidencia externa ciega, etiquetas del organizador [presumiblemente expertas, no verificado], **runtime medido por terceros** | Macro AUC público y privado, runtime oficial y Efficiency Score/rango; nº total de submissions hechas (riesgo de sobreajuste al público) | Un solo número sin IC ni test pareado. Tras el cierre, las **late submissions** dan score privado: usarlas para 3–5 puntos clave del Pareto (5/día) |
| MRNet (opcional, CMIG) | 120 valid. | Generalización a otro sitio/protocolo | AUC ACL/menisco/anormal con IC bootstrap | Mapeo de etiquetas imperfecto |

### C.2 Pruebas estadísticas

- **Endpoint primario único**: macro AUC (métrica oficial). Comparación primaria pre-especificada, p. ej. "estudiante
  podado vs referencia completa". Para un paper de eficiencia, la forma natural es **no inferioridad**: margen
  pre-registrado (p. ej. ΔmacroAUC ≥ −0.01) con IC bootstrap pareado unilateral 95 %, junto al ahorro en s/estudio.
- **Bootstrap pareado** a nivel estudio (mismas réplicas para ambos modelos) para ΔmacroAUC: DeLong no aplica
  directamente a un promedio de 12 AUC.
- **DeLong** (DeLong 1988, Biometrics) por hallazgo para pares de modelos correlacionados; secundario.
- **Corrección múltiple**: 12 hallazgos → Holm–Bonferroni (FWER) o Benjamini–Hochberg (FDR) en los secundarios; el
  primario (macro) no se corrige. Si se comparan k variantes de Pareto, declarar cuáles son confirmatorias y cuáles
  exploratorias.
- Variabilidad de entrenamiento: ≥3 semillas para el modelo final y la referencia (media ± DE) o al menos la DE entre
  folds; los revisores de CMIG preguntan por esto.

### C.3 Medición de eficiencia reproducible (Dehghani 2022; Ozdemir 2026, CHIL, en `docs/papers/05`)

Reportar **varias** métricas, porque params, FLOPs y latencia no correlacionan:
1. **Runtime end-to-end por estudio** (s/estudio) en el entorno de Kaggle: 1×T4, 2×T4 y CPU (4 vCPU), contenedor/imagen
   declarada, versión de PyTorch/CUDA, fp16 sí/no, batch, nº de workers. Incluye **lectura y decodificación DICOM**
   (es lo que mide Kaggle; con JPEG2000/JPEG lossless puede ser el 60–70 % del tiempo) **y además** el tiempo solo de
   modelo, con desglose I/O+decode / preprocesado / forward / post. Calentamiento (≥10 estudios descartados), mediana
   y p95 sobre ≥3 corridas completas; cronometrar con `torch.cuda.synchronize()`.
2. **Runtime oficial de Kaggle** del notebook de submission (suma total sobre ~1300 estudios) y Efficiency Score.
3. **GFLOPs por estudio** (suma sobre todos los cortes × series que realmente se procesan, después de la poda) con
   **fvcore** (https://github.com/facebookresearch/fvcore) y verificación cruzada con **thop**
   (https://github.com/Lyken17/pytorch-OpCounter). Ojo: ambos cuentan una multiplicación-suma como 1 → en realidad son
   GMACs; decir explícitamente "GMACs (FMA = 1)" y no mezclar con FLOPs = 2×MACs. Las capas de atención custom pueden
   no contarse: añadir conteo manual.
4. **Parámetros**: totales y activos en inferencia (tras fusionar LoRA/eliminar cabezas auxiliares).
5. **Memoria pico**: `torch.cuda.max_memory_allocated()` (GPU) y RSS pico (CPU).
6. Opcional: energía por estudio (NVML / CodeCarbon).
7. Figura de Pareto: macro AUC (OOF y privado Kaggle) vs s/estudio (T4) y vs GMACs/estudio, con líneas de igual
   Efficiency Score (pendiente = 0.465/32400 por segundo total).

### C.4 Calibración

- ECE (15 bins, Guo et al. 2017, https://arxiv.org/abs/1706.04599) + Brier + diagramas de fiabilidad por hallazgo,
  sobre OOF (n grande); en gold solo descriptivo (58 es insuficiente para ECE estable).
- Si se calibra (temperature scaling por hallazgo), ajustar en OOF, **nunca** en gold. La métrica oficial (AUC) es
  invariante a calibración monotónica: decirlo, y presentar la calibración como requisito clínico, no como mejora de AUC.
- Para INT8/fp16 o poda: reportar ΔECE además de ΔAUC (Ozdemir 2026).

### C.5 Checklists aplicables

- **CLAIM 2024** (Tejani et al., Radiol AI 2024, https://pubs.rsna.org/doi/10.1148/ryai.240300): obligatorio en la
  práctica para CMIG/Radiol AI. Puntos críticos aquí: fuente y licencia de datos (MIRA), criterios de inclusión, cómo se
  derivan las etiquetas de referencia (LLM sobre informes; gold experto: quién, cuántos lectores, acuerdo), separación
  train/val/test y prevención de fuga (grupos de paciente, gold fuera), preprocesado, detalle del modelo, métricas con
  IC, análisis por subgrupo (vendor, campo, idioma), test externo, disponibilidad de código/pesos.
- **TRIPOD+AI** (Collins et al., BMJ 2024;385:e078378, https://www.bmj.com/content/385/bmj-2023-078378): aplicable
  como estudio de desarrollo + validación de modelo diagnóstico; cubre tamaño muestral, manejo de missing (lateralidad,
  series ausentes), calibración y equidad.
- Para ISBI basta con declarar la adherencia en una línea; para CMIG, adjuntar la checklist CLAIM completa como
  material suplementario.

### C.6 Riesgos que un revisor señalará (tenerlos respondidos)

1. Las 58 gold son demasiado pocas para cualquier afirmación por hallazgo → el peso confirmatorio cae en el LB privado
   de Kaggle (sin IC) y en OOF (etiquetas ruidosas). Declararlo y triangular.
2. Split no agrupado por sitio → sensibilidad leave-language/vendor-out.
3. Optimismo por selección sobre OOF de muchas variantes → separar exploratorio/confirmatorio; pre-registrar la
   comparación primaria antes de ver el LB privado.
4. Licencia MIRA: no mostrar imágenes del dataset sin confirmarlo; no liberar pseudo-etiquetas sin permiso; no compartir
   código fuera de Kaggle antes del 22-oct (arXiv/ISBI después del cierre).
