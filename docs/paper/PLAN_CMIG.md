# Plan del paper largo — Computerized Medical Imaging and Graphics (CMIG)

Fecha: 2026-10-06. Autor del plan: Claude (orquestador). Basado en `BRIEF.md`.
Leyenda: **[V]** = verificado en la web hoy (URL citada) · **[NV]** = no verificado / conocimiento previo, comprobar antes de usarlo.

---

## 0. Ficha de CMIG (verificada)

| Punto | Qué dice | Fuente |
|---|---|---|
| Scope | Incluye ML, foundation models, CAD, MRI de forma explícita; pide "algorithmic advances" y aplicaciones con orientación clínica. Un trabajo de eficiencia + etiquetas débiles en RM de rodilla entra en scope. **[V]** | [Guide for Authors](https://www.sciencedirect.com/journal/computerized-medical-imaging-and-graphics/publish/guide-for-authors) |
| Revisión | Revisión con anonimato simple (*single anonymized*). Primero filtra el editor (desk) y después va a **≥1 revisor**. **[V]** | Guide for Authors |
| Tipos de artículo / límite de palabras / límite de figuras | **La guía actual no fija límite de palabras, de figuras ni de tablas, y tampoco enumera tipos de artículo.** Sí recomienda usar las tablas "con moderación" y sin duplicar. Por lo publicado, la longitud habitual es de 8–15 págs. a doble columna: apuntar a ~7–9k palabras, 8–10 figuras y 5–7 tablas, y mandar el resto a material suplementario. **[V] la ausencia del límite / [NV] la longitud típica** | Guide for Authors |
| Abstract | ≤250 palabras, autocontenido y sin referencias. **[V]** | Guide for Authors |
| Keywords | De 1 a 7. **[V]** | Guide for Authors |
| Highlights | Recomendados (*encouraged*), no obligatorios: 3–5 viñetas de ≤85 caracteres con espacios, en un archivo aparte con "highlights" en el nombre. **[V]** | Guide for Authors |
| Graphical abstract | **No aparece en la guía de CMIG**, así que se trata como opcional. Recomendación: hacerlo igual, reutilizando la Fig. 1. **[V] ausencia** | Guide for Authors |
| Datos | Rige la **opción C** de datos de investigación: depositar los datos en un repositorio y citarlos, **o** explicar por qué no se pueden compartir. Se recomienda incluir un *data statement*. En nuestro caso: los datos de RSNA/MIRA no se pueden redistribuir → se explica y se enlaza a Kaggle. El código y los pesos van a GitHub + Zenodo con DOI (el código cuenta como "research data"). **[V]** | Guide for Authors |
| Preprints | Se permiten. Un preprint no cuenta como publicación previa. **[V]** | Guide for Authors |
| Extensión de un paper de congreso | La guía solo dice que el trabajo "no se ha publicado previamente salvo como preprint, abstract, conferencia publicada, tesis…" y remite a la política de Elsevier sobre publicación múltiple/redundante. **No hay un porcentaje explícito de contenido nuevo.** La práctica habitual es ≥30–50 % de contenido nuevo, citar el paper ISBI y declararlo en la cover letter. **[V] política general / [NV] umbral** | Guide for Authors; [Elsevier – prior publication](https://www.elsevier.support/elsevier/answer/which-uses-of-a-work-does-elsevier-view-as-a-form-of-prior-publication); [Elsevier ethics](https://www.elsevier.com/ethicalguidelines) |
| Uso de IA generativa | Obligatorio declararla en una sección antes de las referencias. No se permite IA para crear o alterar imágenes de pacientes. **[V]** | Guide for Authors |
| Otros obligatorios | Contribuciones en formato CRediT, declaración de conflictos de interés y financiación, análisis por sexo y género (SAGER) o declararlo como limitación, referencias en estilo autor-año. Word a una columna o LaTeX. **[V]** | Guide for Authors |
| Coste | **Suscripción: gratis.** Open access (gold): **APC de 2,920 USD** sin impuestos, que puede bajar según país o institución. Embargo de 12 meses para el autoarchivo del manuscrito aceptado. **[V]** | [Open access info](https://www.sciencedirect.com/journal/computerized-medical-imaging-and-graphics/publish/open-access-options) |
| Métricas | **IF 5.5, CiteScore 8.8** (el brief decía 4.9; está desactualizado). **[V]** | [Insights](https://www.sciencedirect.com/journal/computerized-medical-imaging-and-graphics/about/insights) |
| Tiempos | 3 días hasta la primera decisión (desk) · **70 días hasta la decisión tras revisión** · **145 días hasta la aceptación** · 10 días de aceptación a publicación online. **[V]** | Insights |
| Tasa de aceptación | **No se publica.** No aparece en Elsevier ni en agregadores (peeref, journalmetrics). No inventar una cifra. | [peeref](https://www.peeref.com/journals/2001) |

**Recomendación:** publicar por suscripción, que es gratis. El preprint en arXiv cubre la visibilidad.

---

## 1. La vara: qué incluyen los papers recientes de CMIG en este nicho

La búsqueda en ScienceDirect, filtrada a CMIG 2023–2026 (“knee MRI”, “MRNet knee”, “knowledge distillation lightweight”), da pocos papers de clasificación en RM de rodilla. Eso es bueno: hay hueco. Los relevantes:

| # | Paper (CMIG) | Qué experimentos y ablations incluye |
|---|---|---|
| P1 | Berrimi, Hans, Jennane, *A semi-supervised multiview-MRI network for the detection of Knee Osteoarthritis*, jun-2024. [link](https://www.sciencedirect.com/science/article/pii/S089561112400048X) | Varias vistas de RM, semi-supervisado (con datos sin etiquetar), OAI con 4,297 sujetos, **ablation por componente**, **mapas Grad-CAM**, AUC 0.932. |
| P2 | *Economical hybrid novelty detection leveraging global aleatoric semantic uncertainty for enhanced MRI-based ACL tear diagnosis*, oct-2024. [link](https://www.sciencedirect.com/science/article/pii/S0895611124001010) | **Validación cruzada entre bases (KneeMRI ↔ MRNet)** más un hospital externo solo para validación, **tests de significación (p<0.05) contra el SOTA**, **energía y huella de carbono** (≥2.6× menos energía), modelos ligeros (YOLOv5-nano). Es el precedente más cercano a nuestro ángulo de eficiencia. |
| P3 | Hsu, Lian, Chang, *Self-supervised isotropic reconstruction for abnormality detection in anisotropic MRI*, jul-2026. [link](https://www.sciencedirect.com/science/article/pii/S0895611126000893) | 2,225 escaneos de OAI de 5 centros, **robustez por centro**, **perturbaciones sintéticas de ruido y contraste**, **transferencia a 2 protocolos más**, **lectura ciega de ortopedistas** (65–67 % de preferencia) y tareas posteriores. |
| P4 | Fox, Ciliberti, Recenti et al., *Impact of automated and manual segmentation errors on knee OA classification…*, may-2026. [link](https://www.sciencedirect.com/science/article/pii/S0895611126000704) | **Robustez ante errores simulados**, importancia de variables y comparación de varios clasificadores. |
| P5 | *Adaptive multi-teacher knowledge distillation framework with foundation models for medical image analysis* (MultiMedDistill), abr-2026. [link](https://www.sciencedirect.com/science/article/pii/S089561112600042X) | **Destilación de un ensemble de foundation models a un estudiante** (8.8M parámetros, compresión 18×), **6 datasets**, **ablation por componente** con Δ cuantificados (+3.2 % / +1.4 %). Es el precedente directo para nuestra fase B. |
| P6 (contexto, 2022) | *Deep convolutional feature details for better knee disorder diagnoses in MR images* (MRPyrNet), dic-2022. [link](https://www.sciencedirect.com/science/article/abs/pii/S0895611122001124) | Módulo que se acopla a baselines sobre **MRNet y KneeMRI**, con una campaña de evaluación extensa por hallazgo (ACL, menisco). |
| P7 (contexto) | *MedFusionNet: hybrid transformer multimodal (imagen + texto clínico)*, ago-2026. [link](https://www.sciencedirect.com/science/article/pii/S0895611126000984) | Fusión de imagen e informe, Grad-CAM y mapas de atención, entrenamiento en dos etapas. Muestra que CMIG acepta trabajos que usan texto clínico. |

**Requisitos mínimos que hay que cumplir (derivados de P1–P6):**
1. Ablation de **cada** componente propuesto, con Δ e intervalos.
2. Comparación con **≥3–5 baselines/SOTA reproducidos** con el mismo split. En nuestro caso: MRNet, ELNet o MRPyrNet, y los notebooks públicos tipo DINOv2-2.5D.
3. **Validación externa entre bases** (MRNet y KneeMRI son el estándar de facto: P2 y P6 los usan).
4. **Estadística**: IC por bootstrap, DeLong o test de permutación pareado.
5. **Interpretabilidad**: Grad-CAM o pesos de atención.
6. **Eficiencia con hardware real**: parámetros, FLOPs, latencia y, como plus, energía/CO2 como en P2.
7. **Robustez**: perturbaciones y series faltantes. Un estudio con lectores es un plus, no un requisito.

---

## 2. Tesis del journal y diferencia con ISBI

### Tesis (una frase)
> *Un único modelo pequeño, entrenado con etiquetas débiles extraídas por un LLM de informes radiológicos y con agregación por hallazgo (label-query attention), alcanza ≥95 % del AUC de un ensemble grande en 12 hallazgos de RM de rodilla multiserie, con 10–50× menos cómputo. Además, la atención por hallazgo permite podar series en inferencia con una curva AUC–costo controlable, medida en CPU y GPU reales y validada en datos externos.*

(Los números 95 % y 10–50× son **objetivos**, no resultados. Se reemplazan por los de `metrics.json`.)

### Qué aporta el ISBI (4 págs., 26-oct)
Arquitectura, curva AUC–cómputo en 12 hallazgos (fold 0 o 5 folds), destilación al estudiante, tiempo de ejecución en Kaggle y AUC sobre el gold de 58 estudios y la pseudo-validación.

### Qué añade el journal (objetivo ≥50 % de contenido nuevo, medido en figuras y tablas nuevas)
| Bloque nuevo | Por qué lo pedirá un revisor de CMIG |
|---|---|
| **N1. Validación externa** en MRNet (ACL, menisco, anormal), KneeMRI (ACL) y fastMRI+ (ACL, MCL, menisco, derrame, quiste, contusión o fractura), sin reentrenar y con *fine-tuning* ligero. | Requisito mínimo 3. Sin esto, un paper con un solo origen de datos y gold n=58 se rechaza. |
| **N2. Etiquetas del LLM frente al gold.** (a) Exactitud de la extracción sobre el informe: un radiólogo etiqueta 300 informes, solo el texto. (b) Ruido de la etiqueta del informe frente a la imagen: el gold de 58 más el gold ampliado. (c) Curva de AUC según la calidad de la etiqueta: Qwen-7B v1 frente a v2, reglas, LLM más grande y ensemble de LLMs. (d) Calibración y *label smoothing* por hallazgo. | Es la contribución metodológica más original y la que diferencia el paper de "otro clasificador de rodilla". |
| **N3. Ablations completas** (5 folds, OOF): agregación, entrada, número de series y cortes, resolución, recorte anatómico, alineación izquierda/derecha y destilación (logits, features o ambos). | Requisito mínimo 1. |
| **N4. Análisis por hallazgo**: AUC, sensibilidad a especificidad fija, PR-AUC y calibración (ECE) por cada una de las 12 etiquetas. Qué hallazgos sufren más con la poda y relación con la prevalencia. | Pone el foco clínico que CMIG exige ("clinically oriented"). |
| **N5. Estudio de costo**: latencia por estudio en **CPU** (x86 de Kaggle y laptop) frente a **GPU** (T4 y A100/4090), con desglose de I/O DICOM, preprocesado y red; FLOPs, parámetros, memoria pico, energía (CodeCarbon) y cuantización INT8/ONNX en CPU. | Precedente: P2. Es el núcleo de la historia de eficiencia. |
| **N6. Interpretabilidad**: pesos de atención sobre series, planos y cortes por hallazgo (¿mira el sagital para el ACL y el coronal para el MCL?), Grad-CAM en cortes y concordancia con las cajas de fastMRI+ (pointing game). | Requisito mínimo 5. fastMRI+ da cajas expertas y lo convierte en una métrica cuantitativa, no solo en figuras bonitas. |
| **N7. Robustez**: series faltantes (k de N), perturbaciones de intensidad, ruido y resolución, campo 1.5T frente a 3T si DICOM lo permite, y cambio de dominio en datos externos. | Precedente: P3. |
| **N8. Resultado en test oculto** (LB privado de Kaggle, ~910 estudios) con *late submissions*, si Kaggle las habilita. | Convierte "58 gold" en "≈1300 de test" y es el mejor argumento contra la crítica de n pequeño. |

---

## 3. Outline sección por sección

**Título tentativo:** *Report-supervised, compute-aware multi-series knee MRI classification: label-query attention enables series pruning with a controllable accuracy–cost trade-off*

0. **Abstract** (≤250 palabras), **Highlights** (5 viñetas de ≤85 caracteres) y graphical abstract (= Fig. 1 simplificada).
1. **Introduction** (~1,000 palabras). La carga de RM de rodilla y la escasez de etiquetas expertas. Los modelos SOTA en Kaggle son stacks de 6–8 h. Hueco: nadie mide la curva AUC–costo en multiserie con 12 hallazgos y etiquetas de informe. Cuatro contribuciones.
2. **Related work** (~900 palabras). 2.1 RM de rodilla con DL (MRNet, ELNet, MRPyrNet, P1, P2). 2.2 Etiquetas débiles a partir de informes (CheXpert labeler, LLM labelers). 2.3 Eficiencia en imagen médica (destilación, P5, early-exit, poda de entradas). 2.4 Agregación multivista y multiserie.
3. **Materials** (~700 palabras). 3.1 Dataset RSNA (estudios, series, distribución de secuencias, prevalencia por hallazgo; **Tab. 1**). 3.2 Extracción de etiquetas con LLM: prompt, versiones y calibración con el gold. 3.3 Split `split_v1_seed42` y política del gold (solo veta). 3.4 Datasets externos y mapeo de etiquetas (**Tab. 2**).
4. **Methods** (~1,800 palabras). 4.1 Preprocesado: recorte físico, 336 px, alineación y 6 *slots* (**Fig. 2**). 4.2 Codificador 2D por corte y atención jerárquica corte → serie. 4.3 Label-query attention. 4.4 Teacher ensemble y destilación. 4.5 Poda de series y cortes con presupuesto de cómputo (política y umbral). 4.6 Protocolo de medida de costo (hardware, *warm-up*, I/O incluido o excluido).
5. **Experimental setup** (~500 palabras). Métricas (macro AUC, por hallazgo, PR-AUC, ECE), estadística (bootstrap de 2,000 réplicas, DeLong, Holm), baselines e hiperparámetros (suplementario).
6. **Results** (~2,000 palabras). 6.1 Resultado principal frente a baselines (**Tab. 3**). 6.2 Curva de Pareto (**Fig. 3**). 6.3 Ablations (**Tab. 4**). 6.4 Etiquetas del LLM frente al gold (**Fig. 4**, **Tab. 5**). 6.5 Análisis por hallazgo (**Fig. 5**). 6.6 Validación externa (**Tab. 6**). 6.7 Costo CPU frente a GPU (**Fig. 6**, **Tab. 7**). 6.8 Interpretabilidad (**Fig. 7**, **Fig. 8**). 6.9 Robustez (**Fig. 9**).
7. **Discussion** (~1,000 palabras). Qué decisión movió la aguja, implicaciones para un despliegue en CPU en hospitales con pocos recursos (ángulo México y LMIC), limitaciones (gold n=58, un solo origen de entrenamiento, ruido del LLM, test etiquetado ¿por expertos? → verificar) y trabajo futuro.
8. **Conclusion** (~200 palabras).
9. Data and code availability · CRediT · Declaration of competing interest · Funding · Declaration of generative AI use · Acknowledgements (RSNA/MIRA, Kaggle).

### Figuras propuestas
| Fig | Mensaje (lo que el lector debe llevarse) |
|---|---|
| 1 | Pipeline completo: informe → LLM → etiquetas débiles; DICOM multiserie → *slots* → codificador → label-query → 12 salidas, con la rama de poda. |
| 2 | Preprocesado: recorte anatómico físico y alineación izquierda/derecha. Ejemplo visual: usar MRNet o fastMRI si MIRA no permite imágenes. |
| 3 | **Figura principal**: Pareto macro-AUC frente a s/estudio (CPU y GPU) y GFLOPs. Incluye nuestros modelos, el teacher, los baselines y los puntos del LB público, más el frente de poda. |
| 4 | Calidad de la etiqueta frente al AUC del modelo: exactitud del LLM por hallazgo (sobre informe) contra el AUC final por hallazgo; versiones de etiquetador. |
| 5 | Heatmap por hallazgo (12) × configuración (teacher, estudiante, podas): ΔAUC. Muestra qué hallazgos aguantan la poda. |
| 6 | Desglose de latencia (I/O, preprocesado, red) en CPU frente a GPU, FP32 frente a INT8. |
| 7 | Pesos de atención por hallazgo sobre tipo de serie y plano (matriz de 12 × 6 *slots*). |
| 8 | Grad-CAM y cajas de fastMRI+ en ejemplos externos, más el pointing-game score. |
| 9 | Robustez: AUC frente a número de series faltantes y frente a la intensidad de la perturbación. |
| S1–Sn | Curvas ROC por hallazgo, diagramas de calibración, curvas de entrenamiento y matrices de confusión externas. |

### Tablas propuestas
| Tab | Mensaje |
|---|---|
| 1 | Datos: estudios, series y prevalencia por hallazgo (train pseudo, gold y externos). |
| 2 | Mapeo de etiquetas de nuestras 12 a MRNet, KneeMRI, fastMRI+, SKM-TEA y OAI-MOAKS. |
| 3 | Principal: macro-AUC (pseudo OOF, gold con IC, LB privado), parámetros, GFLOPs y s/estudio en CPU y GPU contra baselines. |
| 4 | Ablations (cada fila cambia una variable contra la referencia). |
| 5 | LLM frente a gold: precisión, recall, F1 y kappa por hallazgo sobre el informe; acuerdo entre la etiqueta de informe y la de imagen. |
| 6 | Validación externa: AUC con IC, cero disparos (*zero-shot*) frente a *fine-tune*, contra el SOTA publicado en cada dataset. |
| 7 | Costo detallado por hardware: latencia, memoria pico, energía (Wh/estudio) y CO2. |

---

## 4. Experimentos priorizados

Supuestos de cómputo **[NV, calibrar con los logs de exp002–008]**: una GPU de clase A100/4090 en el pod; estudiante a 336 px sobre el cache v2 ≈ 1–1.5 GPU-h por fold; teacher grande ≈ 3–5 GPU-h por fold. "OOF" = 5 folds. Las ablations, tras el screen en fold 0 (regla SDD), solo pasan a OOF las finalistas.

**Leyenda de dependencia:** **K** = necesita Kaggle (notebook oficial, *late submission* o ambos) · **D** = necesita los datos de la competición (licencia MIRA vigente después del 22-oct → **verificar en las reglas**) · **E** = datos externos · **H** = horas de un experto humano.

### MUST (sin esto no se envía)
| # | Experimento | GPU-h | Dep. | Cuándo |
|---|---|---|---|---|
| M1 | Modelo final del estudiante y del teacher en 5 folds OOF, con IC por bootstrap | 30–40 | D | Fase B, antes del 22-oct (reutilizar ISBI) |
| M2 | Curva de Pareto con poda de series y cortes (≥6 presupuestos) en OOF | 5 (solo inferencia) | D | Antes del 26-oct |
| M3 | Medida de costo en CPU y GPU con protocolo fijo: T4 de Kaggle, CPU de Kaggle, laptop y A100/4090 | 3 + tiempo de CPU | **K** + D | Kaggle: antes del 22-oct; el resto, nov |
| M4 | Validación externa en **MRNet** (*zero-shot*; *fine-tune* con su split oficial) | 8 | E | Nov–dic |
| M5 | Validación externa en **KneeMRI** (ACL, una sola serie sagital PD-FS) | 3 | E | Nov–dic |
| M6 | Ablations núcleo (≥6): agregación (mean, attention o label-query), cache v1 frente a v2, número de *slots*, resolución, destilación sí o no y tipo de destilación | 40–60 | D | Nov–ene |
| M7 | **Gold ampliado**: el radiólogo etiqueta **≥200 estudios** de train (con informe e imágenes) fuera de los folds de selección. Hace que el IC del gold deje de ser ±0.06. Además, etiqueta **300 informes solo de texto** para medir el LLM. | 0 (≈20–25 h H) | D + H | Nov–ene |
| M8 | Análisis LLM frente a gold (Tab. 5, Fig. 4) y análisis por hallazgo (Fig. 5) | <2 | D | Ene |
| M9 | Baselines reproducidos con el mismo split: MRNet (AlexNet 3 planos), ELNet, un notebook público DINOv2-2.5D (reducido) y el mejor backbone de exp002–008 | 25–35 | D | Dic–ene |

### SHOULD (eleva mucho la probabilidad de aceptación)
| # | Experimento | GPU-h | Dep. | Cuándo |
|---|---|---|---|---|
| S1 | **fastMRI+** externo: 1,172 volúmenes coronales PD/T2, con 7 hallazgos mapeables y cajas para el pointing game | 6 (+ reconstrucción RSS) | E | Dic–feb |
| S2 | **LB privado** con *late submission* del estudiante y de 2–3 puntos de la curva | 0 (cuota de Kaggle) | **K** | Nov, si Kaggle lo habilita |
| S3 | Curva de calidad de etiqueta: reentrenar con etiquetas de reglas, Qwen v1, v2 y un LLM grande o ensemble (3–4 entrenamientos OOF del estudiante) | 25–30 | D | Ene–feb |
| S4 | Robustez: series faltantes y perturbaciones (solo inferencia) | 3 | D | Feb |
| S5 | Cuantización INT8 / ONNX Runtime en CPU con su ΔAUC | 1 | D | Feb |
| S6 | Energía con CodeCarbon (Wh/estudio, CO2) para todos los modelos de la Tab. 3 | 2 | D | Feb |
| S7 | Interpretabilidad: matriz de atención y Grad-CAM | 1 | D + E | Feb |

### NICE (si sobra tiempo o para la revisión)
| # | Experimento | GPU-h | Dep. | Notas |
|---|---|---|---|---|
| N1 | **OAI con MOAKS**: menisco medial y lateral por región, ACL, derrame-sinovitis, quiste de Baker, BML y OA por compartimento (medial, lateral y PF) a partir de cartílago y osteofitos. Es el mejor mapeo a nuestras 12. | 15–20 | E | Acceso por NDA con cuenta Login.gov o eRA Commons; descarga pesada; protocolo 3T distinto. **Guardarlo para la revisión.** |
| N2 | SKM-TEA (155 qDESS sagitales, 16 patologías con caja) | 3 | E | Muestra pequeña y secuencia DESS, que es otro dominio. |
| N3 | Estudio con lectores: 2 radiólogos con y sin modelo sobre 50 casos | 0 (≈15 h H) | H | Lo hicieron MRNet y P3; mucho valor clínico, pero caro. |
| N4 | MOST | — | E | Disponible vía NIA Aging Research Biobank; no queda claro si incluye imágenes. Descartar salvo que sobre tiempo. |
| N5 | Early-exit adaptativo por estudio (presupuesto dinámico frente a fijo) | 10 | D | Podría ser un paper aparte. |

**Total aproximado:** MUST ≈ 115–155 GPU-h · SHOULD ≈ 40–45 · NICE ≈ 30. Con un pod de ~0.8–2 USD/h, el MUST+SHOULD cuesta **150–400 USD** [NV].

### Disponibilidad, licencia y mapeo de los datasets externos
| Dataset | Tamaño / series | Licencia y acceso | Mapeo a nuestras 12 | Fuente |
|---|---|---|---|---|
| **MRNet** (Stanford) | 1,370 exámenes; sagital T2, coronal T1 y axial PD (en `.npy` de 256 px, sin metadatos DICOM ni espaciado [NV]) | Research Use Agreement de Stanford AIMI: solo uso no comercial, sin redistribución ni reidentificación; descarga por Redivis; DOI 10.71718/rcbp-8c35. **[V]** | ACL → ACL; meniscal tear → max(MM, LM); abnormal → max(12) (aproximado). **3 etiquetas.** Hace falta un mapeo de *slots* por secuencia. | [AIMI MRNet](https://aimi.stanford.edu/datasets/mrnet-knee-mris), [AIMI datasets](https://aimi.stanford.edu/shared-datasets) |
| **KneeMRI** (Štajduhar, Rijeka) | 917 exámenes, sagital PD-FS, 1.5T Siemens; ACL sano, parcial o roto | **CC BY-NC-ND 4.0**, en Zenodo. **[V]** | ACL (sano frente a parcial o roto; también roto frente al resto). **1 etiqueta**, una sola serie: sirve también como test de series faltantes. | [Zenodo 14789903](https://zenodo.org/records/14789903) |
| **fastMRI+** (NYU + anotaciones de Microsoft) | 1,172 volúmenes de rodilla, coronal PD o PDFS; 22 categorías con cajas por corte, de un solo radiólogo MSK | Imágenes: acuerdo de NYU fastMRI (investigación y docencia, sin redistribución). Anotaciones: repositorio MIT y artículo CC BY 4.0. **[V]** | Meniscus tear → MM/LM (medial o lateral según la posición de la caja y la lateralidad: arriesgado → usar max); ACL sprain → ACL; MCL sprain → MCL; joint effusion → derrame; periarticular cysts → Baker (aproximado); bone fracture/contusion/dislocation → max(contusión, fractura); subchondral edema → contusión u OA (aproximado); cartilage loss → OA (aproximado, sin compartimento). **≈7 etiquetas.** Ojo: el radiólogo vio una sola vista, así que las etiquetas son "potenciales", y las imágenes están volteadas arriba-abajo. | [fastMRI+ paper (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/), [GitHub](https://github.com/microsoft/fastmri-plus), [fastMRI](https://fastmri.med.nyu.edu/), [RSNA AI fastMRI](https://pubs.rsna.org/doi/full/10.1148/ryai.2020190007) |
| **SKM-TEA** (Stanford) | 155 escaneos qDESS (86/33/36), segmentación de 6 tejidos y cajas 3D de 16 patologías | Stanford AIMI / Redivis (RUA no comercial [NV]); DOI 10.71718/2ghb-nv62. **[V] el acceso** | Meniscal tear → MM/LM; ligament tear → ACL/MCL; effusion → derrame; cartilage lesion → OA (aproximado). Muestra pequeña. | [AIMI SKM-TEA](https://aimi.stanford.edu/datasets/skm-tea-knee-mri), [arXiv](https://arxiv.org/abs/2203.06823) |
| **OAI** (+ MOAKS) | 4,796 participantes, 3T (DESS sagital, IW-FS sagital, IW coronal [NV]) | NIMH Data Archive: login por RAS (eRA Commons, Login.gov o PIV), cuenta NDA y aceptar los términos de OAI. **[V]** | Con MOAKS [NV]: menisco por región → MM/LM; ACL; derrame-sinovitis → derrame y sinovitis; quiste de Baker; BML → contusión (aproximado); cartílago y osteofitos por compartimento → OA medial, lateral y PF. **Hasta ~10 de 12.** | [NYU data catalog OAI](https://datacatalog.med.nyu.edu/dataset/10162), [FNIH OAI](https://fnih.org/our-programs/osteoarthritis-initiative/) |
| **MOST** | 3,026 participantes | NIA Aging Research Biobank (datos públicos de uso; imágenes no confirmadas) [V parcial] | Similar a OAI con MOAKS [NV] | [NYU catalog MOST](https://datacatalog.med.nyu.edu/dataset/10668) |

**Orden recomendado:** MRNet → KneeMRI → fastMRI+ (los tres son de descarga inmediata) → OAI en la revisión.

---

## 5. Cronograma y reparto

Personas: **R** = Rodrigo (experto: arquitectura, entrenamiento, redacción técnica) · **A** y **B** = no expertos.
Reparto: **A** = datos externos y pipelines de evaluación (tareas guiadas por specs con Claude/Codex). **B** = costo, benchmarks, figuras y gestión de referencias y formato.
Los radiólogos (gold ampliado) son colaboradores externos y coautores; hay que reclutarlos **ya**.

| Periodo | Hito | R | A | B |
|---|---|---|---|---|
| 6–22 oct | Kaggle fases A y B; congelar el modelo; medir el runtime oficial (M1, M2, M3-K) | Modelo | Smoke de eficiencia | Logs de tiempo y CodeCarbon |
| 22–26 oct | **Enviar ISBI** + arXiv (respetar la regla de no compartir código antes del 22-oct) | Redacción | Figuras | Formato y bib |
| 27 oct–15 nov | Liberar código en GitHub; consultar a RSNA/MIRA: ¿uso post-competición, imágenes en figuras, publicar pseudo-etiquetas?; activar *late submissions* (S2); pedir acceso a MRNet, KneeMRI, fastMRI (y OAI) | Consulta a RSNA | Accesos a datasets | Entorno de benchmark |
| nov | M4/M5 en *zero-shot*; arrancar M7 (radiólogo); especificaciones SDD de las ablations | Specs y revisión | MRNet y KneeMRI loaders + mapeo | M3 sin Kaggle (CPU laptop, A100) |
| dic | M6 (screen en fold 0 → OOF), M9 baselines | Ablations | fastMRI+ loader (S1) | Baselines MRNet y ELNet (con plantilla) |
| ene 2027 | M7 cerrado → M8; S3 (calidad de etiqueta); **notificación ISBI (~ene) [NV]** | Análisis LLM | *Fine-tune* externo | Tablas de costo (Tab. 7) |
| feb | S4–S7 robustez, INT8, energía e interpretabilidad; **congelar experimentos el 28-feb** | Interpretabilidad | Robustez | INT8 y energía |
| mar | Borrador completo v1 (todas las secciones), revisión interna y revisión por el radiólogo | Intro, métodos y discusión | Resultados externos | Figuras finales y suplementario |
| abr | v2; **ISBI en Lausana (~abr) [NV]**: presentar y recoger feedback; comprobar solapamiento de texto con ISBI (<30 % reutilizado) | Pulido | Checklist de reproducibilidad | Highlights, graphical abstract y cover letter |
| **14 may 2027** | **ENVÍO a CMIG** (margen hasta el 30-jun) | Envío | — | — |
| ~jul–ago 2027 | Primera decisión tras revisión (~70 días) → revisión (añadir OAI si la piden) | | | |
| ~oct 2027 | Aceptación esperada (~145 días según Insights) | | | |

**Por qué el 14-may y no antes:** el gold ampliado (M7) y la validación externa son los pasos críticos. Enviar antes de presentar en ISBI no lo prohíbe la política, pero conviene incorporar el feedback de ISBI y tener citable el DOI de IEEE.

---

## 6. Riesgos de rechazo y mitigación

| Riesgo | Prob. | Mitigación |
|---|---|---|
| **Desk reject por "incremental / solo un resultado de Kaggle"** | Alta | Tesis centrada en método más análisis (label-query que permite poda, etiquetas LLM frente a gold, costo en CPU), no en el ranking. Validación externa en ≥2 datasets. Cover letter que explique la novedad frente a P2 y P5. |
| **Gold n=58** → IC enormes, conclusiones débiles | Alta | M7 (≥200 estudios con gold) + S2 (LB privado) + bootstrap. Reportar el gold como veto, tal como el SDD lo diseñó. |
| **Solapamiento con ISBI** (redundancia) | Media | ≥50 % de contenido nuevo; citar y declarar ISBI en la cover letter; reescribir el texto en lugar de copiarlo; texto reutilizado <30 %. |
| **Licencia RSNA/MIRA**: imágenes en figuras, uso post-competición o liberar pseudo-etiquetas | Media | Consultar por escrito en noviembre; las figuras cualitativas con MRNet o fastMRI; el data statement explica que no se puede redistribuir y se liberan código y pesos (si la licencia lo permite). |
| **Datos externos con dominio muy distinto** (MRNet en `.npy` sin espaciado, fastMRI con un solo plano) → AUC bajo | Media | Presentarlo como hallazgo de robustez; reportar *zero-shot* y *fine-tune* con pocos datos; usar la poda por series como ventaja ("funciona con un solo plano"). |
| **Test de Kaggle sin *late submission*** o con etiquetas no expertas | Media | Verificar en noviembre; si no se puede, M7 basta. |
| **Revisor: "faltan baselines SOTA"** | Media | M9 con ≥4 baselines y comparación con las cifras publicadas de MRNet y KneeMRI (Tab. 6). |
| **Afirmaciones de eficiencia no reproducibles** | Baja-media | Protocolo de medida fijo, hardware declarado, script público, mediana de N=100 estudios y *warm-up*. |
| **Equipo pequeño / no expertos** → retraso | Media | Congelar experimentos el 28-feb; la sección NICE es sacrificable; specs con criterios de aceptación y Codex para el código repetitivo. |
| **Declaración de IA generativa** (uso de LLM en etiquetas y en la redacción) | Baja | Declarar ambos: el LLM como método en la sección 3.2 y la ayuda en la redacción en la sección de declaración. |

### Cascada si CMIG rechaza
1. **CMIG** (IF 5.5, suscripción gratis) — objetivo.
2. **Computers in Biology and Medicine** (Elsevier). Se puede usar el Article Transfer Service de Elsevier, que **sí aparece en la guía de CMIG** [V], y reaprovechar las revisiones. IF y APC [NV].
3. **IEEE JBHI** (el formato IEEE facilita la conversión desde ISBI; exige más novedad metodológica) [NV]. Alternativa: **Journal of Imaging Informatics in Medicine** (Springer), si el enfoque se vuelve más clínico o de despliegue [NV].
4. **Machine Learning for Biomedical Imaging (MELBA)**: open access sin APC y respetada en la comunidad MICCAI [NV].
5. Respaldo final: **Scientific Reports** (APC alto [NV]) o **Journal of Imaging (MDPI)** (rápido, APC [NV]).

---

## 7. Acciones inmediatas (esta semana, sin tocar el plan de Kaggle)
1. Reclutar 1–2 radiólogos MSK como coautores para M7 (300 informes de texto y ≥200 estudios con imagen).
2. Dejar el protocolo de medida de costo (M3) escrito y montado **antes del 22-oct**: el runtime oficial de Kaggle solo se puede medir mientras la competición está abierta.
3. Guardar `metrics.json` con las predicciones OOF por estudio de cada modelo de la curva, para poder recalcular la estadística sin reentrenar.
4. Pedir hoy el acceso a MRNet, KneeMRI (descarga abierta) y fastMRI. OAI tarda más.
5. Leer las reglas de la competición: uso post-competición de los datos, *late submissions*, publicación de pseudo-etiquetas e imágenes.
