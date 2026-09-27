# 07 — Soluciones ganadoras de Kaggle en imagen médica volumétrica

Qué hicieron los top 1–5 de las competencias RSNA en Kaggle más parecidas a la nuestra, qué movió la aguja
y cuánto costaba correrlo. Los writeups se leyeron directamente en Kaggle (septiembre de 2026). Los PDFs son
papers derivados (de la competencia, del dataset o de los ganadores).

> Nota sobre las fuentes: los writeups de Kaggle no son PDF; aquí van resumidos, con su URL. Los números que
> no aparecen en el writeup se marcan como "n/d", porque no los inventé.

---

## 0. Primero, la métrica de eficiencia de NUESTRA competencia (overview de Kaggle)

```
Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400        (se minimiza)
```
`Benchmark` es la AUC de `sample_submission.csv` (≈ 0.5) y `maxAUC` es la mejor AUC privada. Como
`Benchmark − maxAUC < 0`, el término de AUC resta.

**Tipo de cambio (estimación mía, suponiendo Benchmark ≈ 0.50 y maxAUC ≈ 0.95):**
0.01 de AUC ≈ 0.022 puntos ≈ **12 min de runtime**. Dicho de otro modo, **1 h de runtime ≈ 0.05 AUC**.
- Ejemplo: 0.90 AUC en 30 min da −2.00 + 0.056 = **−1.944**. Un ensemble de 0.94 AUC en 9 h da
  −2.09 + 1.00 = **−1.09**. El modelo barato gana por mucho.
- El runtime es **de punta a punta**: leer DICOM, preprocesar, detectar y clasificar. En estos
  pipelines, la carga de DICOM y la etapa 1 (localización) suelen costar tanto como el clasificador.
- Pendiente de verificar en el foro: si el track de eficiencia exige notebook **solo CPU**, como en
  Feedback Prize. El overview actual solo dice "CPU o GPU ≤ 9 h".

---

## 1. Tabla rankeada de papers (PDFs en esta carpeta)

| rank | paper (año, venue) | PDF | puntaje | por qué nos sirve | números clave |
|---|---|---|---|---|---|
| 1 | Liu et al., *Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification* (AnyMC3D) (2025, arXiv; 1er lugar en el VLM3D Challenge) | sí: `2025_Liu_Revisiting2DFoundationModels3D.pdf` · [arXiv 2512.12887](https://arxiv.org/abs/2512.12887) | **9** | Es casi nuestra receta: un backbone 2D **congelado** (DINOv2/v3), adaptadores LoRA y fusión de cortes, con multi-vista y multi-modalidad. Además compara directamente contra la "receta RSNA-Kaggle" (2.5D + BiLSTM). | AUC media de 0.894 en 10 tareas con **1.3 M de parámetros entrenables por tarea** (de 10 a 40× menos que MST, que tiene 23 M). La receta RSNA-Kaggle, con 25 M entrenables, queda en 0.810. En su benchmark, los métodos 2D superan a los 3D. |
| 2 | Shi et al., *Intracranial Aneurysm Classification and Segmentation via Tri-Axial ROI and Multi-Task Learning* (2026, arXiv; 2º lugar de RSNA 2025 Aneurysm) | sí: `2026_Shi_AneurysmKaggle2ndPlace.pdf` · [arXiv 2606.26706](https://arxiv.org/abs/2606.26706) | 7 | ROI rápido con segmentación 2D en tres ejes antes del modelo 3D. En inferencia usan **solo encoder + cabeza** (sin decoder), con auxiliares multitarea. Es multi-modalidad (CTA/MRA/T2/T1-post). | Volúmenes de 224³, TTA ×8 y un ensemble de 2 folds. Con 2×T4 tardan unas 9 h en el ensemble "conservador". |
| 3 | Shang et al., *An Effective Transformer-based Solution for RSNA Intracranial Hemorrhage Detection Competition* (2022, arXiv) | sí: `2022_Shang_TransformerRSNAICHCompetition.pdf` · [arXiv 2205.07556](https://arxiv.org/abs/2205.07556) | 7 | Es el único paper que revisa el top 5 de una RSNA **y lo supera con menos cómputo**. Sirve de plantilla para un paper de eficiencia. | Supera al ganador de RSNA-ICH 2019 con **1/4 de los parámetros y el 10 % de los FLOPs**. Los top 5 usaban ensembles de 7 a 31 modelos. |
| 4 | Batra et al., *M-SCAN: Multistage Framework for Lumbar Spinal Canal Stenosis Grading Using Multi-View Cross Attention* (2025, arXiv) | sí: `2025_Batra_MSCAN_LumbarStenosisMultiView.pdf` · [arXiv 2503.01634](https://arxiv.org/abs/2503.01634) | 6 | Usa el dataset RSNA 2024 lumbar, que es multi-serie como el nuestro. Hace atención cruzada sagital↔axial sobre secuencias. | AUROC de 0.971 en estenosis de canal, con 1,975 estudios (arXiv, sin revisión por pares). |
| 5 | Lee et al., *Performance of the Winning Algorithms of the RSNA 2022 Cervical Spine Fracture Detection Challenge* (2024, Radiology: AI) | no (acceso cerrado): [doi 10.1148/ryai.230256](https://pubs.rsna.org/doi/full/10.1148/ryai.230256) | 6 | Evaluación formal del top 8 de una RSNA. Es la referencia que debemos citar para "estos pipelines funcionan". | Top 8: AUC media de 0.96, sensibilidad del 88 % y especificidad del 94 %. El mejor modelo previo tenía AUC 0.85. |
| 6 | Hermans et al., *RSNA 2023 Abdominal Trauma AI Challenge: Review and Outcomes* (2025, Radiology: AI) | no (acceso cerrado): [doi 10.1148/ryai.240334](https://pubs.rsna.org/doi/10.1148/ryai.240334) | 6 | Igual que el anterior, para trauma abdominal. Muestra qué hallazgos siguen siendo difíciles (los sutiles). | AUC media: hígado 0.92, bazo 0.91, riñón 0.94; intestino 0.85 y extravasación 0.85 (los hallazgos sutiles son los peores). |
| 7 | Saeed et al., *MGMT promoter methylation status prediction using MRI scans? An extensive experimental evaluation of deep learning models* (2023, Medical Image Analysis) | sí: `2023_Saeed_MGMTPredictionMRIEvaluation.pdf` · [arXiv 2304.00774](https://arxiv.org/abs/2304.00774) | 5 | Es una lección de validación: con una señal débil y un test chico, el CV es ruido. Aplica a nuestro gold de 58 estudios. | Usa el dataset RSNA-MICCAI 2021 (585 pacientes). Ningún modelo predice MGMT de forma fiable. |
| 8 | Tran et al., *Video Classification with Channel-Separated Convolutional Networks* (CSN) (2019, ICCV) | sí: `2019_Tran_CSN_ChannelSeparatedConvNets.pdf` · [arXiv 1904.02811](https://arxiv.org/abs/1904.02811) | 5 | Es el backbone del 4º lugar de cervical ("CSN is all you need for 3D"). Es 3D barato, preentrenado en video, y no sobreajusta tanto como un 3D puro. | ir-CSN-152 preentrenado en ig65m/kinetics. Fue el mejor 3D en un dataset chico (según el writeup del 4º lugar). |
| 9 | Isensee et al., *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation* (2024, MICCAI) | sí: `2024_Isensee_nnUNetRevisited.pdf` · [arXiv 2404.09556](https://arxiv.org/abs/2404.09556) | 4 | nnU-Net ResEnc es la base del 1º y el 2º de RSNA 2025 Aneurysm, que usan la segmentación como guía de ROI y como backbone preentrenado. | n/a (metodológico). |
| 10 | Richards et al., *The RSNA Lumbar Degenerative Imaging Spine Classification (LumbarDISC) Dataset* (2026, Radiology: AI) | no: [doi 10.1148/ryai.250480](https://pubs.rsna.org/doi/full/10.1148/ryai.250480) | 4 | Es el paper del dataset de la competencia más parecida a la nuestra. | 2,697 pacientes, 8 instituciones de 6 países; 1,874 equipos en Kaggle. |
| 11 | Rudie et al., *The RSNA Abdominal Traumatic Injury CT (RATIC) Dataset* (2024, arXiv → Radiology: AI) | sí: `2024_Rudie_RATIC_AbdominalTraumaDataset.pdf` · [arXiv 2405.19595](https://arxiv.org/abs/2405.19595) | 3 | Modelo de "paper de dataset" de RSNA: cómo describe el etiquetado y el split. | 4,274 CT. |
| 12 | Lin et al., *The RSNA Cervical Spine Fracture CT Dataset* (2023, Radiology: AI) | no: [doi 10.1148/ryai.230034](https://pubs.rsna.org/doi/full/10.1148/ryai.230034) | 3 | Contexto del dataset cervical. Solo 87 estudios tenían segmentación, y aun así bastaron para la etapa 1. | 3,112 CT de 12 instituciones. |
| 13 | Baid et al., *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification* (2021, arXiv) | sí: `2021_Baid_BraTS2021Benchmark.pdf` · [arXiv 2107.02314](https://arxiv.org/abs/2107.02314) | 3 | Es el paper de la competencia de MRI multisecuencia (FLAIR/T1/T1CE/T2). | n/a |
| 14 | Bönisch & Losaria, *Kaggle Chronicles: 15 Years of Competitions, Community and Data Science Innovation* (2025, arXiv) | sí: `2025_Bonisch_KaggleChronicles.pdf` · [arXiv 2511.06304](https://arxiv.org/abs/2511.06304) | 3 | Sirve para citar en la introducción que las soluciones de Kaggle generalizan. | Análisis de millones de kernels y discusiones. |

---

## 2. Competencia por competencia (writeups de Kaggle)

### 2.1 RSNA 2024 Lumbar Spine Degenerative Classification. Es la más parecida: MRI multi-serie (Sag T1, Sag T2/STIR, Ax T2), 25 etiquetas por nivel y lado

| # | writeup | arquitectura | trucos que movieron la aguja | costo |
|---|---|---|---|---|
| 1 | [Avengers / NANACHI](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution) | **Etapa 1:** una ConvNeXt 3D (volumen con padding a 32 cortes) predice el *instance_number* por nivel, combinando clasificación y regresión con la mediana. ConvNeXt-base y EffV2-L 2D a 512² regresan (x, y). **Etapa 2:** recorte de 5 cortes alrededor del punto, ConvNeXt-S / EffV2-S, **BiLSTM + attention-MIL** con pérdida auxiliar sobre la atención. | La attention-MIL bajó el LB público de 0.37 a 0.35. BiLSTM, pérdidas auxiliares y ensemble lo bajaron de 0.35 a 0.33. El **jitter aleatorio del índice de corte (±2) y de las coordenadas (±10 px)** fue "crucial" para tolerar errores de la etapa 1. Preentrenar el localizador con un dataset externo de coordenadas ayudó. **No funcionó:** backbones grandes (large < base < small), ViTs, Mamba y self-attention en lugar de LSTM, ni entrenar muchas épocas (usaron 7–14). | Entrenado en Colab T4. Tiempo de inferencia n/d. |
| 2 | [Ian Pan, Bartley, Yuji Ariyasu, kevin](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution) | Selección de cortes con 2.5D, luego YOLOX para las cajas por nivel y recorte, y después ConvNeXt-S con MIL de 5 cortes (Yuji). Ian: un CNN-transformer localiza el corte y la etapa 2 usa encoder + LSTM + attention pooling. | **Denoising de etiquetas:** con el OOF del ensemble quitaron las muestras con \|label − pred\| ≥ 0.8 y reentrenaron (+1 % en público y privado). Pseudo-etiquetas suaves (0.5·gt + 0.5·pseudo). **Niveles y lados compartidos** en un solo modelo (más datos). Postproceso: ×1.25 al severo máximo. | TTA de 9 rotaciones (Ian). Tiempo n/d. |
| 3 | [SonySpine / Moyashii](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/sonyspine-s-tkmn-moyashii-3rd-place-solution) · [código](https://github.com/Moyasii/Kaggle-2024-RSNA-Pub) | Keypoints con CenterNet (EffB6 y EffB4 + FPN) y recortes. Clasificador **encoder 2D + atención entre cortes**: 15 cortes sagitales y 10 axiales, las 3 series a la vez. Encoders chicos: **ResNet18 a 160–224, MNasNet-S, ConvNeXt-N/T, EffV2-S**. | **Modelos independientes de nivel y lado**, con el lado derecho espejado: 5× y 10× más datos por condición. Pesos de CE [1, 2, 4] para las clases severas, pérdidas auxiliares, TTA con flip y temperatura 0.91. **No funcionó:** 3D-CNN, 2.5D + atención, 2D + LSTM, modelos especializados por nivel y focal loss. | Mejor modelo individual: CV 0.3858. Ensemble de **30 modelos**: CV 0.3643. |
| 4 | [SPINE CHART (yu4u, tattaka)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/spine-chart-4th-place-solution) | Keypoints con UNet 2.5D y **128²** (baratos). Recorte de 2× la distancia entre keypoints. Encoder 2D por corte + transformer encoder + attention pooling, multi-vista y multi-condición (caformer_s18, resnetrs50, rdnet_tiny, maxxvitv2_nano). | Attention pooling **antes** del transformer, con pérdida auxiliar. Stacking con MLP sobre Nelder-Mead y con LightGBM/XGB. | n/d |
| 5 | [two people (siwooyong, ElFazouani)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/two-people-5th-place-solution) | Detección por heatmap (UNet 2D EffB5 + modelo secuencial en z), etiquetas gaussianas expandidas y recortes de **5×64×64 → 128²**. 2.5D (CNN + RNN) con RNN **a lo largo de las condiciones del mismo nivel**. | **Entrenar en dos pasos:** primero backbone con BCE sin pesos (maximiza la AUC) y después congelar el backbone y ajustar solo la cabeza a la métrica ponderada. CutMix con p = 1.0 fue el mejor regularizador. Correlación entre etiquetas vía RNN. No sirvió un backbone más grande. | n/d |

### 2.2 RSNA 2023 Abdominal Trauma Detection (CT, organ-level + bowel/extravasation)

| # | writeup | arquitectura | trucos | costo |
|---|---|---|---|---|
| 1 | [Team Oxygen](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/writeups/team-oxygen-1st-place-solution-team-oxygen) · [código](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution) | Segmentación 3D de órganos y recortes. **2.5D con 32 cortes equidistantes × 3 canales (cortes adyacentes), 384²**, CoaT-lite medium/small y EffV2-S + **GRU**. | **Pérdida auxiliar de segmentación (Dice): +0.01–0.03**. Etiquetas suaves según la visibilidad del órgano por corte. Agregación por máximo sobre cortes. | Mejor modelo individual con OOF 0.326; ensemble 0.31x. Tiempo n/d. |
| 2 | [Theo Viel](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/writeups/on-strike-2nd-place-solution) · [código](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma) | Un efficientnetv2_rw_t rápido clasifica qué órganos hay en cada corte. Modelo 2D por corte (maxvit_tiny 512) + recortes de órgano con ResNet18 3D + coatnet_1 224 con **11 cortes**. **Un RNN aparte agrega las probabilidades.** | **Muestrear cortes con inteligencia** (dentro del órgano) para reducir el ruido de etiqueta. Usar **1 de cada 2 cortes, pila ≤ 600**, DICOM con dicomsdl y preproceso en GPU. Quitó la cabeza secuencial de la etapa 1 porque 2D bastaba: **gran aceleración**. | **Pipeline < 4 h sin ensemble.** El ensemble no ayudó en el privado. |
| 3 | [Yuji Ariyasu](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/writeups/aillis-jp-yuji-ariyasu-3rd-place-solution) | Segmentación 3D (código de Qishen), cubos por órgano y **2.5D con 15 imágenes × 3 canales a 128²** + pooling/LSTM/GRU. | Enmascarar el hígado (la caja trae mucho ruido). Sampler por órgano para entrenar un solo modelo con todos los órganos. Dos tipos de recorte. | CV 0.3316. |
| 4 | [sheep](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/writeups/sheep-4th-place-solution) | Máscaras de TotalSegmentator, 14 cortes y UNet con encoder PVT-v2 / MaxViT (clasificación + segmentación). | Preentrenar con máscaras 2D. El autor cree que perdió por no usar RNN. | pvt-b2 sola: CV 0.378; ensemble 0.353. |

### 2.3 RSNA 2022 Cervical Spine Fracture Detection (CT, 7 vértebras + a nivel paciente)

| # | writeup | arquitectura | trucos | costo |
|---|---|---|---|---|
| 1 | [Qishen Ha](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution) · [inferencia](https://www.kaggle.com/code/haqishen/rsna-2022-1st-place-solution-inference) | **Segmentación 3D a 128³** (ResNet18d / EffV2-S UNet, entrenada con solo 87 máscaras) y recorte por vértebra. **2.5D: 15 cortes × 5 canales (±2) + la máscara como 6º canal**, CNN 2D + **LSTM**. Hay dos tipos: por vértebra y por paciente (7×15 imágenes). | Un CNN 3D directo **no funcionó**. La máscara como canal separa las vértebras vecinas. | Ensemble de unos 30 modelos (5 folds × varias arquitecturas, 384–512²). **Envío: 7.5 h.** |
| 2 | [RAWE](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/rawe-2nd-place-solution-segmentation-2-5d-cnn-gru-) · [código](https://github.com/ryanyuerong/RSNA2022RAWE) | Segmentación con **UNet 2.5D de EffB0** (3 canales i−1, i, i+1). Recorte por vértebra de **24 cortes** y CNN (EffV2-S / ResNeSt50d) + **BiGRU + atención**. | Buscaron explícitamente "la solución más concisa y eficiente". Inicializaron bien la GRU y usaron SpatialDropout. | n/d |
| 3 | [Darragh](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/darragh-3rd-place-solution) · [código](https://github.com/darraghdog/RSNA22) | Una EffV2 predice la caja por corte y se hace **un solo recorte por estudio**, reescalado a 512. 2.5D CNN + RNN 1D + **atención** (ResNeSt50d, SE-ResNeXt50). | **Etiquetas débiles por corte** = (fracción de vértebra en el corte) × (etiqueta del estudio). Así evitó anotar cajas de fractura. La atención sobre el RNN "ayudó mucho"; los transformers no mejoraron. | Unas 9 h de entrenamiento por fold. |
| 4 | [Selim Seferbekov](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/selim-seferbekov-4th-place-solution-csn-is-all-you) · [código](https://github.com/selimsef/rsna_cervical_fracture/) | Segmentación con encoder ir-CSN-50 y clasificación con **ir-CSN-152 3D preentrenado en video** (40 cortes, 256²), con la máscara como canal. | Submuestreo z ×2 y memory-mapping. Un 3D preentrenado en video no sobreajusta como un 3D puro. | 4 folds; tiempo n/d. |

### 2.4 RSNA-MICCAI 2021 Brain Tumor Radiogenomic (MRI multisecuencia: FLAIR/T1w/T1wCE/T2w, predecir MGMT)

| # | writeup | arquitectura | lección |
|---|---|---|---|
| 1 | [Tunisia.ai ("1st place with very simple code")](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/writeups/tunisia-ai-i-love-this-competition-1st-place-solut) | **ResNet10 3D (MONAI), 256², bs 8, 15 épocas, sin ensemble.** Truco del "mejor corte central": centra el volumen en el corte con más cerebro (+0.01–0.02 CV). | El CV era casi aleatorio: el mismo modelo con la misma semilla daba AUC entre 0.53 y 0.62. Rankeó ideas con **100 a 250 entrenamientos por idea**. Los modelos grandes, el ensemble y los 2D no sirvieron. **Con una señal débil, lo simple y bien validado gana el shake-up.** |
| 2 | [Minh Phan](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/writeups/minh-phan-2nd-place-solution) | **Fusión temprana de secuencias:** las 4 secuencias entran como 4 canales, con conv 4→3, EffNet-B0 + LSTM (2 capas, h = 32) sobre T = 10 cortes uniformes, a 256². La secuencia que falta se rellena con ceros. | Con LB público de 0.65–0.67 quedó 2º. Aumentos simples. |
| 3 | [Cedric Soares](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/writeups/cedric-soares-3rd-place-solution) | 4 EffNet-B3 2D, una por secuencia, con agregación por paciente (max/mean/min). | — |

### 2.5 Extra (muy parecida y reciente): RSNA 2025 Intracranial Aneurysm Detection (CTA/MRA/T2/T1-post, 13 localizaciones + presencia)

| # | writeup | arquitectura | eficiencia |
|---|---|---|---|
| 1 | [1st Place: Location-Aware Aneurysm Detection via Vessel-ROI Masking](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/1st-place-solution) | **Coarse-to-fine:** un nnU-Net a 1 mm localiza, recorta un ROI de 140 mm³ y dos nnU-Net finos segmentan vasos. El clasificador 3D (128×256×256) usa el encoder del nnU-Net preentrenado y hace **pooling enmascarado por región vascular** (13 vectores), con un transformer entre localizaciones y un auxiliar de "esfera" en la lesión. | "Coarse-to-fine mejora precisión **y velocidad**". Quitó el último bloque del decoder **sin pérdida**. Usa pesos EMA. |
| 2 | [2nd Place](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution) · paper arriba (#2) | ROI con nnU-Net 2D en tres ejes y multitarea 3D con nnU-Net a 224³. | Inferencia solo con encoder + cabeza; unas 9 h en 2×T4. |

### 2.6 "Efficiency Prize" en Kaggle
- **No encontré ningún Efficiency Prize de imagen médica volumétrica en Kaggle.** Los que hubo fueron de NLP o
  tabulares. Ejemplos: Feedback Prize – Effectiveness, [Team Hydrogen](https://www.kaggle.com/competitions/feedback-prize-effectiveness/writeups/team-hydrogen-team-hydrogen-1st-place-solution),
  con un kernel de eficiencia de **0.557 privado en 5 min 40 s**; LLM Detect AI Generated Text,
  [1st Efficiency](https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/471898), con TF-IDF + NB/SGD/LGBM
  **en CPU en 17 min** (ambos datos salen del resultado de búsqueda y no los verifiqué en el writeup).
  El patrón común es que el ganador de eficiencia **no es una versión recortada del ensemble ganador,
  sino otro modelo mucho más simple**.
- Resumen general de lo que hicieron los ganadores de eficiencia: L. Monigatti, *Towards Green AI*
  (1er lugar del 2023 Kaggle AI Report), [Medium/TDS](https://medium.com/data-science/towards-green-ai-how-to-make-deep-learning-models-more-efficient-in-production-3b1e7430a14).
  No pude abrir el texto completo.
- **Consecuencia:** nuestro track no tiene precedente directo en RSNA. Un paper que muestre el frente de
  Pareto AUC–runtime en un reto RSNA multi-serie sería novedoso por sí mismo.

---

## 3. Patrones que se repiten en ganadores

1. **Dos etapas: localizar y después clasificar un recorte.** Aparece en el 100 % de los top 5 de lumbar,
   cervical, abdominal y aneurisma. La etapa 1 es barata: keypoints/heatmap a 128–512², segmentación 3D a 128³ o
   a 1 mm. El recorte del ROI es lo que más sube la AUC (los notebooks de knee a ~0.94 hacen lo mismo con 130 mm).
2. **2.5D + agregador secuencial, no 3D.** Un encoder 2D ImageNet por corte (con ±1–2 cortes como
   canales) y encima LSTM/GRU/attention-MIL/transformer. Qishen, SonySpine y lumbar #1 reportan
   explícitamente que **el 3D puro no funcionó**. La excepción es el 3D preentrenado en video (CSN) o en
   segmentación (nnU-Net). El paper AnyMC3D (#1) lo confirma: 2D > 3D.
3. **Backbones chicos y medianos.** ConvNeXt-S/T/N, EffV2-S, ResNet18, caformer_s18 y MaxViT-tiny.
   Lumbar #1: large < base < small. Lumbar #5: un backbone más grande no ayudó. Brain #1: los modelos grandes
   dieron CV aleatorio.
4. **Pocos cortes.** 5 (lumbar), 10–15 (lumbar #3, brain #2), 11 (abdominal #2), 15 (cervical #1) o 24 (cervical #2).
   El muestreo es uniforme dentro del ROI, no sobre todo el volumen.
5. **Robustez a errores de la etapa 1:** jitter del índice de corte y de las coordenadas durante el
   entrenamiento (lumbar #1, lo llama "crucial").
6. **Compartir pesos entre niveles, lados o órganos, con espejado.** Multiplica los datos entre 5 y 10 veces
   (lumbar #2 y #3; abdominal #3 con su sampler por órgano).
7. **Pérdidas auxiliares:** segmentación (abdominal #1, +0.01–0.03), atención o corte (lumbar #1 y #4),
   esfera de lesión (aneurisma #1) y máscara como canal extra (cervical #1 y #4).
8. **Denoising de etiquetas con OOF** (lumbar #2, +1 %) y **pseudo-etiquetas suaves** (lumbar #2 y #3).
9. **Entrenar en dos pasos:** primero la representación con BCE plana (AUC) y después solo la cabeza con la métrica
   (lumbar #5).
10. **Ensembles grandes para el LB** (30 modelos en lumbar #3, unos 30 en cervical #1), aunque la ganancia sobre el mejor
    modelo individual es chica (lumbar #3: de 0.3858 a 0.3643; abdominal #2: "el ensemble no ayudó en el privado").

## 4. Qué sirve para eficiencia

- **El recorte guiado por un localizador barato es el mejor trato AUC/cómputo.** Sube la AUC y permite
  una resolución menor en la etapa 2 (lumbar #5: recortes de 64² reescalados a 128²; abdominal #3: 128²).
- **Submuestrear cortes:** uno de cada dos cortes, pila ≤ 600 y 5–15 cortes por ROI (abdominal #2 corrió todo en < 4 h
  sin ensemble).
- **La E/S de DICOM se optimiza** con dicomsdl, preproceso en GPU y memory-mapping (abdominal #2, cervical #4). En
  nuestra fórmula, cada minuto cuenta.
- **Un solo modelo bien regularizado** (CutMix con p = 1, jitter, pérdidas auxiliares) en lugar de un ensemble. La diferencia
  entre el mejor individual y el ensemble es de ~0.02 en log-loss, que en AUC equivale a pocas centésimas. Al tipo de cambio
  de la §0, eso no paga horas de ensemble.
- **Cortar la inferencia:** solo encoder + cabeza, sin decoder de las auxiliares (aneurisma #1 y #2), y
  quitar bloques pesados que no aportan.
- **Backbone congelado + LoRA + fusión de cortes** (AnyMC3D) iguala o supera la receta 2.5D + LSTM con
  20× menos parámetros entrenables. Con eso se entrena y valida rápido y hay un solo backbone para todas las series.
- **TTA y ensembles solo si el Δ AUC justifica el tiempo:** 0.01 AUC ≈ 12 min con nuestra fórmula (supuesto de la §0).

---

## Ideas para nosotros

1. **Etapa 1 barata de ROI de rodilla + etapa 2 sobre el recorte.** Un localizador tipo heatmap/keypoint a 128²
   (lumbar #4 y #5, aneurisma #1) que dé el centro de la articulación por serie. Después, 5–16 cortes recortados a 224 con
   DINOv2-S. Es la brecha principal entre nuestro 0.78 y el 0.94 público.
2. **DINOv2 congelado + LoRA + attention-MIL entre cortes y series** (AnyMC3D #1 + lumbar #1). Es
   publicable como "adaptación eficiente de un FM 2D a MRI multi-serie" y mide AUC por GFLOP.
3. **Jitter de corte y coordenadas al entrenar** (lumbar #1) para que un localizador barato y algo impreciso
   no hunda la AUC. Es gratis en inferencia.
4. **Denoising de las pseudo-etiquetas LLM con OOF** (lumbar #2: quitar \|y − p\| ≥ 0.8) y entrenamiento en dos pasos
   (lumbar #5). Encaja con nuestro problema de etiquetas ruidosas (acuerdo de ~0.80 con gold).
5. **Pérdidas auxiliares baratas** (abdominal #1, aneurisma #1): p. ej., predecir el nivel del corte o la
   visibilidad de la estructura. No cuestan nada en inferencia.
6. **Reportar el frente de Pareto AUC–runtime con la fórmula oficial** (§0), usando a Shang et al. 2022 (#3) como
   modelo de paper del tipo "igualamos al ganador con el 10 % de los FLOPs".

## Qué NO haría

- **Un ensemble grande de backbones grandes para el track de eficiencia.** Cada hora cuesta unas 0.05 AUC en la métrica,
  y los propios ganadores reportan que los modelos large no ayudan.
- **Un CNN 3D entrenado desde cero** (cervical #1, lumbar #3 y brain #1 reportan que falla). Si se quiere 3D, que sea
  preentrenado (CSN de video, nnU-Net).
- **Seleccionar con un CV ruidoso** (brain #1; Saeed 2023). Con 58 estudios gold, una diferencia de 0.01 es ruido. Hay que
  decidir con la pseudo-AUC en folds y repetir semillas, como ya dicta la SDD.
