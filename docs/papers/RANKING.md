# RANKING global de la literatura (docs/papers)

Síntesis de `_BRIEF_AGENTES.md` y de los README de `00_competencia` … `08_preprocesamiento_mri` (2026-09-26).
Nada de aquí es nuevo: todo número sale de esos README (y estos, de los PDFs o de Kaggle). Si un README marcó algo
como "no verificado" o preprint, aquí se mantiene el aviso.

Deduplicación hecha: cada PDF vive en una sola carpeta. Duplicados eliminados (idénticos por hash, o mismo paper):
MRNet, ELNet (se conservó la versión PMLR publicada), MRPyrNet, Vuskov y OrthoFoundation → `01`; MST, AnyMC3D y CoPAS → `03`;
Baharoon → `02`. Quedan **99 PDFs, 397 MB**. Todos empiezan por `%PDF`; ninguno pasa de 25 MB.

---

## 1. Regla de eficiencia (copia textual de `00_competencia`)

> "We compute a submission's **efficiency score** by:
> Efficiency = AUC / (Benchmark − max AUC) + RuntimeSeconds / 32400
>
> where AUC is the submission's score on the main competition metric, Benchmark is the score of the benchmark
> `sample_submission.csv`, max AUC is the maximum AUC of all submissions on the Private Leaderboard, and
> RuntimeSeconds is the number of seconds it takes for the submission to be evaluated. **The objective is to
> minimize the efficiency score.**"

Implicación (de `00`, §1): `Benchmark` ≈ 0.5 (deducido), `max AUC` ≈ 0.96 → Efficiency ≈ −AUC/0.46 + t/32400.
**Tipo de cambio: 0.001 de AUC ≈ 70 s de runtime** (32400·0.001/0.459 = 70.6 s). 1 min ≈ 0.00085 AUC · 5 min ≈ 0.004 ·
30 min ≈ 0.025 · 1 h ≈ 0.051. El runtime es reloj de pared del rerun oculto (~1300 estudios). La regla **no** exige CPU.
Consecuencia (`00` §2.1): top-10 de eficiencia tiene AUC pública ≥ 0.943; **un modelo con AUC < ~0.945 no gana eficiencia
aunque corra en 1 min**. `07` §0 da un cambio compatible (0.01 AUC ≈ 12 min).

---

## 2. Tabla maestra — top 40

Puntaje global (1–10) = relevancia para nuestro objetivo (track de eficiencia + paper con novedad) × solidez
(peer-review, tamaño, cercanía a rodilla MRI multi-serie). Carpeta = donde está el PDF (o "—" si solo hay enlace).

| # | paper (año, venue) | carpeta / PDF | pts | eje | idea que nos da |
|---|---|---|---|---|---|
| 1 | Müller-Franzes, *Medical Slice Transformer (MST)* (2025, Sci Rep) | 03 / `2025_MullerFranzes_MedicalSliceTransformer.pdf` | 10 | agregación | Es nuestra arquitectura publicada (DINOv2-S por corte + 1 capa transformer) en MRNet: menisco 0.85 vs 0.69 de 3D ResNet; ablaciones de congelar, tamaño y media. Baseline citable. |
| 2 | Liu, *AnyMC3D: Revisiting 2D FMs for 3D classification* (2025, arXiv; 1.º VLM3D) | 03 / `2025_Liu_AnyMC3D_2DFoundation3DClassif.pdf` | 9.5 | agregación · fine-tuning | Backbone 2D congelado + LoRA + pooling por query aprendida por vista: +0.11 AUC sobre pooling medio; 1.3 M entrenables/tarea; la receta RSNA-Kaggle (2.5D+BiLSTM) queda en 0.810 vs 0.894. |
| 3 | Qiu, *CoPAS: co-plane attention, 12 knee abnormalities* (2024, Nat Commun) | 03 / `2024_Qiu_CoPAS_KneeCoPlaneAttention.pdf` | 9 | rodilla · agregación | Comparador directo (12 hallazgos, multisecuencia, 5 centros). AUC 0.812 interna; no reporta FLOPs → podemos ganarles en AUC/cómputo. Matriz hallazgo×plano. |
| 4 | Vuskov, *Multi-condition analysis of knee MRI* (2025/26, Eur Radiol) | 01 / `2025_Vuskov_MultiCondition_SliceTransformer.pdf` | 9 | rodilla · backbone | ResNet18 2D + bloques 3D residuales inicializados a cero + transformer; 23 hallazgos, validación externa. Receta barata de contexto entre cortes. |
| 5 | Dehghani, *The Efficiency Misnomer* (2022, ICLR) | 05 / `2022_Dehghani_EfficiencyMisnomer.pdf` | 9 | eficiencia | Parámetros, FLOPs, throughput y latencia no correlacionan: reportar varios, con hardware explícito. Obligatorio en Métodos. |
| 6 | Beyer, *KD: a good teacher is patient and consistent* (2022, CVPR) | 05 / `2022_Beyer_KD_PatientConsistent.pdf` | 8.5 | eficiencia · etiquetas | Receta para destilar un ensemble/teacher grande a un modelo único (mismas vistas, schedule largo); las soft labels del teacher limpian ruido. |
| 7 | Kumar, *Fine-tuning can distort pretrained features (LP-FT)* (2022, ICLR) | 04 / `2022_Kumar_LPFT_FineTuningDistortsFeatures.pdf` | 8.5 | fine-tuning | Explica el colapso al descongelar; LP-FT +1 % ID / +10 % OOD vs FT. Coste cero en inferencia. |
| 8 | Tsai, *ELNet* (2020, MIDL) | 01 / `2020_Tsai_ELNet.pdf` | 8 | eficiencia · rodilla | Único antecedente de rodilla centrado en eficiencia: 53k–211k parámetros, una serie por tarea; normalización multi-corte 0.904 vs BN 0.751 (menisco). |
| 9 | Harvey, *MIL benchmark 3D neuroimaging* (2026, arXiv) | 03 / `2026_Harvey_MILBenchmark3DNeuro.pdf` | 8 | agregación | Mean-pooling empata/gana a ABMIL/TransMIL/3D en 4/6 tareas y entrena 25× más rápido → ablación "media vs atención" obligatoria. |
| 10 | Shang, *Transformer for RSNA ICH competition* (2022, arXiv) | 07 / `2022_Shang_TransformerRSNAICHCompetition.pdf` | 8 | kaggle · eficiencia | Plantilla de paper: supera al ganador RSNA-ICH con 1/4 de parámetros y 10 % de FLOPs. |
| 11 | Wortsman, *Model Soups* (2022, ICML) | 04 / `2022_Wortsman_ModelSoups.pdf` | 8 | fine-tuning · eficiencia | Promediar pesos de fine-tunes → ensemble a coste de 1 modelo (soup de folds). |
| 12 | Bolya, *Token Merging (ToMe)* (2023, ICLR) | 05 / `2023_Bolya_TokenMerging.pdf` | 8 | eficiencia | Sin reentrenar; 2× throughput con −0.2–0.3 % en ViT-L/H; funciona en video → ToMe entre cortes vecinos. |
| 13 | Bien, *MRNet* (2018, PLOS Med) | 01 / `2018_Bien_MRNet.pdf` | 8 | rodilla | Baseline canónico 2D + max-pool de cortes + LR sobre 3 planos (ACL 0.965, menisco 0.847). |
| 14 | Hu, *LoRA* (2022, ICLR) | 04 / `2022_Hu_LoRA.pdf` | 7.5 | fine-tuning | Se fusiona en W → misma latencia y FLOPs que el backbone. |
| 15 | Liu, *Does DINOv3 set a new medical vision standard?* (2025, arXiv) | 02 / `2025_Liu_DINOv3MedicalBenchmark.pdf` | 7.5 | backbone | S ≈ B ≈ L en médico; escalar tamaño/resolución no paga. |
| 16 | Yu, *CUFIT: curriculum fine-tuning under label noise* (2024, NeurIPS) | 04 / `2024_Yu_CUFIT_LabelNoise.pdf` | 7.5 | fine-tuning · etiquetas | LP robusto al ruido; LP → adapter filtra muestras limpias (+2–6 pp con 40 % de ruido). |
| 17 | Madaan, *Normalization schemes robustness in MR diagnosis* (2023, MIDL) | 08 / `2023_Madaan_NormSchemesRobustnessMRI.pdf` | 7.5 | preprocesamiento | En rodilla (fastMRI+), BN causa caídas de hasta −10 % AUROC con artefactos; GN/LN lo corrigen gratis. |
| 18 | Li, *GreenRFM* (2026, arXiv, preprint) | 06 / `2026_Li_GreenRFM.pdf` | 7.5 | etiquetas · eficiencia | Etiquetas de informes por consenso de 2 LLM; cohorte AH-Knee (3,662 estudios) 72.9 → 79.0 AUC zero-shot con 24 GPU-h. |
| 19 | Dutt, *PEFT for medical imaging: the missed opportunity* (2024, MIDL) | 04 / `2024_Dutt_PEFT_Medical_MissedOpportunity.pdf` | 7.5 | fine-tuning | 17 métodos PEFT; LoRA el mejor en ViT; con 100 % de datos FT full sigue siendo óptimo. |
| 20 | Ridnik, *ML-Decoder* (2023, WACV) | 03 / `2023_Ridnik_MLDecoder.pdf` | 7.5 | agregación | Cabeza con una query por etiqueta, coste casi constante con nº de clases. |
| 21 | Ozdemir & Li, *INT8 quantization effects in medical imaging* (2026, CHIL) | 05 / `2026_Ozdemir_INT8_CalibrationMedical.pdf` | 7.5 | eficiencia | Protocolo AUC + calibración + latencia p50/p99 GPU/CPU; FP16 ≈ FP32; INT8-PTQ puede degradar, QAT recupera. |
| 22 | Kreutzer, *LLM uncertainty-adjusted labels* (2025, Eur Radiol) | 06 / `2025_Kreutzer_LLMUncertaintyAdjustedLabels.pdf` | 7 | etiquetas | En MSK, cómo se trata "incierto" no cambia el AUC de forma significativa → no gastar experimentos ahí. |
| 23 | Yu, *OrthoFoundation* (2026, arXiv, preprint) | 01 / `2026_Yu_OrthoFoundation.pdf` | 7 | backbone · rodilla | FM de rodilla (DINOv3-L continuado): candidato a **teacher**, no a backbone (60.9 GMACs/corte, 11× DINOv2-S). |
| 24 | Doerrich, *MedMNIST+* (2025, Sci Rep) | 02 / `2025_Doerrich_MedMNISTplus.pdf` | 7 | backbone | CNNs siguen competitivas; la resolución satura; SAM el peor encoder para clasificar. |
| 25 | Dunnhofer, *MRPyrNet* (2021, MIDL) | 01 / `2021_Dunnhofer_MRPyrNet.pdf` | 7 | rodilla · agregación | Plug-in piramidal para lesiones pequeñas: menisco 0.843 → 0.889, ACL 0.955 → 0.976 sobre MRNet. |
| 26 | Dunnmon, *Cross-modal data programming* (2020, Patterns) | 06 / `2020_Dunnmon_CrossModalDataProgramming.pdf` | 7 | etiquetas | Reglas sobre el informe → etiquetas probabilísticas → modelo de imagen sin informe en test; a 3.75 pts del modelo con etiquetas manuales. |
| 27 | Zhu & Wu, *CSRA residual attention* (2021, ICCV) | 03 / `2021_Zhu_CSRA_ResidualAttention.pdf` | 7 | agregación | Atención por clase de coste despreciable; baseline mínimo de cabeza por etiqueta. |
| 28 | Namiri, *ACL hierarchical staging, 2D vs 3D* (2020, Radiol AI) | 01 / `2020_Namiri_ACL_HierarchicalStaging_2Dvs3D.pdf` | 6.5 | rodilla | Mismos datos: 2D 92 % vs 3D 89 % (P = .27) → apoya 2D/2.5D. |
| 29 | Chavoshi, *LLM label noise biases evaluation* (2025, arXiv) | 06 / `2025_Chavoshi_LLMLabelNoiseEvaluationBias.pdf` | 6.5 | etiquetas | Evaluar con etiquetas LLM sesga a la baja según prevalencia → cuidado con pseudo-AUC en hallazgos raros. |
| 30 | Liu, *ELR* (2020, NeurIPS) | 06 / `2020_Liu_ELR.pdf` | 6.5 | etiquetas | Regularizador anti-memorización del ruido, coste ≈ 0. |
| 31 | Qin, *MobileNetV4* (2024, ECCV) | 02 / `2024_Qin_MobileNetV4.pdf` | 6.5 | backbone | CNN Apache con preentreno IN-12k a 384; MNv4-M @384 = 2.43 GMACs/corte (18× menos que CoAtNet-384). |
| 32 | Siméoni, *DINOv3* (2025, arXiv) | 02 / `2025_Simeoni_DINOv3.pdf` | 6.5 | backbone | ViT-S/16 @224 = 4.33 GMACs (−21 % vs DINOv2-S/14); licencia propia. |
| 33 | Lan, *OrthoDiffusion* (2026, arXiv, preprint) | 01 / `2026_Lan_OrthoDiffusion.pdf` | 6.5 | rodilla · agregación | Módulo MPAE: fusión por plano específica de paciente y etiqueta (macro AUC 90.8); backbone pesado, solo como idea. |
| 34 | Kushol, *MRI scanner manufacturer effects* (2023, Sci Rep) | 08 / `2023_Kushol_ScannerManufacturerEffects.pdf` | 6 | preprocesamiento | Cross-vendor hunde el DL y ComBat no ayuda → mezclar sitios y augment de apariencia. |
| 35 | Mead, *Systematic review knee MRI DL* (2025, Eur Radiol) | 01 / `2025_Mead_SystematicReview_KneeDL.pdf` | 6 | rodilla | 54 estudios, mayoría tarea única → justifica el hueco "multi-hallazgo + eficiente". |
| 36 | Mosbach, *Stability of fine-tuning* (2021, ICLR) | 04 / `2021_Mosbach_StabilityFineTuning.pdf` | 6 | fine-tuning | Colapsos = optimización: LR bajo + warmup, bias correction, más iteraciones. |
| 37 | Lian, *SSF* (2022, NeurIPS) | 04 / `2022_Lian_SSF.pdf` | 6 | fine-tuning | PEFT re-parametrizable también en CNN/CoAtNet, coste cero en inferencia. |
| 38 | Beyer, *FlexiViT* (2023, CVPR) | 05 / `2023_Beyer_FlexiViT.pdf` | 6 | eficiencia | Un solo set de pesos para varios patch sizes → curva AUC-vs-costo de un único modelo. |
| 39 | Liang, *EViT* (2022, ICLR) | 05 / `2022_Liang_EViT.pdf` | 6 | eficiencia | Poda de tokens de fondo; a igual coste permite más resolución (+1 % acc). |
| 40 | Bergström, *When are radiology reports useful for image classifiers?* (2025, arXiv) | 06 / `2025_Bergstrom_WhenReportsUsefulImageClassifiers.pdf` | 6 | etiquetas | Usar el informe durante el fine-tuning (información privilegiada) pesa más que el método de pretraining. |

Fuera del top 40 pero citables: Xie 2025 (eClinicalMedicine, sin PDF, comparador), Atito SB-SSL (01), M3T y
TomoGraphView (03), Huix, RAD-DINO, Baharoon, Wu (02), Tomihari, DoRA, WiSE-FT, Ruffini, Surgical FT (04), H0-mini,
Li 2025, MedPTQ, DinoAtten3D, Selvan (05), GCE/SCE, Hendrycks GLC, CheXpert, Pham, MOSAIC (06), Batra M-SCAN, Shi,
Saeed (07), Mills, Truong, Kasmanoff, BigAug (08).

---

## 3. Huecos en la literatura (candidatos a novedad del paper)

1. **Nadie reporta una curva AUC-vs-costo (frontera de Pareto) en MRI volumétrica multi-etiqueta por estudio.**
   `05` no encontró ningún trabajo revisado por pares con AUC contra cortes/latencia; `01`: "casi ningún trabajo de RM de
   rodilla reporta FLOPs, latencia ni AUC por cómputo" (solo ELNet cuenta parámetros). CoPAS no reporta FLOPs.
   Respaldo: MST, DinoAtten3D, ELNet (solo params/ablaciones sueltas), Mead 2025, Dehghani 2022 (cómo medirlo bien),
   Ozdemir 2026 (protocolo latencia p50/p99).
2. **Primer efficiency track en un reto RSNA / imagen médica volumétrica.** `00`: ningún RSNA previo tuvo efficiency track;
   `07`: no hay Efficiency Prize previo de imagen médica volumétrica en Kaggle. Un paper que mida el frente AUC–runtime con
   la fórmula oficial no tiene precedente. Respaldo: `00` §3, `07` §2.6, Shang 2022 como plantilla ("igualamos al ganador
   con el 10 % de los FLOPs").
3. **Modelo único eficiente destilado (ensemble/FM → student) en rodilla multi-serie, entrenado con etiquetas LLM de
   informes multilingües y sin informe en test.** Respaldo: Beyer KD 2022, H0-mini 2025, Li 2025 (destilación sin GT),
   Feedback Prize/Team Hydrogen (`00` §3), GreenRFM (rodilla, pero zero-shot/VLM), Bergström 2025 (informe como
   información privilegiada). Aviso de `06`: el repo RoroDev2023 ya hace soft labels LLM en este reto → la novedad no puede
   ser solo "soft labels de LLM".
4. **Evaluación de hallazgos sin benchmark: fractura en RM y sinovitis binaria.** `01`: fractura casi sin literatura DL en RM
   (fastMRI+ solo 11 cajas); sinovitis sin benchmark claro. `00`: sinovitis silenciada en ~50 % de positivos expertos;
   fractura 7 % en informes vs 31 % en gold. Respaldo: Zhao fastMRI+ 2022, Chavoshi 2025 (sesgo por prevalencia).
5. **Técnicas de eficiencia sobre el eje de cortes** (ToMe entre cortes vecinos, selección top-k de cortes por atención,
   cascada) sin validar en rodilla. `05` califica ToMe entre cortes como "novedad propia"; `03` idea 6: "no tengo un paper de
   rodilla que lo valide". Respaldo: Bolya ToMe (video), DinoAtten3D, MST, ABMIL.
6. **Soup de folds en lugar de ensemble de folds** en clasificación médica volumétrica, reportado como AUC por FLOP
   (`04`: "novedad publicable para el track de eficiencia"). Respaldo: Wortsman Model Soups, WiSE-FT.

---

## 4. Comparadores directos para el paper

| sistema | datos | números (de los README) | coste reportado |
|---|---|---|---|
| **CoPAS** (Qiu 2024, Nat Commun) | 1,748 pacientes, 5 centros, 12 hallazgos, 5 volúmenes | AUC media 0.812 interna (0.734–0.877); externa 0.721 / 0.726 | ResNet3D-18 por plano; **sin FLOPs** |
| **Vuskov** (2025/26, Eur Radiol) | 3,121 estudios + 448 externos, 23 hallazgos | Int/ext: MCL 0.95/0.88 · Baker 0.93/0.85 · derrame 0.91/0.94 · ACL 0.89/0.89 · men. medial 0.88/0.83 · men. lateral 0.84/0.77 · edema óseo 0.83/0.82; mediana 0.85 → 0.78 | ResNet18 + bloques 3D, 256×256×32 |
| **Xie** (2025, eClinicalMedicine, sin PDF) | 14,962 exámenes, 5 centros, 9 hallazgos | Primarias/secundarias: interna 0.898/0.815 · ext I 0.852/0.744 · ext II 0.812/0.774 | no reportado |
| **MRNet** (Bien 2018) | 1,370 exámenes, 3 tareas | anormal 0.937 · ACL 0.965 · menisco 0.847; ACL externo 0.824 (0.911 reentrenado) | AlexNet × 9 modelos |
| **MST** (Müller-Franzes 2025) | MRNet menisco | 0.85 ± 0.04 (3D ResNet 0.69); congelado 0.79; media 0.82; DINOv2-B 0.83 | 23 M params, 224×224×32 |
| **ELNet** (Tsai 2020) | MRNet | menisco ≈0.904, ACL ≈0.960, anormal ≈0.941 | 53k–211k params, <1 MB |
| **OrthoDiffusion** (2026, preprint) | 10,940 estudios, 8 hallazgos | macro AUC 90.8 (linear probe 87.6; ResNet-18 3D 83.2) | 3 U-Net 3D de difusión (pesado) |
| **GreenRFM** (2026, preprint) | AH-Knee 3,662 estudios, 13 hallazgos | zero-shot 79.0 (MRCLIP 72.9) | 24 GPU-h entrenamiento |
| Kaggle (este reto) | test ~1300 estudios | #1 eficiencia 0.958 pública en ~5–6 min (ResNet pequeño, 1 fold); single model 0.949 LB / ~0.930 gold; notebooks públicos ~0.94 | foro, no verificado por el host |
| Nuestra referencia | exp002 | gold ≈ 0.78 | DINOv2-S 224, 5×16 cortes ≈ 441 GMACs/estudio |

Aviso (`01`): no comparar con cifras "SOTA" de MRNet que re-parten el train (KneeXNet 0.97–0.99, MV2SwimNet).

---

## 5. Ideas consolidadas (deduplicadas), ordenadas por valor esperado para el track de eficiencia

Costo = efecto sobre el runtime de inferencia. Ganancia = esperanza de ΔAUC según la evidencia citada.

| # | idea | costo inf. | ganancia AUC | novedad | papers de respaldo | riesgo |
|---|---|---|---|---|---|---|
| 1 | **Mejores etiquetas**: soft labels (no redondear), consenso de varios LLM, denoising con OOF (quitar \|y−p\| ≥ 0.8), teacher → pseudo-labels → student | = | alta | baja–media | `00` §2.5 (LLM 0.878 vs regex 0.814; "labels > arquitectura"), Dunnmon 2020, GreenRFM, lumbar #2 (+1 %), Feedback Prize Hydrogen | ya lo hace otro equipo (RoroDev2023); gold nunca puede calibrar (regla 6) |
| 2 | **Destilar ensemble/teacher (p. ej. OrthoFoundation-L o ensemble tipo notebooks) a un solo modelo pequeño** | ↓ (vs ensemble) | alta | media | Beyer KD 2022, H0-mini 2025, Li 2025, OrthoFoundation, Hydrogen (−0.003 vs ensemble) | teacher caro de correr offline; OrthoFoundation es preprint |
| 3 | **Recorte a ROI física fija (~130 mm) con localizador barato**, remuestreo a mm/px, bajar a 160–224 px | ↓ (≈ cuadrático en px) | alta | baja | `07` patrón 1 (100 % de top-5 RSNA), Qiu/Xie (crop en menisco), Wang 2024, nnU-Net, notebooks ~0.94 | localizador impreciso → usar jitter de corte/coordenadas (lumbar #1) |
| 4 | **I/O DICOM rápido** (lectura multihilo, cabeceras sin pydicom, `np.frombuffer`, uint8 a GPU, fp16 puro) | ↓↓ (60–70 % del runtime del #1) | ninguna | baja | `00` §2.3 (foro 743374: 40 → <10 min), abdominal #2 (dicomsdl, GPU) | fallo en rerun = sin puntuación → try/except por estudio |
| 5 | **Curva de Pareto cortes × resolución × nº de series**, con selector top-k de cortes por atención | ↓ | = (es medición) / baja | **alta** | MST, DinoAtten3D, Dehghani, Vuskov (32 cortes), Qiu (24), ELNet (1 serie), Sabottke | lesiones pequeñas (fractura, contusión) pueden pedir más px/cortes |
| 6 | **Modelo único + soup de folds** en vez de ensemble de folds | ↓ (1× vs 5×) | media | media | Model Soups, WiSE-FT, `07` §4 (ensemble apenas ayuda), Chris Deotte (`00` foro 735304) | si los folds caen en cuencas distintas el soup rompe; comparar con ensemble de logits |
| 7 | **Backbone más barato, no más grande**: DINOv3-S/16 (−21 % FLOPs), MNv4-M o EffV2-B3 @384; nunca ViT-B/L | ↓ | baja–media | baja | Liu DINOv3 bench, MST (B ≈ S), Doerrich, lumbar #1 (large < base < small), foro 735304 | licencias (DINOv3, ConvNeXt-V2 NC) |
| 8 | **LP-FT + adaptación fusionable** (LoRA q,v en ViT; SSF en CNN) + warmup/LLRD | = | media | baja–media | Kumar 2022, Tomihari 2024, Dutt 2024, MeLo, LoRA, SSF, AnyMC3D, Mosbach | "LoRA gana" puede ser solo "LoRA no colapsa"; desbalance degrada PEFT (Ruffini) |
| 9 | **Cabeza por etiqueta con queries / pooling por query aprendida** (12 queries sobre tokens de cortes y series; matriz hallazgo×serie) | ≈ = | media | baja–media | AnyMC3D (+0.11), ML-Decoder, CSRA, MPAE de OrthoDiffusion, CoPAS | Harvey: la media a menudo empata → ablación obligatoria |
| 10 | **FP16 → INT8-QAT** sobre el modelo final, midiendo AUC por etiqueta + ECE + p50/p99 | ↓ | ninguna (riesgo de pérdida) | baja | Ozdemir 2026, MedPTQ 2026 (2.74× menos latencia), `00` (fp16 puro 2.08× en T4) | INT8-PTQ degrada según arquitectura y se esconde en la media de 12 AUC |
| 11 | **ToMe / EViT, incluido merge entre cortes vecinos** | ↓ | = o leve pérdida | media–alta (entre cortes) | Bolya ToMe, EViT, DynamicViT, A-ViT | cifras de ToMe son en ViT-L/H; en ViT-S @224 hay pocos tokens que fundir (no medido) |
| 12 | **SSL continuado en dominio** sobre ~820k cortes sin etiqueta (backbone pequeño) | = | media | media | OrthoFoundation, RAD-DINO, SB-SSL, RadImageNet (+4.5–4.8 AUC ACL/menisco) | coste de preentreno; ganancia no medida en S |
| 13 | **Informe como información privilegiada** solo en entrenamiento (pérdida auxiliar hacia embedding del informe) | = | media | **alta** | Bergström 2025, GreenRFM | sin precedente en rodilla; encoder de texto multilingüe |
| 14 | **Adaptadores 3D residuales inicializados a cero** en el backbone 2D | ↑ (leve) | media | media | Vuskov 2025, M3T (stem 3D ligero) | aumenta FLOPs; hay que medirlo en la curva |
| 15 | **Losses robustas / curriculum** (ELR, GCE, SCE, CUFIT) | = | baja–media | baja | Liu ELR, Zhang GCE, Wang SCE, Yu CUFIT, Karimi 2020 | en etiquetas raras el filtrado puede tirar positivos reales |
| 16 | **Normalización y robustez**: GN/LN en lo que se añada, clip percentil + z-score por serie, augment de apariencia de sitio | = | baja–media (robustez) | baja | Madaan, ELNet, Mills, Carré, Kushol, BigAug, TorchIO | efecto pequeño frente al domain shift (Mills) |
| 17 | **Enrutado de series por metadatos** (RF/reglas sobre TR, TE, ImageOrientation) y ordenar por posición | ↓ (ms en CPU) | baja | baja | Kasmanoff 2023, Truong 2026, `00` §2.6 (InstanceNumber invertido en >1/3) | SeriesDescription en 12 idiomas no sirve; usar vectores |
| 18 | **Lateralidad antes de flips** (normalizar a "rodilla derecha" o intercambiar etiquetas M/L) | = | protege Medial/Lateral | baja | `08` "Qué NO haría", MRNet/Xie (flip 50 %), lumbar #3 (espejado de lado), `00` (lateralidad falta ~50 %) | detectar lateralidad con ImageOrientationPatient |
| 19 | **Pérdidas auxiliares y entrenamiento en dos pasos**, jitter de ROI | = | baja–media | baja | abdominal #1 (+0.01–0.03), lumbar #1/#4/#5, aneurisma #1 | ninguna en inferencia si se quitan las cabezas auxiliares |
| 20 | **Detail pooling piramidal** para lesiones pequeñas | ↑ (leve) | media (menisco +4.6 pp en MRNet) | baja | MRPyrNet | medido sobre AlexNet/ELNet, no sobre DINOv2 |

Transversal (no son experimentos de AUC): reportar latencia de pared en hardware Kaggle, throughput, params, FLOPs y
energía (Dehghani, Ozdemir, Selvan); pseudo-AUC por etiqueta con IC por bootstrap (Chavoshi); ablación media vs atención
(Harvey).

---

## 6. Contradicciones o tensiones entre fuentes

- **Congelar vs afinar el backbone.** AnyMC3D: backbone congelado + LoRA + query pooling gana; Harvey: con encoder congelado,
  la media basta. Pero MST: congelado cae 0.85 → 0.79 en rodilla (0.62 en mama); Wu 2025: end-to-end > linear en 12/12;
  `02` y `05` desaconsejan cachear features congeladas. En cambio, CUFIT y Ruffini dicen que el LP es lo más robusto al
  ruido y en pocos datos. Lectura: "congelado + LoRA" ≠ "congelado + linear"; hay que medirlo.
- **Escalar el backbone.** MST (B no mejora a S), benchmark DINOv3 (S ≈ B ≈ L), lumbar #1 (large < base < small), lumbar #5 y
  foro 735304 dicen que no ayuda. Pero los notebooks públicos a ~0.94 usan CoAtNet-384 (~7.8× nuestro coste), y
  OrthoFoundation-L gana +3–4 pp sobre DINOv2/DINOv3 (por dominio, no por tamaño).
- **Resolución.** Doerrich y Wu: satura (64–224 px marginal); foro: 224/288 es el punto dulce y >288 no ayuda. Pero los
  notebooks usan 384 px + recorte y Sabottke dice que los hallazgos sutiles piden más px; `02` advierte que MedMNIST no
  tiene lesiones pequeñas.
- **Atención vs media sobre cortes.** AnyMC3D: query pooling > media/LSTM/Transformer; MST: media 0.82 < transformer 0.85;
  lumbar #1: attention-MIL 0.37 → 0.35. Harvey: media ≥ ABMIL/TransMIL en 4/6. Darragh (cervical #3): transformers no
  mejoraron sobre RNN+atención; SonySpine: 2D+LSTM no funcionó.
- **2D vs 3D.** Namiri, MST, AnyMC3D, cervical #1, lumbar #3 y brain #1: 3D puro no gana. Pero Ke 2023: 3D preentrenado
  en video supera a 2D en su setup, y CSN (cervical #4) funcionó; Vuskov usa un híbrido 2D + bloques 3D.
- **Flips medial/lateral.** MRNet y Xie usan flip horizontal al 50 % sin tratar lateralidad; `08` lo desaconseja porque espeja
  Medial↔Lateral en coronal/axial; lumbar #3 sí espeja, pero con modelos por lado. Además la lateralidad falta en ~50 % de
  estudios (`00`).
- **Qué hacer con "incierto/no mencionado".** Kreutzer: la política no cambia el AUC; CheXpert: depende del hallazgo; `00`:
  sinovitis silenciada en ~50 % de positivos y fractura 7 % en informes vs 31 % en gold → al menos para esas dos, "no
  mencionado" ≠ negativo.
- **Modelo barato vs AUC mínima.** `07` §0: un 0.90 en 30 min vence a un ensemble 0.94 en 9 h. `00` §2.1: con el LB real, AUC
  < ~0.945 no gana el premio aunque corra en 1 min. No se contradicen en la fórmula, sí en la estrategia: el techo lo marca
  la AUC de los rivales.
- **CPU vs GPU.** `07` dejó abierto si el track es solo CPU; `00` verificó que la regla de RSNA Knee no lo exige (a diferencia
  de 8/9 competencias previas), aunque sin aclaración explícita del host.
- **Uso de datos de confianza.** GLC, Meta-Weight-Net y temperature scaling necesitan un set limpio; nuestra regla 6 prohíbe
  usar los 58 gold para ajustar nada (`06`).
- **INT8.** MedPTQ: 2.74× menos latencia con Dice casi idéntico (segmentación, TensorRT). Ozdemir: INT8-PTQ puede degradar AUC
  y calibración según arquitectura en clasificación.
- **Licencias.** `02`: ConvNeXt-V2 (CC-BY-NC), EUPE (NC) y DINOv3 (licencia propia) pueden chocar con el premio; `00`: la
  licencia ganadora exigida es CC-BY-NC 4.0 y MRNet/KneeMRI (ND) son probablemente problemáticos para pesos publicados.
  Sin resolución del host.
