# 02 · Backbones 2D eficientes para cortes de MRI de rodilla

Pregunta: ¿qué encoder 2D por corte da el mejor AUC por unidad de cómputo para nuestro pipeline
(5 series × 16 cortes → atención cortes→series), hoy DINOv2-small 224 px?

Convención de costo: **GMACs por corte** medidos localmente con `timm 1.0.28` + `torch.utils.flop_counter`
(solo conv/matmul, `num_classes=0`, entrada 3 canales). **Costo por estudio ≈ 80 × GMACs/corte**
(5 series × 16 cortes). Ej.: referencia DINOv2-S/14 @224 = 5.51 × 80 ≈ **441 GMACs/estudio**.
Los FLOPs del paper original pueden diferir ligeramente (p. ej. MNv4 cuenta con cabeza).

---

## 1. Papers rankeados

Puntaje = relevancia × eficiencia × novedad × solidez (1–10).

| rank | paper (año, venue) | PDF + enlace | puntaje | por qué nos sirve | números clave |
|---|---|---|---|---|---|
| 1 | Müller-Franzes et al., *Medical Slice Transformer (MST): improved diagnosis and explainability on 3D medical images with DINOv2* (2025, Scientific Reports 15:23979) | sí `2025_MullerFranzes_MedicalSliceTransformer.pdf` · [arXiv 2411.15802](https://arxiv.org/abs/2411.15802) · [código](https://github.com/mueller-franzes/MST) | 10 | Es casi nuestra arquitectura (DINOv2 2D por corte + transformer sobre cortes) en **MRNet (rodilla)**. Sus ablaciones responden directamente qué backbone usar. | Menisco MRNet: MST-DINOv2-S **0.85±0.04** vs 3D ResNet 0.69 (P=0.001). Ablación: DINOv2-**Base 0.83** (no mejora a Small), **registers 0.82** (no ayuda), **backbone congelado 0.79** (y 0.62/0.66 en mama/tórax), ResNet 2D ImageNet 0.78. Entrada 224×224×32. |
| 2 | Liu, Chen, Shi et al., *Does DINOv3 set a new medical vision standard?* (2025, arXiv, Imperial/TUM) | sí `2025_Liu_DINOv3MedicalBenchmark.pdf` · [arXiv 2509.06467](https://arxiv.org/abs/2509.06467) | 9 | Único benchmark sistemático de DINOv3 S/B/L en médico 2D/3D. Muestra que **escalar tamaño/resolución no paga** en médico → argumento directo para el track de eficiencia. | Linear probe @256, NIH-14 AUC: S 0.7788 / B 0.7833 / L 0.7865 (BiomedCLIP 0.7771). RSNA-Pneumonia: S 0.8667 / B 0.8666 / L 0.8708. AUC en NIH-14 tope a 512 px y luego baja. Falla en WSI/EM/PET. |
| 3 | Doerrich, Di Salvo, Brockmann, Ledig, *Rethinking model prototyping through the MedMNIST+ dataset collection* (2025, Scientific Reports 15:7669) | sí `2025_Doerrich_MedMNISTplus.pdf` · [arXiv 2404.15786](https://arxiv.org/abs/2404.15786) · [Sci Rep](https://www.nature.com/articles/s41598-025-92156-9) | 8 | 10 backbones × 4 resoluciones (28–224) × 3 esquemas (end-to-end, linear, kNN) × 12 datasets, con tests pareados. Evidencia sólida de que **CNNs siguen competitivas** y de que **más resolución satura**. | ACC mediana: DINO ViT-B/16 **86.1** (mejor), DenseNet-121 82.6, ViT-B/16 82.5, CLIP 80.3, EVA-02 80.9, EfficientNet-B4 76.3, **SAM 69.6** (peor). Resolución: sin mejora consistente > umbral. |
| 4 | Pérez-García et al., *RAD-DINO: Exploring scalable medical image encoders beyond text supervision* (2025, Nature Machine Intelligence) | sí `2025_PerezGarcia_RAD-DINO.pdf` · [arXiv 2401.10815](https://arxiv.org/abs/2401.10815) · [HF microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino) (MIT) | 7 | Demuestra que **DINOv2 continuado en imagen médica sin texto** supera a CLIP médicos (BiomedCLIP, CheXzero). Receta que podríamos replicar a pequeña escala sobre nuestros 820k DICOM. Pero pesos solo de CXR, ViT-B. | VinDr-CXR AUPRC agregado: RAD-DINO 52.8 vs MRM 51.3, CheXzero 40.0, BiomedCLIP 35.9. ViT-B/14 @518 ≈ 117 GMACs/imagen (caro). |
| 5 | Yu, Wang, Yuan et al., *A multimodal vision foundation model for generalizable knee pathology (OrthoFoundation)* (2026, arXiv) | sí `2026_Yu_OrthoFoundation.pdf` · [arXiv 2601.18250](https://arxiv.org/abs/2601.18250) · [GitHub](https://github.com/ytrsk/OrthoFoundation) (MIT, checkpoint `OrthoFoudation-L.pth`) | 7 | **FM específico de rodilla** (1.2M imágenes RX+MRI; OAI + fastMRI + hospital), DINOv3-L continuado. Mismas etiquetas que nosotros (ACL, PCL, meniscos, colaterales, bone bruise). Candidato a **teacher** para destilar, no a backbone de producción. | 10 tareas MRI binarias: 1.º en todas; PCL AUC 84.3 (+4 pp vs In-ViT). ACL 97.7 AUC con 50% de etiquetas. +3–4 pp vs DINOv2/DINOv3 en transferencia a cadera/hombro/tobillo. ViT-L/16: 303M params, **60.9 GMACs @224** (11× DINOv2-S). Preprint, sin revisión por pares. |
| 6 | Pales Huix et al., *Are natural domain foundation models useful for medical image classification?* (2024, WACV) | sí `2024_Huix_NaturalFMsForMedicalClassification.pdf` · [arXiv 2310.19522](https://arxiv.org/abs/2310.19522) | 7 | Compara SAM, SEEM, DINOv2, BLIP, OpenCLIP vs ImageNet en 4 datasets médicos. Justifica mantener DINO y **descartar encoders de SAM/segmentación** para clasificar. | "DINOv2 consistently outperforms ImageNet pretraining"; SAM/SEEM/BLIP/CLIP no superan de forma consistente la línea base ImageNet. |
| 7 | Baharoon et al., *Evaluating general purpose vision foundation models for medical image analysis: DINOv2 on radiology benchmarks* (2023/24, arXiv v4) | sí `2023_Baharoon_DINOv2RadiologyBenchmarks.pdf` · [arXiv 2312.02366](https://arxiv.org/abs/2312.02366) | 6 | >200 evaluaciones de DINOv2 (S/B/L/g) en RX, CT y **MRI** con kNN, few-shot, linear, fine-tune y PEFT. Útil para régimen de pocas etiquetas (58 gold). | DINOv2 ≥ supervisados/SSL/weakly-sup en clasificación RX/CT/MRI; incluye Brain-Tumor MRI 2D. |
| 8 | Wu & Papież, *Rethinking foundation models for medical image classification through a benchmark study on MedMNIST* (2025, MIDL 2025 submission) | sí `2025_Wu_FMsMedMNISTBenchmark.pdf` · [arXiv 2501.14685](https://arxiv.org/abs/2501.14685) | 6 | 12 modelos (4 CNN + 8 ViT: CLIP, EVA-02, DINO, DINOv2…) end-to-end vs linear, tamaño de imagen y de train. | **End-to-end > linear probing en los 12 datasets**; DINO/DINOv2 lideran el grupo "feature extraction". Diferencias 64/128/224 px **marginales**. |
| 9 | Dong, Chen, Gu, Konz, Mazurowski, *MRI-CORE: a foundation model for MRI* (2025, arXiv) | sí `2025_Dong_MRI-CORE.pdf` · [arXiv 2506.12186](https://arxiv.org/abs/2506.12186) · [GitHub](https://github.com/mazurowski-lab/mri_foundation) (Apache-2.0) | 5 | FM **MRI-específico 2D**: 6.9M cortes, 116k volúmenes, 18 regiones (incluye rodilla/KNEEMRI), DINOv2 inicializado con SAM ViT-B. Pero entrada 1024 px y solo comparado con SAM/MedSAM. | Mejora en 13 tareas de segmentación few-shot; linear probe > SAM/MedSAM. SAM ViT-B @1024 = **486 GMACs/corte** (88× DINOv2-S). Inviable en eficiencia salvo a 256 px (34.5 GMACs). |
| 10 | Siméoni et al., *DINOv3* (2025, arXiv, Meta) | sí `2025_Simeoni_DINOv3.pdf` · [arXiv 2508.10104](https://arxiv.org/abs/2508.10104) | 6 | Familia destilada: ViT-S/16, S+/16, B/16 y **ConvNeXt T/S/B destilados** del 7B. Patch 16 → menos tokens que DINOv2/14 a igual resolución. | ViT-S/16 @224: 21.6M, **4.33 GMACs** (vs 5.51 DINOv2-S/14). ConvNeXt-T DINOv3: 27.8M, 4.45 GMACs. Licencia propia "DINOv3 License". |
| 11 | Woo et al., *ConvNeXt V2: co-designing and scaling ConvNets with masked autoencoders* (2023, CVPR) | sí `2023_Woo_ConvNeXtV2.pdf` · [arXiv 2301.00808](https://arxiv.org/abs/2301.00808) | 6 | Familia atto→huge con preentreno FCMAE; los tamaños atto/femto/pico/nano son el frente bajo de Pareto CNN. | Atto 3.7M, 0.55 GFLOPs, 76.7% IN-1k. **Pesos CC-BY-NC-4.0** (riesgo para premio Kaggle). |
| 12 | Qin et al., *MobileNetV4: universal models for the mobile ecosystem* (2024, ECCV) | sí `2024_Qin_MobileNetV4.pdf` · [arXiv 2404.10518](https://arxiv.org/abs/2404.10518) | 6 | CNN/híbrido más eficiente con pesos Apache-2.0 en timm, incluidos preentrenos **IN-12k a 384 px**. | Conv-S 73.8% (3.8M, 0.2G MACs); Conv-M 79.9% (9.2M, 1.0G); Hybrid-M 80.7%; Conv-L 82.9%. |
| 13 | Wang et al., *RepViT: revisiting mobile CNN from ViT perspective* (2024, CVPR) | sí `2024_Wang_RepViT.pdf` · [arXiv 2307.09283](https://arxiv.org/abs/2307.09283) | 5 | CNN reparametrizable, muy rápida en GPU/CPU; alternativa Apache a ConvNeXtV2-pico. | M0.9 78.7/79.1% (5.1M, 0.8G); M1.1 80.7/81.2% (8.2M, 1.3G); M1.5 82.3/82.5%; M2.3 83.3/83.7%. |
| 14 | Codella et al., *MedImageInsight: an open-source embedding model for general domain medical imaging* (2024, arXiv, Microsoft) | sí `2024_Codella_MedImageInsight.pdf` · [arXiv 2410.06542](https://arxiv.org/abs/2410.06542) | 4 | Embedding médico generalista (incluye MRI). Encoder DaViT **360M** + texto 252M: demasiado caro para eficiencia; útil solo como referencia. | Imagen 360M params; UniCL. |
| 15 | Mei et al., *RadImageNet: an open radiologic deep learning research dataset for effective transfer learning* (2022, Radiology: AI) | **no** (bloqueo anti-bot) · [RSNA](https://pubs.rsna.org/doi/full/10.1148/ryai.210315) · [PMC9530758](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/) · [pesos](https://github.com/BMEII-AI/RadImageNet) | 6 | 1.35M imágenes CT/MRI/US; incluye **rodilla MRI (ACL, menisco)**. Evidencia directa de que preentreno radiológico ayuda en datasets pequeños de rodilla. Arquitecturas viejas (ResNet50, DenseNet121, InceptionV3). | ACL MRI +4.8% AUC y menisco +4.5% AUC vs ImageNet (P<.001); mejoras 0.9–9.4% AUC en 8 tareas. |

### Solo enlace (arquitecturas / FMs de referencia, no descargados para no pasar de ~15)
- DINOv2 — Oquab et al., TMLR 2024 — [arXiv 2304.07193](https://arxiv.org/abs/2304.07193) (Apache-2.0).
- SigLIP 2 — Tschannen et al., 2025 — [arXiv 2502.14786](https://arxiv.org/abs/2502.14786). CLIP — Radford et al., ICML 2021 — [arXiv 2103.00020](https://arxiv.org/abs/2103.00020). EVA-02 — Fang et al., 2023 — [arXiv 2303.11331](https://arxiv.org/abs/2303.11331).
- BiomedCLIP — Zhang et al., NEJM AI 2025 — [arXiv 2303.00915](https://arxiv.org/abs/2303.00915) · [HF](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) (MIT).
- MedSAM — Ma et al., Nature Communications 2024 — [arXiv 2304.12306](https://arxiv.org/abs/2304.12306). BiomedParse — Zhao et al., Nature Methods 2025 — [arXiv 2405.12971](https://arxiv.org/abs/2405.12971). (Encoders de segmentación: ver Huix y Doerrich, SAM rinde mal en clasificación.)
- EfficientNetV2 — Tan & Le, ICML 2021 — [arXiv 2104.00298](https://arxiv.org/abs/2104.00298).
- EfficientViT (MIT, `efficientvit_b*/l*`) — Cai et al., ICCV 2023 — [arXiv 2205.14756](https://arxiv.org/abs/2205.14756). EfficientViT (Microsoft, `efficientvit_m*`) — Liu et al., CVPR 2023 — [arXiv 2305.07027](https://arxiv.org/abs/2305.07027).
- FastViT — Vasu et al., ICCV 2023 — [arXiv 2303.14189](https://arxiv.org/abs/2303.14189) (licencia Apple "other").
- TinyViT — Wu et al., ECCV 2022 — [arXiv 2207.10666](https://arxiv.org/abs/2207.10666).
- CoAtNet — Dai et al., NeurIPS 2021 — [arXiv 2106.04803](https://arxiv.org/abs/2106.04803). MaxViT — Tu et al., ECCV 2022 — [arXiv 2204.01697](https://arxiv.org/abs/2204.01697). Next-ViT — Li et al., 2022 — [arXiv 2207.05501](https://arxiv.org/abs/2207.05501).
- EUPE (Efficient Universal Perception Encoder) — arXiv 2603.22387 según la model card de `timm/vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m` (no verificado más allá de la card).

---

## 2. Tabla de candidatos: costo y disponibilidad

Params y GMACs medidos aquí (timm 1.0.28, sin cabeza). "Est." = GMACs/estudio a 224 con 80 cortes.
Licencia = tag de la model card en HF (importante: el premio de Kaggle suele exigir licencia abierta; **revisar reglas**).

### (a) Foundation models generales
| modelo timm / HF | params | GMACs @224 | GMACs @384 | est. @224 | licencia | nota |
|---|---|---|---|---|---|---|
| `vit_small_patch14_dinov2.lvd142m` (**referencia**) | 21.6M | 5.51 | 15.67 | 441 | Apache-2.0 | MST: S ≈ B en rodilla |
| `vit_small_patch14_reg4_dinov2.lvd142m` | 21.6M | 5.60 | 15.75 | 448 | Apache-2.0 | registers no ayudaron en MST |
| `vit_base_patch14_dinov2.lvd142m` | 85.7M | 21.94 | 62.33 | 1755 | Apache-2.0 | no mejora a S (MST) |
| `vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m` | 5.5M | 1.10 | 3.17 | 88 | FAIR non-commercial | ViT diminuto destilado; licencia NC |
| `vit_small_patch16_dinov3.lvd1689m` | 21.6M | **4.33** | 12.51 | 346 | DINOv3 License | −21% costo vs referencia |
| `vit_small_plus_patch16_dinov3.lvd1689m` | 28.7M | 5.75 | 16.62 | 460 | DINOv3 License | |
| `vit_base_patch16_dinov3.lvd1689m` | 85.6M | 17.19 | 49.69 | 1375 | DINOv3 License | S≈B≈L en benchmark médico |
| `convnext_tiny.dinov3_lvd1689m` | 27.8M | 4.45 | 13.09 | 356 | DINOv3 License | CNN destilada de DINOv3-7B; escala bien a 384 |
| `convnext_small.dinov3_lvd1689m` | 49.5M | 8.68 | 25.52 | 694 | DINOv3 License | |
| `vit_base_patch16_siglip_256.v2_webli` (SigLIP2) | 92.9M | 17.00 | 49.95 | 1360 | Apache-2.0 | texto-imagen; no hay S en timm |
| `vit_base_patch16_clip_224.openai` | 85.8M | 16.85 | 49.35 | 1348 | — | CLIP < DINO en médico |
| `eva02_tiny_patch14_224.mim_in22k` / `eva02_small_…` | 5.5M / 21.6M | 1.39 / 5.51 | 3.96 / 15.67 | 111 / 441 | MIT | |

### (b) Foundation models médicos
| modelo | arquitectura | params | GMACs | pesos | comentario |
|---|---|---|---|---|---|
| RadImageNet | ResNet50 / DenseNet121 / InceptionV3 | 23.5M / 7.0M | 4.09 / 2.83 @224 | [GitHub BMEII-AI](https://github.com/BMEII-AI/RadImageNet) (no timm) | incluye rodilla MRI; arquitectura vieja |
| RAD-DINO | ViT-B/14 @518 | 86.6M | 117 @518 / ~22 @224 | [HF microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino) (MIT) | solo CXR |
| BiomedCLIP | ViT-B/16 @224 | ~86M | ~16.8 | [HF](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) (MIT) | PubMed; ≤ DINOv3 en benchmark |
| MedSAM / SAM-Med2D | SAM ViT-B @1024 | 89.7M | 486 @1024 | GitHub | encoder de segmentación; malo para clasificar |
| MRI-CORE | SAM ViT-B + DINOv2 @1024 | 89.7M | 486 @1024 (34.5 @256) | [Google Drive vía GitHub](https://github.com/mazurowski-lab/mri_foundation) (Apache-2.0) | MRI-específico, caro |
| OrthoFoundation-L | DINOv3 ViT-L/16 | 303M | 60.9 @224 / 176 @384 | [GitHub ytrsk](https://github.com/ytrsk/OrthoFoundation) (MIT, pero base DINOv3) | **rodilla**; mejor como teacher |
| MedImageInsight | DaViT | 360M (imagen) | — | Azure / HF comunidad | demasiado grande |
| BiomedParse | SAM-like + texto | grande | — | HF microsoft | segmentación; descartado |

### (c) Arquitecturas eficientes (ImageNet)
| modelo timm | params | GMACs @224 | GMACs @384 | est. @224 | licencia |
|---|---|---|---|---|---|
| `mobilenetv4_conv_small.e2400_r224_in1k` | 2.5M | 0.18 | 0.54 | 14 | Apache-2.0 |
| `mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k` | 8.4M | 0.83 | 2.43 | 66 | Apache-2.0 |
| `mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k` | 9.8M | 0.97 | 2.97 | 78 | Apache-2.0 |
| `mobilenetv4_conv_large.e600_r384_in1k` | 31.3M | 2.17 | 6.37 | 174 | Apache-2.0 |
| `convnextv2_atto.fcmae_ft_in1k` | 3.4M | 0.55 | 1.61 | 44 | **CC-BY-NC-4.0** |
| `convnextv2_femto.fcmae_ft_in1k` | 4.8M | 0.78 | 2.29 | 62 | CC-BY-NC-4.0 |
| `convnextv2_pico.fcmae_ft_in1k` | 8.6M | 1.37 | 4.02 | 110 | CC-BY-NC-4.0 |
| `convnextv2_nano.fcmae_ft_in22k_in1k_384` | 15.0M | 2.44 | 7.18 | 195 | CC-BY-NC-4.0 |
| `convnextv2_tiny.fcmae_ft_in22k_in1k_384` | 27.9M | 4.45 | 13.09 | 356 | CC-BY-NC-4.0 |
| `efficientnet_b0.ra4_e3600_r224_in1k` | 4.0M | 0.38 | 1.13 | 30 | Apache-2.0 |
| `efficientnet_b3.ra2_in1k` | 10.7M | 0.96 | 2.82 | 77 | Apache-2.0 |
| `tf_efficientnetv2_b0.in1k` | 5.9M | 0.72 | 2.10 | 58 | Apache-2.0 |
| `tf_efficientnetv2_b3.in21k_ft_in1k` | 12.8M | 1.61 | 4.72 | 129 | Apache-2.0 |
| `tf_efficientnetv2_s.in21k_ft_in1k` | 20.2M | 2.85 | 8.36 | 228 | Apache-2.0 |
| `efficientvit_b1.r224_in1k` / `b2` | 7.5M / 21.8M | 0.52 / 1.58 | 1.52 / 4.64 | 42 / 126 | Apache-2.0 |
| `efficientvit_m5.r224_in1k` | 12.1M | 0.52 | 1.49 | 42 | Apache-2.0 |
| `repvit_m0_9.dist_450e_in1k` / `m1_1` / `m1_5` / `m2_3` | 4.7 / 7.8 / 13.6 / 22.4M | 0.82 / 1.34 / 2.28 / 4.52 | 2.40 / 3.93 / 6.69 / 13.28 | 66 / 107 / 182 / 362 | Apache-2.0 |
| `fastvit_t12.apple_dist_in1k` / `sa12` / `sa24` | 6.5 / 10.6 / 20.5M | 1.07 / 1.48 / 2.87 | 3.14 / 4.35 / 8.44 | 86 / 118 / 230 | Apple "other" |
| `tiny_vit_5m_224.dist_in22k_ft_in1k` / `11m` / `21m` | 5.1 / 10.5 / 20.6M | 1.25 / 2.02 / 4.25 | 3.95 / 6.48 / 13.59 | 100 / 162 / 340 | Apache-2.0 |
| `coatnet_nano_rw_224.sw_in1k` | 14.6M | 2.27 | 6.66 | 182 | Apache-2.0 |
| `coatnet_0_rw_224.sw_in1k` | 26.7M | 4.21 | 12.37 | 337 | Apache-2.0 |
| `coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k` | 41.0M | 7.66 | 22.52 | 613 | Apache-2.0 |
| `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (**notebooks públicos**) | 72.9M | 14.58 | **42.84** | 3427 @384 | Apache-2.0 |
| `maxvit_nano_rw_256.sw_in1k` / `maxvit_tiny_tf_224.in1k` | 14.9 / 30.4M | 3.23 / 5.55 | 9.50 / 17.39 | 258 / 444 | Apache-2.0 |
| `nextvit_small.bd_ssld_6m_in1k` | 30.7M | 5.76 | 16.93 | 461 | Apache-2.0 |

Lectura rápida: el CoAtNet-384 de los notebooks públicos cuesta **~7.8× nuestra referencia por estudio**
(3427 vs 441 GMACs). Un MobileNetV4-M @384 cuesta 2.43 GMACs/corte (**18× menos que CoAtNet-384** a igual resolución).

---

## 3. Ideas para nosotros
- **Cambiar DINOv2-S/14 por DINOv3-S/16 (o ConvNeXt-T DINOv3) es la apuesta más barata**: −21% FLOPs a 224 y
  mejor base SSL; MST (#1) y el benchmark DINOv3 (#2) indican que S ≈ B ≈ L en médico, así que no subir de tamaño.
- **Subir resolución con un backbone barato en vez de subir tamaño**: MobileNetV4-M o EfficientNetV2-B3 @384 cuestan
  menos que DINOv2-S @224 (2.43 / 4.72 vs 5.51 GMACs). Los notebooks públicos ganan con 384 px + recorte; Doerrich (#3)
  y Wu (#8) dicen que la resolución satura, pero sus imágenes no tienen lesiones pequeñas como un menisco → hay que medirlo.
- **Fine-tune siempre, nunca congelado**: MST (#1) pierde 0.06 AUC en rodilla (0.85→0.79) y ~0.3 en mama/tórax con DINOv2
  congelado; Wu (#8) confirma end-to-end > linear en 12/12. Cachear embeddings congelados para ahorrar cómputo es mala idea.
- **Destilación de OrthoFoundation-L (#5) → estudiante pequeño** (DINOv3-S/16 o MobileNetV4-M): teacher específico de rodilla
  con nuestras mismas patologías, estudiante 14–70× más barato. Novedad publicable ("knee-FM distilled to an efficient
  multi-series student"). Riesgo: preprint sin peer review, ViT-L caro de correr offline para generar targets.
- **Preentreno SSL continuado estilo RAD-DINO (#4) sobre nuestros ~820k cortes no etiquetados** con un backbone pequeño
  (DINOv3-S o ConvNeXt-pico): usa datos que ya tenemos y que no requieren etiquetas; RadImageNet (#15) muestra +4.5–4.8 AUC
  en ACL/menisco por preentreno radiológico. Riesgo: cómputo de preentreno (fuera del notebook de 9 h, pero se permite).
- **Licencias**: ConvNeXt-V2 (CC-BY-NC), EUPE (NC) y DINOv3 (licencia propia) pueden chocar con las reglas de premio
  de Kaggle. Para el track de eficiencia priorizar Apache/MIT (DINOv2, MobileNetV4, EfficientNetV2, RepViT, CoAtNet, TinyViT).

## 4. Qué NO haría
- **No usar encoders de SAM/MedSAM/MRI-CORE/BiomedParse** como backbone de clasificación: 486 GMACs/corte a 1024 px y
  SAM es el peor en MedMNIST+ (#3) y Huix (#6).
- **No escalar a ViT-B/L** (DINOv2-B, DINOv3-B/L, SigLIP2-B, BiomedCLIP): 3–11× costo sin ganancia en MST (#1) ni en el
  benchmark DINOv3 (#2). Tampoco añadir registers.
- **No gastar tiempo en CLIP/SigLIP para visión pura**: sin texto en test, y CLIP < DINO en #3, #6, #8.

---

## 5. Lista corta para el barrido (paso 1)

Un experimento por backbone contra el padre (exp002, DINOv2-S @224), mismo split y pipeline; solo cambia el encoder
(una variable). Ordenados por costo; cubren el frente de Pareto de ~0.8 a ~43 GMACs/corte.

| # | nombre exacto (timm) | params | GMACs/corte | resolución | papel en el barrido | licencia |
|---|---|---|---|---|---|---|
| 1 | `mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k` | 8.4M | 0.83 @224 / 2.43 @384 | 224 y 384 | suelo de costo, CNN Apache con preentreno IN-12k a 384 | Apache-2.0 |
| 2 | `vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m` | 5.5M | 1.10 @224 | 224 | ¿aguanta un ViT DINOv3-family diminuto? | FAIR NC ⚠ |
| 3 | `tf_efficientnetv2_b3.in21k_ft_in1k` | 12.8M | 1.61 @224 / 4.72 @384 | 384 | CNN clásica de Kaggle médico, alta resolución barata | Apache-2.0 |
| 4 | `convnextv2_nano.fcmae_ft_in22k_in1k_384` | 15.0M | 2.44 @224 / 7.18 @384 | 384 | ConvNeXt-V2 FCMAE (familia atto–nano) | CC-BY-NC ⚠ |
| 5 | `vit_small_patch16_dinov3.lvd1689m` | 21.6M | 4.33 @224 | 224 | reemplazo directo de la referencia, −21% costo | DINOv3 License ⚠ |
| 6 | `vit_small_patch14_dinov2.lvd142m` | 21.6M | 5.51 @224 | 224 | **control** = padre exp002 | Apache-2.0 |
| 7 | `convnext_tiny.dinov3_lvd1689m` | 27.8M | 4.45 @224 / 13.09 @384 | 384 | SSL fuerte + CNN que escala a 384 | DINOv3 License ⚠ |
| 8 | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` | 72.9M | 42.84 @384 | 384 | **techo**: lo que usan los notebooks públicos (~0.94) | Apache-2.0 |

Sugerencia de orden: 5 → 1 → 7 → 3 (más señal por hora), luego 8 como techo de referencia, 2 y 4 si 1 y 3 prometen.
Candidatos de reserva Apache: `repvit_m1_1.dist_450e_in1k` (1.34 GMACs), `tiny_vit_11m_224.dist_in22k_ft_in1k` (2.02),
`coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k` (7.66). Los FMs médicos (OrthoFoundation-L, RAD-DINO) no entran en el barrido
de backbone de producción; entran como teacher / preentreno en experimentos aparte.

Notas prácticas: DINOv2 usa patch 14 → a "384" usar 378 o 392. ViTs de timm aceptan `img_size=` al crear; CoAtNet/MaxViT
tienen resolución fija por checkpoint. Todos los pesos se pueden bajar de HF y subir como Kaggle Dataset (el notebook es offline).
