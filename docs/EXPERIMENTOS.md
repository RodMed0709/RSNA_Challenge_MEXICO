# Registro de experimentos

**Referencia actual: exp002_dinov2s_ref** (5 folds: OOF pseudo 0.802, gold 0.788; fold 0: pseudo 0.827, gold 0.777).

Una fila por experimento. `nuevo_experimento()` agrega la fila con estado `spec`; su autor la completa en el verdict.
Estados: `spec`, `running`, `worked`, `didnt`, `killed`. Los números salen de `results/expNNN_tag/metrics.json`.

| exp | tag | autor | padre | variable cambiada | estado | pseudo f0 | gold | OOF 5f | aprendizaje |
|---|---|---|---|---|---|---|---|---|---|
| exp001 | preparacion | rodrigo | — | exploración, cache v1_256, etiquetas Qwen v1, split v1 | worked | — | — | — | Sin IDs de paciente; el idioma del informe aproxima el hospital; "no mencionado" quedó como 0.5 en etiquetas v1 |
| exp002 | dinov2s_ref | rodrigo | — | referencia DINOv2-small sobre split_v1 | worked | 0.827 | 0.788 | 0.802 | fold 1 inestable (0.738); el mejor checkpoint es la última época en 4/5 folds: le faltan épocas |
| exp003 | mnv4m | rodrigo | exp002 | paso 1: backbone mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k | killed | 0.743 | 0.701 (f0) | — | −86 % GMACs pero no aprende con la receta de DINOv2 (lr backbone 1e-5); gold −0.076 (Δ pseudo −0.083) |
| exp004 | effv2b3 | rodrigo | exp002 | paso 1: backbone tf_efficientnetv2_b3.in21k_ft_in1k | killed | 0.771 | 0.707 (f0) | — | −74 % GMACs y aun así más lento en A100; gold −0.070 (Δ pseudo −0.056) |
| exp005 | cnxv2n | rodrigo | exp002 | paso 1: backbone convnextv2_nano.fcmae_ft_in22k_in1k | didnt | 0.802 | 0.757 (f0) | — | mejor CNN pura, gold justo dentro del veto (−0.019); más lenta que el padre; pesos CC-BY-NC (Δ pseudo −0.025) |
| exp006 | dinov3s | rodrigo | exp002 | paso 1: backbone vit_small_patch16_dinov3.lvd1689m | killed | 0.779 | 0.745 (f0) | — | mismo tamaño que DINOv2-S y peor; gold −0.031; corrió en RTX 4090 (Δ pseudo −0.047) |
| exp007 | cnxt_dinov3 | rodrigo | exp002 | paso 1: backbone convnext_tiny.dinov3_lvd1689m | didnt | 0.808 | 0.767 (f0) | — | 2.º del barrido; −27 % GMACs pero más lento (18.9 vs 14.6 ms) (Δ pseudo −0.018) |
| exp008 | coatnet2 | rodrigo | exp002 | paso 1: backbone coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | didnt | 0.814 | 0.773 (f0) | — | el más cercano al padre pero ×2.45 GMACs y ×4.5 ms: dominado en eficiencia (Δ pseudo −0.012) |
| exp009 | preparacion_v2 | rodrigo | exp001 | paso 3: metadatos DICOM v2 (plano, secuencia, lateralidad, slots) y cache v2 | worked | — | — | — | meta_v2: plano IOP 99.3 %, lado 98.6 % de estudios con 97.0 % vs informe (tag 96.6, texto 98.6, geometría 97.2); slot 1 SAG líquido FS 94.3 % (<95 %: el hospital alemán no hace SAG FS); cache v2_05mm completo: 20,790 series, 0 errores, 68.0 GB |
| exp025 | perfil_raptor | rodrigo | — | perfilado por etapa de la inferencia pública Raptor/0.949 (CoAtNet-2 + TTA espejo), 50 estudios, 112 y 4 hilos | worked | — | 0.929 (sanity, 25 gold, inflado) | — | lectura+decode+norm = 41 % (study_lut, 4 h; 63 % con pydicom); serie 0.79 s/estudio; en Kaggle manda el forward T4 (TTA = mitad); channels_last −12 % |
| exp026 | t4_variantes | rodrigo | exp025 | inferencia Raptor/0.949 medida en Kaggle T4 ×2 (150 estudios train) y 4 variantes del forward (channels_last, sin TTA, TTA sagital, 47 ventanas) | worked | 0.9267→0.9282 (V2, pseudo 150, inflado) | 0.9229→0.9219 (V2, 58 gold, inflado) | — | usa las 2 T4; V0 1.20 s/estudio (≈ 26.8 min @1300), forward T4 5.3× PRO 4500; channels_last −3 % (H1 rechazada); sin TTA −50 % (13.7 min) con ΔAUC gold −0.001: V2 para eficiencia, V1 para AUC |
| exp027 | t4_rapido | rodrigo | exp026 | variantes rápidas de V2 en Kaggle T4 ×2 (150 estudios train): sin cudnn.benchmark, modelo 224 del autor, model.half(), torch.compile, combinaciones | didnt | 0.9282→0.9371 (B94/E1, pseudo 150, inflado) | 0.9219→0.9150 (B94, 58 gold, inflado) | — | R 12.96 min @1300; half −9 % sin cambiar predicciones (C 11.8 min); sin benchmark 0 %; compile −24 % steady pero 296 s de compilación (no paga a 1300); 224 −48 % (B94 6.8 min) con ΔAUC gold −0.007 [−0.019, +0.005] vs pseudo +0.009: veto gold, decide el LB; ya manda la prep en CPU (0.21 s/est.); sin TensorRT/ORT en la imagen |
