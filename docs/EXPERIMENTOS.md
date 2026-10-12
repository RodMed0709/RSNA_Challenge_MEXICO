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
| exp029 | raptor_entreno | rodrigo | — | reproducción receta Raptor/nartaa d96/384 (CoAtNet-2, MIL por hallazgo, etiquetas blandas Raptor) en fold 0, selección pseudo L2 | run | | | | |
| exp031 | raptor_etiquetas_consenso | rodrigo | exp029 | etiquetas de entrenamiento = promedio por celda de Raptor, L2, L3 y L4 (en vez de solo Raptor `labels_llm_soft`) | worked | 0.884 (pseudo L2) | 0.914 (f0) | — | Δ pseudo L2 +0.0155, gold +0.003, v1 +0.004, 2.49 GPU-h; sube Synovitis/MCL/Effusion/ACL; vara L2 sesgada a favor (consenso contiene L2) |
| exp045 | raptor_aug_espacial | rodrigo | exp031 | aug espacial de train: espejo L/R anatómico p 0.5 + afín por estudio (rot ±10°, escala 0.9–1.1, tras ±6 %) | worked | 0.890 (pseudo L2) | 0.920 (f0) | — | Δ pseudo L2 +0.0063 (SWA +0.0060), v1 +0.008, gold +0.006, 2.51 GPU-h; sube ACL/MCL/meniscos; all-data hecho (SWA 13–15, gold 0.919) |
| exp057 | raptor_eficiente | rodrigo | exp045 | producto eficiencia: exp045 a 224 px, 3 slots (Sag fluid, Sag, Cor fluid), sin TTA; trade-off AUC vs runtime | spec | | | | |
