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
| exp066 | compact_f0 | rodrigo | — (ref. exp045) | heliosli compact ConvNeXt-S (E_lrprior) entrenada limpia en split_v1 fold 0; blend con exp045 f0 | worked | 0.8755 (blend 0.8955) | 0.8972 (blend 0.9256) | — | compact sola < exp045 (−0.014) pero el blend rank 0.5/0.5 con exp045 sube +0.0060 pseudo / +0.0055 gold; 14 GPU-h por fold |
| exp068 | compact_r7_receta | rodrigo | exp066 | trainer público R7 v9.8 + flags reproducibles (EMA, jitter, mirror, aug v2, cover, syn-decomp), fold 0 | running | — | | | worked si pseudo ≥ 0.8805 y gold ≥ 0.8772 |
| exp069 | compact_r7_ft | rodrigo | exp068 | fine-tune de los pesos públicos R7 f0 con consenso exp031, 5 ép LR x0.3, all-data | didnt | — | 0.9271 (init R7 0.9331) | — | LB r7-raptor-exp069 0.957 < 0.958 (ref 0.956); veto pasa (Δ gold −0.0060 vs init); 4345 estudios (3 errores de caché); init no estricto |
