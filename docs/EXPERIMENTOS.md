# Registro de experimentos

**Referencia actual: exp002_dinov2s_ref** (5 folds: OOF pseudo 0.802, gold 0.788; fold 0: pseudo 0.827, gold 0.777).

Una fila por experimento. `nuevo_experimento()` agrega la fila con estado `spec`; su autor la completa en el verdict.
Estados: `spec`, `running`, `worked`, `didnt`, `killed`. Los números salen de `results/expNNN_tag/metrics.json`.

| exp | tag | autor | padre | variable cambiada | estado | pseudo f0 | gold | OOF 5f | aprendizaje |
|---|---|---|---|---|---|---|---|---|---|
| exp001 | preparacion | rodrigo | — | exploración, cache v1_256, etiquetas Qwen v1, split v1 | worked | — | — | — | Sin IDs de paciente; el idioma del informe aproxima el hospital; "no mencionado" quedó como 0.5 en etiquetas v1 |
| exp002 | dinov2s_ref | rodrigo | — | referencia DINOv2-small sobre split_v1 | worked | 0.827 | 0.788 | 0.802 | fold 1 inestable (0.738); el mejor checkpoint es la última época en 4/5 folds: le faltan épocas |
| exp003 | mnv4m | rodrigo | exp002 | paso 1: backbone mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k | spec | | | | |
| exp004 | effv2b3 | rodrigo | exp002 | paso 1: backbone tf_efficientnetv2_b3.in21k_ft_in1k | spec | | | | |
| exp005 | cnxv2n | rodrigo | exp002 | paso 1: backbone convnextv2_nano.fcmae_ft_in22k_in1k | spec | | | | |
| exp006 | dinov3s | rodrigo | exp002 | paso 1: backbone vit_small_patch16_dinov3.lvd1689m | spec | | | | |
| exp007 | cnxt_dinov3 | rodrigo | exp002 | paso 1: backbone convnext_tiny.dinov3_lvd1689m | spec | | | | |
| exp008 | coatnet2 | rodrigo | exp002 | paso 1: backbone coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | spec | | | | |
| exp009 | preparacion_v2 | rodrigo | exp001 | paso 3: metadatos DICOM v2 (plano, secuencia, lateralidad, slots) y cache v2 | worked | — | — | — | meta_v2: plano IOP 99.3 %, lado 98.6 % de estudios con 97.0 % vs informe (tag 96.6, texto 98.6, geometría 97.2); slot 1 SAG líquido FS 94.3 % (<95 %: el hospital alemán no hace SAG FS) |
