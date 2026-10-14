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
| exp012 | entrada_v2 | rodrigo | exp002 | Receta exp002 con cache v2 (recorte físico, 336 px, alineado, 6 slots) | spec | | | | |
| exp013 | cabeza_hallazgo | rodrigo | exp002 | Cabeza con una query por hallazgo (12 queries) sobre las series en lugar de pooling único + MLP | worked | 0.838 | 0.789 (f0) | 0.813 | +0.011 OOF confirmado; atención casi uniforme (la ganancia es la cabeza separada, no la selección); en v2 no suma (exp018) |
| exp014 | mnv4m_lr | rodrigo | exp003 | `BACKBONE_LR` 1e-5 → 1e-4 en MobileNetV4-M | killed | 0.737 | 0.633 (f0) | — | el lr no era el problema: gold −0.068 vs exp003; sospechas: pesos r384 a 224 px y BatchNorm con batch 2 |
| exp016 | entrada_25d | rodrigo | exp002 | Entrada 2.5D: cada corte con sus 2 vecinos como canales RGB en lugar de repetir el corte 3 veces | didnt | 0.824 | 0.778 (f0) | — | neutral (Δ pseudo −0.003, gold +0.001); con v1_256 los vecinos no aportan; costo igual |
| exp017 | atencion_dispersa | rodrigo | exp013 | Penalización de entropía sobre la atención por hallazgo para que cada hallazgo elija pocas series | killed | 0.793 | 0.732 (f0) | — | colapso: los 12 hallazgos atienden la misma serie coronal (entropía 0.01, coseno 1.00); λ 0.05 demasiado fuerte |
| exp018 | cabeza_hallazgo_v2 | rodrigo | exp012 | Cabeza por hallazgo de exp013 (12 queries) sobre la entrada v2 de exp012 | didnt | 0.846 | 0.777 (f0) | — | en v2 la cabeza no suma (Δ −0.004); atención casi uniforme pero sesgo coherente a slots FS |
| exp019 | poda_slots | rodrigo | exp012 | Solo inferencia: qué slots se usan (resto enmascarado) con el checkpoint de exp012 f0 | didnt | 0.850 (6 slots) | 0.784 | — | sin reentrenar, quitar slots cuesta ≥0.009 pseudo; 3 slots (61 % costo) −0.026; ~280 s de prepro valen solo ~0.004 AUC en la métrica |
