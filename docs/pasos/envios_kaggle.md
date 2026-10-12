# Envíos a Kaggle

Registro de cada envío: qué modelo, LB público, puesto y runtime. Los kernels y logs viven fuera del repo (scratchpad de la
sesión); aquí solo los números. Kernels privados, sin internet, GPU T4.

| Fecha | ID envío | Kernel | Modelo | LB público | Puesto aprox. | Runtime notebook (test visible, 3 estudios) |
|---|---|---|---|---|---|---|
| 2026-10-08 | 56963705 | rsna-knee-exp002-inference v2 | exp002, ensamble 5 folds (8 épocas) | 0.812 | ~4368 / 5530 | 47.2 s |
| 2026-10-08 | 56969845 | rsna-knee-exp015-inference v1 | exp015, solo fold 0 (16 épocas) | 0.810 | ~4374 / 5530 | 45.0 s |
| 2026-10-09 | 56981839 | rsna-knee-exp021-inference v1 | exp021, solo fold 0 (8 épocas, etiquetas L2) | **0.836** | ~4266 / 5543 | 62.7 s |
| 2026-10-11 | 57081163 | mirror-exp023 v1 | exp023: copia tal cual de `nartaa/rsna-knee-0949-anatomical-mirror` (CoAtNet Raptor SWA + espejo L/R) | 0.949 | ~1175–1305 / 5760 (empate) | 66.6 s (score en ~25–29 min) |
| 2026-10-11 | 57081259 | velci-exp023 v1 | exp023: copia tal cual de `bliverrigpdoge/velci-kcd25-20`; `rsna-knee-kcd25leg` privado → queda la base 0.954 (Raptor 0.7 + compact S6 0.3) | **0.954** | ~135–535 / 5760 (empate) | 194.1 s (score en ~3 h 25 min) |
| 2026-10-11 | 57082625 | blendmine-exp023 v2 | exp023: copia tal cual de `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters` (Raptor 0.5 + compact S6 0.5); `rsna-knee-compact-r6cnxfsss` v2 re-alojado privado | **0.954** | ~135–535 / 5760 (empate) | 184.3 s (score en ~4 h 10 min) |
| 2026-10-11 | 57084774 | exp029-raptor-f0 v1 | exp029, solo fold 0 (receta Raptor/nartaa CoAtNet-2 MIL, etiquetas blandas Raptor, best_f0 ép. 15) + TTA espejo, inferencia nartaa 0949 | **0.933** | — | 63 s (script 25 s) |
| 2026-10-12 | 57088360 | exp030-raptor-alldata v1 | exp030 all-data (receta exp029 con etiquetas L2, 4349 no-gold, SWA épocas 12/14/15) + TTA espejo, inferencia nartaa 0949 | 0.930 | — | 73.6 s (script 27 s; score en ~27 min) |
| 2026-10-12 | 57089745 | exp041-sub-e1-94 v1 | exp027/041 Ch: pesos nartaa 224 (`0945-efficient-224crop`), `model.half()`, 94 ventanas en un lote, sin TTA (eficiencia) | **0.945** | — | 54.1 s (script 12 s; score en ~8 min) |

## Lectura

- **exp023 (reproducción de públicos):** la base pública se reproduce en nuestra cuenta: velci y blend-mine dan 0.954,
  el CoAtNet solo de nartaa 0.949. El "runtime" de la columna es el del run sobre el test visible (3 estudios); el tiempo
  hasta tener score (incluye cola de Kaggle y test oculto) es la cota que tenemos del runtime real: ~0.5 h el CoAtNet solo,
  ~3.5–4 h los blends con el compact S6. Para el premio de eficiencia, el compact cuesta ~3 h por +0.005 AUC.

- **exp021 confirma la hipótesis de las etiquetas:** misma arquitectura y receta que exp002, solo etiquetas L2 → LB +0.024.
  Es el primer movimiento real del LB. La pseudo contra v1 bajó (−0.038) mientras el LB subió: v1 ya no sirve de vara interna.

- 1 fold de 16 épocas empata con el ensamble de 5 folds de 8 épocas, con 1/5 del cómputo del modelo (forward 0.52 s/estudio
  contra 0.94 s). Para el track de eficiencia, el ensamble no paga.
- La brecha con el LB es grande: mediana del LB 0.936, nosotros 0.81, mientras la pseudo AUC interna sube a 0.85. Lo que
  más mueve la validación interna (entrada, épocas) apenas toca el LB. Hipótesis principal: las etiquetas de Kaggle siguen
  definiciones estrictas (ACL y MCL solo alto grado o completa; derrame y Baker solo moderado o grande; OA con pérdida de
  cartílago >50 %) y nuestras etiquetas v1 no. La validación interna mide contra v1, no contra la definición del test.
  Esto sube la prioridad de exp011 (etiquetas v2 por LLM con las definiciones oficiales) por encima de más arquitectura.
- En el notebook, leer DICOM domina (22 s de workers en 3 estudios); el forward es menor. Coincide con
  `costo_inferencia.md`: el preprocesado manda el runtime.
