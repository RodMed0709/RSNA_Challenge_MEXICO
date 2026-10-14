# exp072_raptor_fm_aux

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial (fold 0: pseudo L2 0.8899 s42 / media s42+s7 0.8880; gold 0.9202; 2.51 GPU-h) |
| Fecha | 2026-10-14 |
| Rama | exp072_raptor_fm_aux |
| Estado | spec |

Idea: V2 del plan de preentreno externo (`docs/pasos/plan_preentreno_externo.md` §3.3): tarea auxiliar fastMRI+ estilo R7
(heliosli) sobre el encoder de la receta Raptor exp045. Es la vía "propia" (datos externos públicos con cajas, sin OAI).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si en cada paso de entrenamiento el mismo encoder CoAtNet-2 aprende además a localizar lesiones con cajas de
radiólogo (fastMRI+, 974 volúmenes coronales con 16k cajas; 11 clases que solapan ~8 de nuestros 12 hallazgos: menisco,
cartílago, edema óseo, fractura/contusión, derrame, quistes, LCA, colaterales), el encoder recibe supervisión espacial densa que
las etiquetas de informe (débiles, a nivel de estudio) no dan, y la pseudo L2 de fold 0 sube ≥ +0.005 sobre exp045.
R7 (ConvNeXt-S) es el modelo público que usa esto (+0.003 LB R6→R7 en blend, confundido con otros cambios).

**Única variable cambiada:** la pérdida de entrenamiento = L_main (exp045, sin cambios) + λ·L_aux, λ = 0.3.

| Parámetro o celda | Padre (exp045) | Este experimento |
|---|---|---|
| Tarea auxiliar fastMRI+ (`FM_AUX_WEIGHT`, celdas 0, 2, 3, 4, 6) | no | λ 0.3; por paso un lote auxiliar de 48 ventanas coronales de 3 cortes (4 volúmenes × 12; 75 % de volúmenes con cajas, 25 % revisados sin hallazgos; en volúmenes con cajas la mitad de ventanas centradas en cortes con caja) por el MISMO backbone (`forward_features`, mapa 10×10 a 320); cabeza aparte LayerNorm + lineal d→11 por posición (sesgo −4); L_aux = BCE densa (fracción cubierta de cada celda 32×32 por las cajas de la clase en los 3 cortes; pos_weight 8) + BCE de presencia (log-mean-exp sobre posiciones; clase presente si una caja conserva ≥ 25 % de su área); segundo backward al mismo .grad, un clip y un paso |

Clases (orden R7): meniscus_tear (Meniscus Tear + Displaced Meniscal Tissue), cart_partial, cart_full, bone_edema,
bone_fx_contusion, effusion, cyst (Periarticular cysts), acl_high, acl_low, collateral (MCL + LCL), other (resto; se descarta
`artifact`). Ventana auxiliar: recorte 320 del corte 384 con desplazamiento U(0..64) (como R7), espejo horizontal p 0.5,
ganancia U(0.9, 1.1) y normalización ImageNet (las del camino principal); sin afín. 5 % de volúmenes en holdout (diagnóstico).

Todo lo demás idéntico a exp045: etiquetas de consenso, corpus 96×384, recorte central 320, aug espacial, 16 épocas,
bs 8 × K 12 (K_EVAL 24), AdamW/OneCycle/LRs (la cabeza auxiliar entra con HEAD_LR), bf16, clip 3.0, semilla 42,
`split_v1_seed42` fold 0, selección por pseudo L2, SWA top-3. La cabeza auxiliar nunca entra en el SWA ni en los pesos de
inferencia (coste de inferencia en Kaggle = el del padre).

Diferencias declaradas frente a R7 (no son variables extra; R7 es otra arquitectura): 48 ventanas por paso en vez de 96 (coste;
el lote principal son 96 ventanas); solo la vista de campo completo (R7 usa además 2 vistas locales de 100 mm de su caché px v4,
que no existe en nuestro corpus); ganancia en vez de gamma/contraste. **BatchNorm:** CoAtNet-2 tiene BN (R7 rechaza backbones
con BN porque el lote fastMRI movería las estadísticas). Mitigación: durante el forward auxiliar `momentum = 0` en todas las BN
(usa estadísticas del lote, no toca las running stats que usa la inferencia).

**Datos (preparados el 14-oct, `/workspace/data/external/fastmri_plus/`):** 1171 de los 1172 volúmenes de
`knee_file_list.csv` (fastMRI single-coil train+val; `file1000248` no está en las copias de Kaggle), `reconstruction_rss`,
volteo arriba-abajo (convención de fastMRI+, `ExampleScripts/example.ipynb`), p2/p98 por volumen → uint8, 320 → 384 bilineal
(FOV fastMRI ≈ 140 mm, igual que nuestro recorte físico); cajas de `knee.csv` × 1.2. Se bajaron los h5 uno a uno y se borraron
(huella ≈ 6 GB). Script: `fm_build.py` en esta carpeta.
**Nota de ejecución (14-oct 15:45 UTC, antes de correr nada de exp072):** la API de Kaggle devolvió 429 (Too Many Requests)
a partir del volumen ~800; quedaron **799 volúmenes** (664 con cajas, 135 revisados sin hallazgos; faltan 321 de
`fastmri-train-knee-2` y 51 de `fastmri-knee-val`). Se corre con lo que haya a las 16:00 UTC (reintentos lentos hasta entonces);
los criterios no cambian. Si funciona, completar la tienda antes de 5 folds.
A las 16:00 UTC los reintentos dejaron **995 volúmenes** (train 784 con cajas + 162 negativos, holdout 49; 13,783 cajas): con eso corre exp072.
El primer smoke murió por OOM al compartir GPU con exp071 (no es error del código); se repite cuando exp071 termina.

**Efecto esperado:** Δ pseudo fold 0 +0.000 a +0.006 (prior baja-media: con 4.4k estudios la supervisión externa suele aportar
poco; R7 lo usa pero su ganancia aislada no está medida). Coste ≈ ×1.4–1.5 por época.

**Criterios (no cambian después de correr):** referencia exp045 fold 0 = media de semillas 42 y 7 = **0.8880** (pseudo L2,
mejor época); gold exp045 fold 0 = 0.9202.
- `worked` (promete → 5 folds / all-data): pseudo L2 fold 0 ≥ 0.8880 + 0.005 = **≥ 0.8930** **y** gold ≥ 0.9202 − 0.02 = **≥ 0.9002**.
- `killed`: pseudo L2 fold 0 ≤ 0.8880 − 0.005 = **≤ 0.8830**, o gold < 0.9002, o GPU-h > 1.5 × 3.75 (presupuesto de abajo).
- `didnt`: cualquier otro caso.
- Gold nunca elige época, λ ni nada (solo veto). La AUC de presencia en el holdout fastMRI+ es diagnóstico (¿aprendió la
  cabeza?), no decide.

**Presupuesto:** smoke ≈ 0.15 h + fold 0 ≈ 3.75 GPU-h equivalentes RTX PRO 4500 (≈ 2.5 h en RTX 5090, $1.19/h ≈ $3).

**Licencia / legal:** las imágenes vienen de copias de terceros en Kaggle (`arafatshovon/trained`,
`arafatshovon/fastmri-train-knee-2`, `arafatshovon/fastmri-knee-val`) del dataset fastMRI de NYU Langone (uso de investigación
bajo el Data Use Agreement de NYU, sin redistribución); las cajas son de microsoft/fastmri-plus (GitHub). Para el paper Rodrigo
firmará el DUA de fastMRI (fastmri.med.nyu.edu) y se citarán fastMRI y fastMRI+. Datos externos: vigilar reglas §6 de la
competición (gratis con registro = "reasonably accessible"). No se redistribuye nada derivado.

**Fuentes:**
- R7: kernel https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click y dataset
  `heliosli/rsna-knee-compact-r7cnxfsss` (trainer `compact_train.py` v9.8 `--fm-aux`: λ 0.3, 96 ventanas, 11 clases,
  pos_weight 8, presencia ≥ 25 %, sesgo −4, LayerNorm + lineal por posición).
- Zhao R, Yaman B, Zhang Y, et al. fastMRI+: Clinical pathology annotations for knee and brain fully sampled multi-coil MRI
  data. Sci Data 2022;9:152. arXiv:2109.03812. Anotaciones: https://github.com/microsoft/fastmri-plus (`Annotations/knee.csv`).
- Zbontar J, Knoll F, Sriram A, et al. fastMRI: An Open Dataset and Benchmarks for Accelerated MRI. arXiv:1811.08839 (2018);
  Knoll F et al. Radiol Artif Intell 2020;2(1):e190007.

**Riesgos:** BN (mitigado con momentum 0; puede quedar un desajuste por las estadísticas de lote); dominio (fastMRI = un sitio
NYU, 2 secuencias coronales, solo slot coronal): la tarea puede tirar del encoder hacia coronal y bajar sagital/axial; coste
×1.5; posible fuga si NYU fuera sitio del RSNA (sin etiquetas del concurso, riesgo bajo).

## 2. PLAN

- Celda 0 (parámetros, celda 2 del .ipynb): `FM_DIR`, `FM_LOCAL`, `FM_AUX_WEIGHT = 0.3`, `FM_AUX_WINDOWS = 48`,
  `FM_VOLS_PER_STEP = 4`, `FM_BOXED_FRAC = 0.75`, `FM_POS_WEIGHT = 8.0`, `FM_PRESENCE_MIN = 0.25`, `FM_HOLDOUT_FRAC = 0.05`,
  `FM_WORKERS = 2`, `NOTES`.
- Celda 2 (datos, celda 6): carga de la tienda fastMRI+ (copia a `/root/fm384`, mapa de clases, cajas × 1.2, holdout 5 %).
- Celda 3 (modelo, celda 8): `fm_window_targets` (port de R7, sin traslación), muestreador auxiliar (IterableDataset, 2
  workers, generadores propios: no consume el RNG global), `FmAuxHead`, pérdida auxiliar, BN momentum 0 en el forward auxiliar.
- Celda 4 (entrenamiento, celda 10): segundo backward λ·L_aux por paso, clip sobre modelo + cabeza, cabeza en el optimizador
  (con HEAD_LR); AUC de presencia en holdout al final del fold; cabeza guardada aparte (`fm_aux_head_f0.pt`).
- Celda 6 (metrics, celda 14): bloque `fm_aux`.
- Ninguna otra. Con `FM_AUX_WEIGHT = 0` el notebook equivale a exp045.
- Smoke `SMOKE = True` → fold 0 (`FOLDS = [0]`, `SMOKE = False`) con `run_notebook.sh` en `rsna-ext1`, auto-stop.

## 3. VERDICT (números de `results/exp072_raptor_fm_aux/metrics.json`)

**Resultado:** worked / didnt / killed · **Fase alcanzada:** screen / 5 folds

| Métrica | Padre | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | | | |
| gold AUC (screen: fold 0 contra fold 0 del padre) | | | |
| OOF pseudo 5 folds | | | |
| GPU-h | | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
