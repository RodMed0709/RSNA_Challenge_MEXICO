# Base de conocimiento — RSNA Knee 2026

Fecha: **2026-10-12**. Consolida todo lo aprendido ese día de ~115 papers (5 temas) y de 6 análisis cruzados
contra nuestro código y nuestros resultados, con **implementaciones concretas** para correr mejores experimentos.

## Cómo usarla

1. ¿Vas a abrir un experimento? Mira primero **`08_backlog_priorizado.md`** (tabla única, top-10 marcado con ★) y luego
   el archivo temático de la idea. Cada idea trae cita, costo GPU-h, riesgo y cómo plantearla como SDD de una variable.
2. ¿Augmentations? **`02_augmentations.md`** es el más detallado: dónde se inserta en la celda 8 de exp045, snippets PyTorch
   en GPU por estudio/slot, parámetros de los papers y criterio sugerido.
3. Antes de diseñar, lee **`07_errores_y_lecciones.md`** (SWA sin BN, exp052 nulo por construcción, mixup entre planos,
   afín p=1, padre s42 bajo, registro roto, interno vs LB).
4. Citas: claves de **`referencias.bib`** (59 entradas de A1 + 20 nuevas = 79; +19 de `09_preprocesado.md` = 98). La sección 1 de cada SPEC nueva debería citar una clave o decir "inferencia".
5. Las reglas del repo siguen mandando (`CLAUDE.md`, `docs/06_SDD.md`): esta carpeta propone, no decide. Los números de
   experimentos se copian de `metrics.json`/verdicts, no de aquí si difieren.

## Archivos

| Archivo | Contenido |
|---|---|
| `01_etiquetas_llm.md` | Consenso multi-LLM actual, lo medido (exp031/052/054/sub056_2), por qué exp052 era nulo, ideas LAB-1..8 (DS, dedup L2/L4, cabezas por fuente, CROWDLAB) |
| `02_augmentations.md` | Aug actual (exp045) y su código, diagnóstico de lo fallido, **fichas implementables de augs de RM** (anisotropía, motion, ghosting, spike, bias por slot, BigAug, GIN, gamma por slot), arreglos del afín, plan |
| `03_arquitectura_mil.md` | Modelo actual, literatura y ganadores Kaggle, ideas ARC-1..13 (embeddings congelados, slot embedding, gated, jerárquico, CoPAS, HeMIS, 3D a cero, ROI) |
| `04_robustez_ensambles_lb.md` | Ruido del LB, pesos de ensamble, estrategia de envíos/finales, ideas ENS-1..12 (fold soup, update_bn, SWAD, WiSE-FT, TTA, TENT) |
| `05_eficiencia.md` | Datos de runtime, ideas EFF-1..9 (destilación, presupuesto de cortes, I/O, INT8) |
| `06_paper_y_novedad.md` | Venue (JIIM), título, C1–C3, huecos, R1–R10, figuras |
| `07_errores_y_lecciones.md` | 14 errores con su regla |
| `08_backlog_priorizado.md` | **Tabla única de 60 ideas** con id, cita, tema, ganancia razonada, GPU-h, riesgo, a qué sirve, prioridad, estado |
| `09_preprocesado.md` | **Preprocesado de RM más allá de CLAHE**: pipeline actual verificado, 10 candidatos rankeados (recorte centrado en rodilla, escala física sin ampliar, Nyúl por slot, z-score foreground, muestreo en mm, orientación), veredicto CLAHE/Nyúl/N4 con tiempos de CPU medidos, diagnósticos D-1..D-5 |
| `referencias.bib` | BibTeX |

## De dónde sale cada cosa

Todo vive en `local/RSNA_papers/`, dentro del repo pero ignorado por git (los PDFs **no** se suben):

| Fuente | Qué es |
|---|---|
| `01_knee_mri_sota/RESUMEN.md` | 17 PDFs: MRNet, ELNet, MRPyrNet, CoPAS, MST, Vuskov, FMs de rodilla |
| `02_llm_report_labels/RESUMEN.md` | 27 PDFs: CheXpert/CheXbert, DS/Raykar/MACE/Snorkel, CROWDLAB, LLM labelers |
| `03_mil_multiseq_arch/RESUMEN.md` + `kaggle_*.md` | 24 PDFs + 8 writeups ganadores Kaggle (C-spine 2022, Abdominal 2023, Lumbar 2024) |
| `04_robustness_aug_ensemble/RESUMEN.md` | 24 PDFs: TorchIO, BigAug, Shaw, AdvBias, GIN, SynthSeg, SWA/SWAD/soups, TTA, calibración, Roelofs |
| `05_efficiency_isbi_novelty/RESUMEN.md` | 23 PDFs: KD, FastViT/RepViT, FLARE, MedPTQ, framing ISBI |
| `analisis/A1_trazabilidad.md`, `A1_referencias.bib` | Trazabilidad técnica → cita; estado del registro |
| `analisis/A2_implementacion_vs_papers.md` | Nuestro código (celdas de cada rama) vs literatura, técnica por técnica |
| `analisis/A3_etiquetas_llm.md` | Consenso exacto, análisis CPU de fuentes, DS/MACE, SPEC consenso_dedup |
| `analisis/A4_interno_vs_LB.md` | Ruido del LB, correlaciones, pesos a priori, estrategia de finales |
| `analisis/A5_paper_y_backlog.md` | Framing JIIM y backlog hasta el 22-oct |
| `analisis/A6_revision_colas.md` | Revisión de las colas de envío del 13 y 14-oct |
| Repo | celda 8 de `exp045_raptor_aug_espacial` (rama), `docs/pasos/paso7_plan_aug_ensamble.md`, `tablero_paso7.md` |

Convenciones de evidencia: **[paper]** sale del PDF; **[Kaggle]** de un writeup; **[inferencia]** razonamiento nuestro;
**[no en corpus]** conocimiento general sin PDF descargado (verificar antes de citar). Ganancias esperadas = hipótesis.

Copia en el volumen: `/workspace/users/rodrigo/orq/conocimiento/`.
