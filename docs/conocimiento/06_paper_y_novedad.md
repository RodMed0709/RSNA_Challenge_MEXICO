# 06 — Paper: framing, novedad y experimentos que faltan

Fecha: 2026-10-12. Fuentes: A5 Parte A, `05_efficiency_isbi_novelty/RESUMEN.md` (framing ISBI), huecos de los RESUMEN 01–04, A1 hallazgos.

## 1. Venue

- **ISBI main track: no alcanza con confianza** tal como están las contribuciones (RESUMEN 05). El challenge track solo acepta challenges alojados en ISBI; RSNA Knee no lo es. Deadline ISBI 26-oct, 4 días tras el cierre.
- **Recomendación: JIIM.** Plan B: one-page abstract ISBI 2027 (1-feb-2027, no archivable). Otros: CMIG (`docs/paper/PLAN_CMIG.md`).
- No publicar código ni preprint antes del 22-oct (`10_PLAN_PAPER.md`). Licencia MIRA: no mostrar imágenes RSNA sin confirmación (usar MRNet/fastMRI en figuras).

## 2. Título y contribuciones (A5 A.1–A.2)

*Learning Twelve Knee MRI Findings from Multilingual Radiology Reports: Multi-LLM Consensus Labels, Pre-Registered Ablations and Budget-Aware Distillation on a Multicenter Challenge Dataset*

- **C1 — Consenso multi-LLM** con efecto medido en AUC de imagen: exp031 +0.0155 pseudo (LB 0.930→0.939); L5 decorrelacionada +0.0060; L5v2 correlacionada +0.0017. Novedad: ningún trabajo de RM de rodilla/MSK entrena imagen con consenso multi-LLM ni separa "otra familia" de "otra lectura" (RESUMEN 01, 02). Precedentes más cercanos: Park 2024 (columna, 1 LLM, arXiv 2410.17235), Vuskov (extracción manual), Mukherjee 2023 (Vicuna, solo texto; **verificar cifras, PDF no bajado**).
- **C2 — Frontera precisión–runtime** en T4 con destilación a student 224/3 slots. Novedad: ningún trabajo multi-hallazgo de rodilla reporta coste.
- **C3 — Ablaciones pre-registradas (SDD)** con suelo de ruido medido (semilla 0.0013): ~25 ablaciones; funcionó aug espacial (+0.0063, OOF 0.8844); **no aditividad** aug × L5. Eisenmann 2023 (CVPR): los ganadores se apoyan en aug y preprocesado, pero casi nadie publica negativos.

## 3. Huecos adicionales detectados hoy

- **Artefactos de adquisición vs intensidad** como ablación en rodilla multi-centro (RESUMEN 04 Huecos #3) → ver `02_augmentations.md`.
- **Fold soup en MIL multi-secuencia** (RESUMEN 04 Huecos #1): "1 modelo con AUC de ensamble a 1/5 de coste".
- **Mini-Roelofs**: correlación de rangos interno ↔ LB sobre nuestro propio registro (RESUMEN 04 Huecos #2; A4).
- **Benchmark de agregadores** con backbone congelado (RESUMEN 03 §4.1).
- **Qué secuencias necesita cada hallazgo** (presupuesto de secuencias vs AUC por hallazgo; exp043: derrame y Baker dependen del axial).
- **Pesos de blend robustos vs optimizados en OOF** bajo shift (evidencia empírica de Müller/Varoquaux/Roelofs).

## 4. Experimentos que faltan para revisión (A5 A.4)

| # | Qué | Coste | Cuándo |
|---|---|---|---|
| R1 | IC bootstrap pareado por estudio (2000 réplicas) de macro-AUC y cada Δ (OOF 5f n≈4407, gold 58) | CPU | ya |
| R2 | DeLong por hallazgo + Holm/BH; no inferioridad student vs ensamble | CPU | ya |
| R3 | ≥3 semillas de exp031, exp045, exp054 (fold 0) | ~10 GPU-h | tras 22-oct |
| R4 | Calidad de etiqueta vs humanos (dev-200, exp010); κ por hallazgo | horas humanas | en curso |
| R5 | Validación externa: **MRNet** (Stanford AIMI RUA, Redivis doi:10.71718/rcbp-8c35; zero-shot sobre train+valid = 1250; 3 etiquetas), **KneeMRI** (Zenodo 14789903, CC BY-NC-ND; Rijeka → descartar solapamiento con sitio croata), **fastMRI+** (~7 de 12, coronal, cajas) | ≤1 GPU-h | nov; **pedir acceso a MRNet ya** |
| R6 | LB privado + runtime oficial de cada punto de la frontera | envíos | **antes del 22-oct** |
| R7 | Calibración (ECE 0.20 en exp040; Brier) | CPU | nov |
| R8 | Sensibilidad por sitio/idioma (croata 0.817 vs inglés 0.880 en exp040) | CPU | nov |
| R9 | Baselines con mismo split: MRNet-style max-pool, ABMIL, CoPAS-style, FM congelado | horas | nov |
| R10 | Checklists CLAIM 2024 y TRIPOD+AI; declarar uso de LLM | — | redacción |

## 5. Figuras (A5 A.5)

Pipeline con coste por etapa · κ entre fuentes + AUC vs nº de fuentes · forest plot de ablaciones con banda de ruido · frontera Pareto con iso-score · heatmap AUC por hallazgo × configuración · atención por secuencia/corte (sobre MRNet/fastMRI+) · suplementaria interno ↔ LB.
