# Plan del paper

Fecha: 2026-10-06. Objetivo: track de eficiencia de Kaggle + paper. Dos entregas: ISBI 2027 (corto) y CMIG (largo).

## Reglas que afectan
- Licencia MIRA: paper académico permitido, sin embargo. No se redistribuyen datos. Imágenes en figuras y liberar
  pseudo-etiquetas: preguntar a informatics@rsna.org antes.
- Hasta el 22-oct no se comparte código fuera de Kaggle (nada de arXiv ni repo público antes).
- Los papers oficiales de RSNA los firman solo los organizadores. Nuestro paper es independiente del puesto.
- Premio de eficiencia: Efficiency = AUC/(Benchmark − max AUC) + RuntimeSeconds/32400, gana el menor, Benchmark = 0.5.
  0.01 de AUC ≈ 697 s de runtime. El tiempo pesa mucho: un modelo rápido de ~0.90 compite. Hay que superar a
  `sample_submission` en el privado. Notebook ≤ 9 h, CPU o GPU, sin internet.

## Tesis

Atención con una query por hallazgo sobre series y cortes, que permite podar series en inferencia, con curva
AUC vs segundos por estudio medida en hardware real, entrenada con etiquetas de informes extraídas con LLM.

| Ángulo | Novedad | Dónde va |
|---|---|---|
| Poda de series guiada por atención por hallazgo | media-alta | ISBI y CMIG (método) |
| Curva AUC–cómputo en rodilla multi-hallazgo | media-alta | ISBI y CMIG (marco) |
| Etiquetas LLM de informes, validadas con dev humano y gold | baja-media | CMIG (datos) |
| Validación externa (MRNet, KneeMRI) | baja | CMIG |

## ISBI 2027
- Envío 26-oct-2026, 23:59 EDT, por EDAS. 4 páginas + 1 de referencias (US$200). Single-blind, arXiv permitido.
- Presencial en Lausana, sin opción remota. Notificación 12-ene-2027, camera-ready 26-ene.
- Secciones obligatorias: ética, conflicto de interés, uso de IA.
- Mínimo: ablación media vs query por hallazgo; curva de poda k = 1..6 vs poda aleatoria; tiempo en T4;
  LB privado completo y podado; gold con IC bootstrap.
- Go/no-go 14-oct. Números congelados 20-oct. Plan B: directo a CMIG o MIDL 2027.

## CMIG
- IF 5.5. Sin límite de palabras. Abstract ≤ 250. Gratis por suscripción (OA US$2,920). ~70 días a primera decisión.
- No enviar antes del 12-ene-2027 (ISBI). Extensión con ≥ 30 % de contenido nuevo y citando ISBI.
- Lo que añade sobre ISBI: destilación a estudiante pequeño, etiquetas LLM vs dev humano vs gold, validación externa,
  ablaciones por componente, CPU vs GPU, energía, robustez, Grad-CAM/atención, calibración.
- Debilidad mayor: gold de 58 estudios. Buscar radiólogo para ampliar a ≥ 200.
- Datos externos: MRNet y KneeMRI (pedir acceso ya), fastMRI+ (~7 de 12 etiquetas). OAI/MOAKS para revisión.
- Fechas: experimentos congelados 28-feb-2027, envío 14-may-2027.
- Cascada: CMIG → Comput Biol Med → JBHI / JIIM → MELBA → Sci Rep / J Imaging.

## Evaluación
- Tres niveles: OOF 5 folds con bootstrap por estudio; gold 58 (IC ~±0.15 por etiqueta); LB privado de Kaggle con
  runtime oficial (medirlo antes del 22-oct, después no se puede).
- No inferioridad con bootstrap pareado en macro AUC; DeLong por etiqueta; Holm o BH para las 12.
- Eficiencia: s/estudio en T4 y CPU incluyendo decodificación DICOM, desglose por fase; GMACs (fvcore/thop), params, memoria pico.
- Calibración: ECE y Brier. Checklists CLAIM 2024 y TRIPOD+AI.
- Sensibilidad: dejar fuera un idioma/fabricante (el split no está agrupado por sitio).

## Calendario

| Fecha | Hito |
|---|---|
| 7–12 oct | Paso 4: exp010–exp014 (etiquetas y entrada) |
| 12–20 oct | Cabeza por hallazgo, poda, destilación, Pareto |
| 14 oct | Go/no-go ISBI |
| 20 oct | Números congelados |
| 22 oct | Cierre Kaggle |
| 26 oct | Envío ISBI |
| 12 ene 2027 | Notificación ISBI |
| 28 feb | Experimentos CMIG congelados |
| 14 may | Envío CMIG |
