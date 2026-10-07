# Plan de envío — ISBI 2027 (deadline 2026-10-26)

Fecha del plan: 2026-10-06. Base: `BRIEF.md`. Todo lo marcado [VERIFICADO] sale de las URLs citadas;
lo marcado [INFERIDO] es razonamiento propio o dato de una edición anterior.

---

## 0. Hechos verificados del venue

| Tema | Dato | Fuente |
|---|---|---|
| Sede / fechas | Lausana (EPFL), 25–28 mayo 2027 | https://biomedicalimaging.org/2027/ |
| Envío 4 páginas | **lunes 26-oct-2026, 23:59 EDT** (= 21:59 hora CDMX del 26-oct) | https://biomedicalimaging.org/2027/ |
| Notificación | martes 12-ene-2027 | ídem |
| Camera-ready ("Final Submission") | martes 26-ene-2027 | ídem |
| Sistema de envío | EDAS: https://edas.info/N35634 | ídem |
| Longitud | 4 páginas de contenido técnico; **5ª página opcional por 200 USD**, solo para referencias, declaración ética y agradecimientos/COI | https://biomedicalimaging.org/2027/papers/ |
| Plantilla | Plantilla ISBI propia (ISBI_template-master.zip en la página de autores); históricamente es IEEE a dos columnas [INFERIDO: la página no detalla columnas/tamaño, verificar al descargar] | https://biomedicalimaging.org/2027/papers/ |
| Revisión | **Single-blind** (no hay que anonimizar) | ídem |
| arXiv | Permitido publicar preprint (arXiv/medRxiv/bioRxiv) antes del envío; dan aviso de copyright IEEE para el preprint | ídem |
| Dual submission | No publicado ni aceptado "in a substantially similar form in any peer-reviewed venue **including journal**, conference, or workshop"; nada sustancialmente similar enviado a otra conferencia/workshop durante la revisión | ídem |
| Obligatorio en el PDF | Sección "Compliance with Ethical Standards" (datos open-access sin revisión ética se declara así) + agradecimientos/COI + **declaración de uso de IA** (salvo corrección gramatical) | ídem |
| Presencia | "**Fully in-person** conference. Virtual presentations will not be permitted." Paper no presentado en persona se **retira de proceedings** | ídem |
| Registro 2027 | Aún "Coming Soon" | https://biomedicalimaging.org/2027/registration/ |
| Registro 2026 (referencia) | Early: 1,050 USD miembro IEEE / 1,250 no miembro / 625 estudiante IEEE / 725 estudiante no miembro (sin IVA). Standard: 1,250 / 1,350 / 725 / 825 | https://biomedicalimaging.org/2026/registration/ |
| One-page abstracts | Envío 1-feb-2027, notificación 15-feb; **solo póster, no aparece en proceedings** | https://biomedicalimaging.org/2027/ y /papers/ |
| Challenges track | Propuestas de challenge cerraron 21-sep-2026; papers del track son para challenges organizados en ISBI (envío 26-feb-2027). **No aplica** a un paper sobre un challenge de Kaggle → vamos por track regular de 4 páginas [INFERIDO] | https://biomedicalimaging.org/2027/ |

### Extensión posterior a journal
- **IEEE:** exige material "sustancialmente" nuevo a discreción del editor; regla práctica ~30% nuevo (algunas sociedades piden más); CrossCheck marca solapes >30%; citar el paper de conferencia y declararlo al editor.
  Fuentes: https://ieee-edusociety.org/publication/tlt/tlt-policy-reuse , https://comsoc.org/publications/journals/ieee-tcom/information-authors/conference-vs-journal-papers
- **Elsevier (CMIG):** el preprint no cuenta como publicación previa; la declaración de envío dice que el trabajo no se ha publicado antes "except in the form of an abstract or as part of a published lecture or academic thesis". Un paper de proceedings IEEE **sí** es publicación previa → la versión journal debe ser una extensión sustancial, citar el ISBI y declararlo en la cover letter. La guía de CMIG (https://www.sciencedirect.com/journal/computerized-medical-imaging-and-graphics/publish/guide-for-authors) devolvió 403; **verificar a mano** si CMIG fija un % (no encontré cifra).
  Fuentes: https://www.elsevier.com/connect/clarification-of-our-policy-on-prior-publication , https://www.elsevier.com/en-xs/connect/ive-seen-this-somewhere-before-what-counts-as-prior-publication
- **Implicación práctica:** no enviar la versión journal antes del 12-ene-2027 (notificación ISBI) para no chocar con la cláusula "including journal". Objetivo journal: ≥40–50% contenido nuevo (experimentos nuevos, no solo prosa), copyright de figuras rehecho.

### Otras restricciones (del BRIEF)
- Kaggle: no compartir código fuera de Kaggle antes del 22-oct → el envío del 26-oct y un arXiv posterior quedan fuera de esa ventana. OK.
- MIRA: paper académico OK; **imágenes en figuras y liberar pseudo-etiquetas: dudoso** → la figura de arquitectura debe funcionar sin imágenes reales.
- Etiquetas de test nunca se publican → el único número de test reportable es el score del LB (público/privado) de nuestras submissions.

---

## 1. Tesis y título

**Tesis (1 frase):** Una cabeza de agregación por hallazgo (una *query* aprendida por etiqueta que atiende sobre series y cortes) permite que un solo modelo pequeño decida qué serie de RM mirar para cada hallazgo, de modo que en inferencia se pueden podar series y trazar una curva AUC–cómputo medida en hardware real, reteniendo casi todo el AUC con una fracción del tiempo.

Lo que NO es la tesis del ISBI (va al journal): el pipeline LLM de etiquetas, la destilación del ensemble, MRNet, la comparación exhaustiva de backbones.

**Títulos tentativos:**
1. *Look Where It Matters: Per-Finding Series Pruning for Compute-Efficient Multi-Label Knee MRI Classification*
2. *Label-Query Attention Enables Budgeted Inference in Multi-Series Knee MRI*
3. *Accuracy–Compute Trade-offs in Multi-Series Knee MRI: Finding-Specific Aggregation and Series Pruning on 12 Abnormalities*

Recomendado: 1 (claim claro y verificable). El 3 es el respaldo si la poda no da un resultado limpio (el paper queda como estudio de Pareto).

---

## 2. Outline página por página (IEEE 2 col, ~2,700 palabras de texto + 2 figuras + 1 tabla)

| Pág. | Contenido | Palabras |
|---|---|---|
| 1 | Título, autores, abstract (150), index terms. **1. Introduction** (~450): RM de rodilla multi-serie y multi-hallazgo; coste de stacks de 6–8 h vs tiempo clínico; gap: la mayoría agrega series de forma fija (mean/max/concat) y no expone qué serie importa por hallazgo ni reporta coste; contribuciones (3 bullets: label-query aggregation; poda de series por presupuesto; curva Pareto AUC vs s/estudio en 12 hallazgos, hardware real). | ~600 |
| 1–2 | **2. Related work** (~250, dentro de la intro o sección corta): MRNet y sucesores (agregación por plano), multi-instance/attention pooling, query-based heads (ML-Decoder/Query2Label), eficiencia en imagen médica (early-exit, selección de cortes). | ~250 |
| 2 | **3. Method** (~650). 3.1 Datos y preprocesado (recorte físico, 336 px, 6 slots por tipo de secuencia; etiquetas débiles de informes, 1 frase + cita). 3.2 Encoder 2D por corte + atención sobre cortes. 3.3 **Label-query attention** sobre series (ecuación). 3.4 **Poda de series**: ranking por masa de atención acumulada en validación por hallazgo; presupuesto k series; política de unión entre hallazgos. **Fig. 1 (ancho de columna o 2 col): arquitectura esquemática** (cajas, sin imágenes reales) con mapa de atención hallazgo×serie como inset (heatmap numérico, sin píxeles de paciente). | ~650 |
| 3 | **4. Experimental setup** (~300): split fijo 5 folds, 4,407 estudios, métrica macro AUC; pseudo-OOF como métrica principal, gold 58 con IC bootstrap como veto/sanidad, LB privado Kaggle como test externo ciego; hardware (Kaggle T4 y CPU), cómo se mide s/estudio (incluye lectura DICOM o no — declararlo), GFLOPs, params. **5. Results** (~500): **Tabla 1** (abajo). **Fig. 2: Pareto AUC vs s/estudio** (log-x): puntos de backbones exp002–008, nuestro modelo completo, curva de poda k=1..6, y referencia del stack público (~0.943 LB, 6–8 h). | ~800 |
| 4 | Resultados (cont.): análisis por hallazgo (texto + 1 línea de Tabla 1 o mini-tabla: qué serie elige cada hallazgo — coherencia clínica, p.ej. sagital PD para ACL, coronal para MCL). **6. Discussion & limitations** (~250): gold n=58 (IC amplio), etiquetas débiles, un solo centro/dataset, tiempos dependen de hardware. **7. Conclusion** (~80). | ~450 |
| 5 (200 USD) | Compliance with Ethical Standards (datos públicos de challenge, sin revisión ética adicional — confirmar redacción con licencia MIRA), Acknowledgments + COI + declaración de uso de IA, References (~15–20). | — |

**Tabla 1 (columnas):** Método · Pseudo-OOF macro AUC (5 folds) · Gold AUC (58, IC 95%) · Kaggle LB privado · s/estudio (T4) · s/estudio (CPU, opcional) · GFLOPs/estudio · M params.
**Filas:** (a) Referencia stack público (solo LB y tiempo, tomado del notebook; citar) · (b) Nuestro mejor modelo/ensemble completo · (c) Modelo pequeño + agregación mean (control) · (d) Modelo pequeño + label-query, todas las series · (e–g) (d) con poda a k = 4, 2, 1 series por hallazgo · (h) poda aleatoria a igual k (control de la política).

---

## 3. Experimentos mínimos para ISBI vs reservados para journal

### Mínimo indispensable (si falta alguno, no hay paper)
1. **Ablación de agregación (una variable):** mean/attention-pooling de series vs label-query, mismo backbone pequeño, mismo cache v2, fold 0 + 5 folds si promete. → filas (c) vs (d).
2. **Curva de poda (solo inferencia, barata):** sobre el modelo (d) ya entrenado, k = 6,4,3,2,1 series; política por atención vs aleatoria vs "solo el plano canónico". → Fig. 2 y filas (e–h).
3. **Benchmark de tiempo reproducible:** un notebook de Kaggle (T4, mismo entorno que el de envío) que mide s/estudio de extremo a extremo con N≥100 estudios, 3 repeticiones, mediana. Mismo script para todos los puntos de la Fig. 2. Medir GFLOPs con la misma herramienta (fvcore/thop) para todos.
4. **≥1 submission a Kaggle** del modelo podado y del completo, para tener LB privado de ambos (decidir antes del 22-oct cuáles son las 2 submissions finales; recordar que las seleccionadas cuentan para el premio).
5. **Puntos de backbones** exp002–008: ya existen AUC; falta s/estudio medido con el script del punto 3 (solo inferencia).
6. Gold 58 con **IC bootstrap** para todas las filas (cálculo, sin entrenar).

Cada uno de 1–2 debe nacer como experimento SDD (`nuevo_experimento(...)`, SPEC confirmada, una variable vs padre). Los puntos 3, 5 y 6 son medición/evaluación; conviene una SPEC de "benchmark" para que los números salgan de `metrics.json`.

### Reservado para el journal (CMIG) — esto es el "≥40% nuevo"
- Pipeline de etiquetas LLM v1 vs v2, calibración con gold, efecto del ruido de etiqueta.
- Destilación ensemble → estudiante (si no entra como fila (b) del ISBI).
- Poda de **cortes** además de series; presupuesto adaptativo por estudio (early-exit).
- Validación externa MRNet (ACL, menisco, anormal).
- Estudio completo de backbones (7+), CPU vs GPU, cuantización.
- Interpretabilidad con imágenes (si MIRA lo permite), análisis de error por hallazgo, varias semillas.

---

## 4. Cronograma día a día (2026-10-06 → 2026-10-26)

Supuestos del BRIEF: Fase A (subir AUC) hasta ~12-oct; merge de equipos 15-oct; Fase B (eficiencia) 12–22-oct; entrenamientos terminan ~20-oct; cierre Kaggle 22-oct.
Roles: R = Rodrigo (modelo + decisión); E1/E2 = compañeros (figuras, bibliografía, plantilla, benchmark de tiempos, revisión de forma).

| Fecha | Kaggle / experimentos | Paper |
|---|---|---|
| Mar 06 | Fase A en curso | Descargar plantilla ISBI; repo LaTeX privado (Overleaf); crear cuenta EDAS para todos los autores; fijar tesis y título. |
| Mié 07 | Fase A | E1: bibliografía (15–20 refs: MRNet, KneeNet, Query2Label/ML-Decoder, MIL attention, eficiencia médica, RSNA challenges). R: escribir SPEC exp label-query (ablación 1). |
| Jue 08 | Lanzar label-query fold 0 (si cabe en Fase A, mejor: es también mejora de AUC) | Borrador Intro (R) |
| Vie 09 | | Borrador Related work (E1 con R). E2: script de benchmark de tiempo en Kaggle (T4) + GFLOPs. |
| Sáb 10 | | Borrador Method 3.1–3.3 (sin números). Fig. 1 v0 (E2, draw.io/TikZ, sin imágenes de pacientes). |
| Dom 11 | Resultado label-query fold 0 | Ética/AI-disclosure/ack redactados. Revisar licencia MIRA para la frase de ética y figuras. |
| Lun 12 | Fin Fase A. **Decidir modelo base** para el paper. | Medir s/estudio de exp002–008 con el script (E2). |
| Mar 13 | label-query 5 folds (si promete) | Method 3.4 (poda) escrito. |
| Mié 14 | **GATE GO/NO-GO ISBI:** ¿label-query ≥ control en fold 0 y la poda en fold 0 muestra curva no trivial (k=2 retiene ≥95% del AUC)? Si NO → plan B (abajo). | Setup (sección 4) completo. |
| Jue 15 | Merge de equipos (deadline Kaggle) | Esqueleto de Tabla 1 y Fig. 2 con datos de exp002–008 (script de plot listo, se rellena solo). |
| Vie 16 | Curva de poda k=1..6 + aleatoria sobre fold 0 | Primera versión completa con huecos `\TODO{}`. Revisión interna 1 (E1/E2: forma, longitud). |
| Sáb 17 | Fase B (destilación/estudiante) | Pulir intro/método; recortar a 4 págs. |
| Dom 18 | | — |
| Lun 19 | Curva de poda sobre OOF 5 folds | Rellenar Tabla 1 con pseudo-OOF y gold+IC. |
| Mar 20 | **Fin de entrenamientos. Congelar números** (todo desde `metrics.json`). Benchmark final de tiempos de las filas (b)–(h). | Results y Fig. 2 definitivos (salvo LB). |
| Mié 21 | Elegir 2 submissions finales (completo y podado) | Discusión, limitaciones, abstract. Revisión interna 2. |
| Jue 22 | **Cierre Kaggle.** | Pasada de consistencia de números (texto vs tabla vs `metrics.json`). |
| Vie 23 | LB privado (si ya está publicado) | Insertar LB privado; si aún no sale, usar LB público y marcarlo como tal. |
| Sáb 24 | | Versión "release candidate". Lectura completa en voz alta; chequeo de plantilla (márgenes, fuentes embebidas, PDF compatible con IEEE PDF eXpress si lo piden). |
| Dom 25 | | Subir borrador a EDAS (se puede reemplazar hasta el deadline). |
| Lun 26 | — | Ajustes finales; **envío final antes de ~18:00 CDMX** (deadline real 21:59 CDMX). Después: decidir arXiv (permitido; recomendable tras el 22-oct). |

---

## 5. Riesgos y plan B

| Riesgo | Prob. | Mitigación |
|---|---|---|
| Label-query no gana al control o la poda no da curva limpia | Media | El paper puede pivotar al título 3 (estudio Pareto AUC–cómputo en 12 hallazgos con hardware real, backbones + poda como análisis). Decidir el 14-oct, no después. |
| No hay números a tiempo (entrenamientos se alargan) | Media | Escritura desacoplada; Tabla/Fig se generan por script desde `metrics.json`; los puntos de backbones ya existen. Fold 0 + LB pueden bastar para un ISBI si se declara. |
| Gold n=58 → IC enorme; revisor lo ataca | Alta | Métrica principal = pseudo-OOF (4.4k estudios) + LB privado (~900 estudios ciegos); gold solo como sanidad, con IC explícito. |
| LB privado no publicado antes del 26-oct | Baja–media | Usar LB público (~390 estudios), declararlo, actualizar en camera-ready (26-ene). |
| MIRA: imágenes en figuras / pseudo-etiquetas | Media | Fig. 1 esquemática, heatmap hallazgo×serie numérico; no liberar etiquetas ni datos. Preguntar a RSNA/MIRA para el journal. |
| Anonimato | Baja | Single-blind: no aplica. Sí aplica: no publicar código antes del 22-oct (regla Kaggle). |
| Presencia obligatoria en Lausana (25–28 mayo 2027), costo | Media | Al menos 1 autor viaja; presupuesto ~1,050–1,350 USD de registro (ref. 2026) + 200 USD 5ª página + viaje. Mexicanos no requieren visa Schengen para estancias cortas [INFERIDO; verificar ETIAS]. Si nadie puede viajar → **no enviar** (se retira de proceedings igual). |
| Choque con la versión journal | Baja | No enviar a CMIG antes del 12-ene-2027; journal con ≥40% nuevo (sección 3), citar ISBI, declararlo. |
| Sobrecarga del equipo durante la recta final de Kaggle | Alta | Kaggle tiene prioridad hasta el 22-oct; el paper usa solo lo que Kaggle ya produce + la ablación de agregación (que también sube AUC). E1/E2 llevan bibliografía, figuras y benchmark. |

### Plan B (si el gate del 14-oct falla o nadie puede viajar)
1. **Saltar ISBI e ir directo a journal (CMIG)** con la historia completa (etiquetas LLM + agregación + poda + MRNet + Pareto). Sin restricción de extensión, envío objetivo enero–febrero 2027. Opción preferida.
2. **MIDL 2027** (full o short paper): CFP aún no publicado; en 2026 el deadline fue 5-dic-2025 (https://mldeadlines.com/conference/midl/), así que esperar ~dic-2026. Ojo: MIDL es double-blind (verificar en el CFP 2027).
3. **ISBI 2027 one-page abstract** (1-feb-2027): solo póster, no proceedings → no bloquea el journal, pero exige viajar igual. Útil solo si alguien viaja de todas formas.
4. MICCAI 2027 (deadline típico ~marzo) [INFERIDO] — más competido; solo si hay resultados fuertes de poda adaptativa.
