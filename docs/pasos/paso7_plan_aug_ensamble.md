# Paso 7 — Subir encima del 0.954: ensamble + augmentations + preprocesado + etiquetas (2026-10-12, ~50 h)

Mandato de Rodrigo (12-oct): trabajar encima del 0.954 (blend-mine = velci sin su extra privado; CoAtNet nartaa +
ConvNeXt compact), sobre todo con ensamble, y con ablations de lo que ya nos funcionó; meter augmentations fuertes
(imagen y "texto"), mejoras de preprocesado. Hasta 7 pods, ~10 agentes más, ~50 h.

## Punto de partida (números reales)

| qué | pseudo L2 f0 | gold | LB |
|---|---|---|---|
| blend-mine / velci (público) | — (vio todo) | — | **0.954** |
| nartaa 0949 | — | — | 0.949 |
| exp029 Raptor nuestro, etiquetas Raptor | 0.8682 | 0.9115 | 0.933 |
| exp031 Raptor + etiquetas consenso (Raptor+L2+L3+L4) | **0.8836** | 0.9144 | pendiente (all-data listo) |
| sub031_2 (exp031 seed 7) | 0.8849 | 0.9172 | — |
| ensamble rangos seed 42 + seed 7 | 0.8878 | — | — |

Hallazgo clave para este paso: la receta Raptor **casi no tiene augmentation**. En `StudyWindows` solo hay ganancia de
intensidad ±10 % y muestreo aleatorio de ventanas. Sin flips, sin afines, sin ruido, sin gamma, sin mixup. exp040
(robustez de exp029) dice dónde duele: ruido 5 % −0.011, 50 % de cortes −0.007, sin sagital −0.028, sitio croata 0.817
vs inglés 0.880, ECE 0.20. Ahí apuntan las augmentations.

"Augmentation de texto": el modelo en inferencia solo ve imágenes; el texto (informes) solo entra por las etiquetas.
Entonces la augmentation de texto = etiquetas más ricas/robustas: muestreo estocástico de la fuente de etiqueta por
época, temperatura de las etiquetas blandas, y nuevas etiquetas LLM con prompts/paráfrasis distintas (L5).

## Reglas (SDD, igual que paso 6)

- Padre de todas las ablations: **exp031** (fold 0, 16 épocas, misma receta). Una variable por exp.
- Promete si Δ pseudo L2 ≥ **+0.003** vs exp031 (0.8836); ruido de semilla medido = 0.0013 (sub031_2).
  Gold cae > 0.02 → killed. Gold nunca selecciona.
- Si promete: sub-experimentos `subNNN_k` sobre la intensidad (una variable cada uno) y luego run **all-data** en el
  mismo pod → los pesos pasan al agente de ensamble.
- Los que funcionan se apilan en **exp056 combo** (uno a uno, cada adición es su propio sub056_k para no mezclar
  variables) → all-data → ensamble.
- Ensamble: pesos de rango fijos, decididos antes de mirar LB; el LB solo elige entre ≤ 2 variantes por día.
  Los modelos públicos vieron todos los datos, así que su OOF de fold 0 no sirve para pesar el ensamble.

## Tracks y experimentos

### T1 — Ensamble encima del 0.954 (Kaggle, agente de envíos/ensamble)
| exp | qué |
|---|---|
| exp038 | blend-mine + exp030 all-data / + exp031 all-data (rank 0.35/0.30/0.35) — envíos 12-oct |
| exp039 | blend-mine + bloque nuestro (exp031 all-data + seed 7 all-data, rank-promedio interno) |
| exp042 | blend-mine con el CoAtNet nartaa sustituido/sumado por exp044 (fine-tune de nartaa con nuestras etiquetas) |
| sub038_k | pesos del bloque nuestro 0.2 / 0.3 / 0.4 (máx 1 por día) |

### T2 — Augmentations de imagen (pods, padre exp031)
| exp | variable |
|---|---|
| exp045 | espaciales: flip L/R (anatómicamente válido, por eso funciona el TTA espejo) + afín (rot ±10°, escala 0.9–1.1, traslación ±6 %) |
| exp046 | intensidad: gamma 0.7–1.5, contraste, ruido gaussiano/Rician, blur, campo de sesgo suave (apunta al gap de sitio y al −0.011 de ruido) |
| exp047 | estructura: dropout de slot completo (p 0.15) + jitter de cortes/ventanas (apunta al −0.028 sin sagital y habilita poda de slots para eficiencia) |
| exp048 | mixup / cutmix a nivel estudio con etiquetas blandas mezcladas |
| sub045_k–sub048_k | intensidad de la aug ganadora (suave / fuerte) |

### T3 — Preprocesado (pods, padre exp031)
| exp | variable |
|---|---|
| exp049 | CLAHE por corte antes de normalizar |
| exp050 | resolución: usar el 384 completo en vez del recorte central 320 (batch/ventanas ajustados para caber) |
| exp051 | normalización: percentiles 0.5–99.5 por slot / z-score por volumen en vez de 2–98 |

### T4 — Etiquetas ("augmentation de texto")
| exp | variable |
|---|---|
| exp052 | muestreo estocástico de la fuente de etiqueta por estudio y época (Raptor/L2/L3/L4) en vez del promedio |
| exp053 | temperatura/afilado de etiquetas blandas (consenso^T, T 0.7) — sub053_k con T 1.5 |
| exp054 | L5: re-etiquetado de informes con GPT y varias paráfrasis/prompts (CPU + API), se suma al consenso |

### T5 — Lo que ya corre (paso 6)
sub031_1 (consenso + JEV, pod G), sub031_3 (lr backbone 6e-5, pod D2), sub031_4 (drop_path 0.2, pod F),
exp044 (fine-tune nartaa all-data, pod C2), exp031 seed 7 all-data (pod E2).

### T6 — Combo y eficiencia
| exp | qué |
|---|---|
| exp055 | eficiencia: exp031 all-data, 3 slots, sin TTA, half, 48–64 ventanas en T4 (agente harness T4) |
| exp056 | combo de todos los worked (sub056_k por adición) → all-data → ensamble |
| exp057–exp065 | reservados (seeds del combo, destilación a estudiante rápido, lo que salga) |

## Recursos

- Pods: los 5 actuales + 2 nuevos (H, I), 1 run por pod. Asignación inicial: H → exp045, I → exp046,
  D2 (tras sub031_3) → exp047, G (tras sub031_1) → exp052, E2 (tras seed 7) → exp049, F (tras sub031_4) → exp048,
  C2 (tras exp044) → exp050. Después, la cola sigue el orden de la tabla.
- Costo real medido: ~$0.74/h por pod (GPU + disco 200 GB + volumen). 7 pods ≈ $5.2/h → 50 h ≈ $260.
  El vigía apaga todo a < $4 (los pods se detienen, el volumen queda).
- Agentes: uno por track (T2 espacial/intensidad, T2 estructura/mixup, T3, T4 etiquetas, T4 L5) + los existentes
  (envíos/ensamble, harness T4, exp031, Raptor).
- Envíos: 5/día. Prioridad: (1) ensambles T1, (2) combo, (3) eficiencia.
