# Plan etiquetas v2 — ¿son las pseudo-etiquetas el cuello de botella?

Fecha: 2026-10-14. Autor: revisión Claude para Rodrigo. Estado: **propuesta, sin SPEC**. Nada de esto se ejecuta
sin SPEC confirmada (regla 1). No se gastó API ni se crearon pods para escribirlo.

## 0. Hallazgo bloqueante

**exp010 (dev-200 humano) no está anotado.** Los dos XLSX (`~/Downloads/RSNA_anotacion_exp010/` y
`/workspace/data/labels/exp010_anotacion/`, 120 filas cada uno, 40 en común) tienen las 1440 celdas de etiqueta
vacías. `results/exp010_dev_informes/metrics.json` dice `status: pending`. Solo existe la traducción
(gpt-5.4-mini, $0.28). Hoy **no hay vara humana**: cualquier etiquetador nuevo solo se puede juzgar por LB.

## 1. Inventario

AUC macro de cada fuente contra gold-58 (descriptivo, calculado hoy en CPU con `scratchpad/labauc.py`; gold **no**
eligió nada, regla 6). n = 58: el error estándar del macro es ~0.01–0.015, así que diferencias < 0.02 son ruido.

| Fuente | Origen / modelo | Formato | Costo | AUC gold-58 |
|---|---|---|---|---|
| v1 | Qwen2.5-7B local, propio | 1/0/0.5 ("no mencionado" = 0.5) | GPU | 0.803 |
| Raptor | `dreaddevelopment/rsna-knee-labels`, LLM, 14 niveles | blando | público | — (no cubre gold) |
| L2 | stevenleehans `llm_labels_v4_blend` (= v2 + GPT-5.6-Sol) | blando | público | 0.893 |
| L3 | pilkwang `report_labels_v2` (YES/NO/UNK→.82/.08/.28) | 5 niveles | público | 0.870 |
| L4 | lixin73 GPT-5.6-Sol | casi binario | público | 0.835 |
| L5 | propio, gpt-4.1-mini × 3 prompts | blando | $8.78 | 0.889 |
| L5v2 | L5 + gpt-5.4-mini strict + gpt-5.4-nano checklist | blando | +$11.9 | 0.896 |
| Consenso 4 (L2–L5) | media | blando | — | 0.895 |
| Consenso dedup ((L2+L4)/2, L3, L5) | media | blando | — | 0.895 |
| JEV HYBRID | riadmohamed42; hard-swap elegido **con gold** | blando | público | 0.917–0.921 (gold-74, contaminado) |
| blendjev (nartaa) | Steven + pilkwang + JEV + propias + OAI | — | **privado** | — |

Por hallazgo, el techo de todas las fuentes de texto es parecido: ACL/MCL 0.97–0.99, Medial Meniscus/Baker's ~0.95,
**débiles: Synovitis 0.79–0.83, Lateral OA 0.80–0.82, Fracture 0.80–0.88, Contusion ~0.84, Lateral Meniscus ~0.88,
Effusion ~0.88.** Acuerdo entre fuentes (unanimidad binaria sobre la muestra exp010) por idioma: hr 0.68, tr 0.79,
bg_ru 0.79, fr 0.80, en 0.84, es 0.89.

## 2. Diagnóstico: ya no son el cuello principal

1. **Meseta del texto.** Pasar de Qwen-7B (0.80) a fuentes GPT (0.89) valió +0.024 LB (exp021). Desde ahí, cinco
   etiquetadores y sus consensos quedan en 0.87–0.90 contra gold. Más LLM o más lecturas no mueven la AUC de la
   etiqueta (L5→L5v2 +0.007, dentro del ruido; sub056_2 +0.0017 interno; JEV como 5.ª fuente, sub031_1, didnt).
2. **El modelo ya supera a su maestro.** exp045 5 folds: gold 0.924 > consenso 0.895 contra gold. La red promedia el
   ruido de etiqueta (Karimi 2020; Rolnick 2017). Lo que queda es en buena parte desacuerdo informe↔imagen y
   definiciones estrictas (derrame moderado/grande, OA > 50 % de cartílago), no errores de lectura.
3. **Pendiente empírica.** Δ AUC-etiqueta → Δ LB ≈ 0.27 (exp021). Subir el consenso de 0.895 a ~0.92 (nivel
   JEV) daría como mucho ≈ +0.007 LB. Con la red fuerte (régimen Raptor) la pendiente es menor: exp031 +0.006 LB,
   L5 ≈ +0.002–0.004. **Techo realista con etiquetas "perfectas" del informe: +0.005–0.010 LB. Esperado con
   un etiquetador nuevo: +0.000–0.004, que es el ruido del LB público (~390 estudios).**
4. **La brecha grande está en otro sitio.** Nuestro mejor modelo da 0.943 frente a 0.954 de blend-mine y 0.957
   de R7+Raptor. exp044 (mismas etiquetas, init nartaa/OAI) llega a 0.953 en blend. El cuello es **init/pretraining
   y ensamble**, no etiquetas.

## 3. Opciones evaluadas

| Opción | Veredicto |
|---|---|
| Otro LLM fuerte (GPT-5.x con razonamiento, Claude, Gemini) sobre los 6 hallazgos débiles, con definiciones oficiales + cita de evidencia | **Sí, acotado**: es la única con margen (los hallazgos débiles) |
| Traducir antes de etiquetar (hr, tr, bg_ru) | Sí, dentro de la opción anterior; el costo es marginal (~$6 el corpus con mini) |
| Self-consistency / más lecturas del mismo modelo | No: medido, ≈ 0 (L5v2) |
| Ensamble de familias distintas (Qwen3/MedGemma local) | Bajo: necesita pod, calidad ≤ GPT; la diversidad ya está saturada en 0.895 |
| Fine-tune / few-shot con humanos exp010 | No antes del 22-oct: no hay anotaciones; 200 ejemplos sirven para **elegir**, no para entrenar |
| Etiquetas por hallazgo con incertidumbre (soft) | Ya lo tenemos (L5, 5 estados); MAPLEZ 2024: lo blando por sí solo no es significativo |
| Dawid-Skene / dedup L2-L4 (LAB-1/2) | Cambian 703 celdas; esperado +0.000–0.003; sin vara humana solo decide LB |
| Copiar a los públicos | JEV HYBRID usa gold para elegir fuentes (prohibido para nosotros). blendjev es privado |

## 4. Plan rankeado

**P0 — Rescatar la vara humana (hoy–16 oct, $0, ~3 h-persona).** Pedir a Alejandra y José Javier **solo los 6
hallazgos débiles** de sus 120 filas (40 en común para κ). Sin esto, nada de lo que sigue se puede validar sin LB.
Rodrigo resuelve los desacuerdos.

**P1 — L6 "definición estricta + evidencia" (15–16 oct, ≤ $25 API, 0 GPU).** Un modelo GPT-5.x de razonamiento,
**solo** sobre Synovitis, Lateral OA, Fracture, Contusion, Lateral Meniscus y Effusion. Informe original + traducción
cacheada. Salida: estado (present/absent/not_mentioned/uncertain) + frase textual. Mapear con `STATUS_P` de L5. Antes de
gastar en el corpus completo, piloto con los 200 de exp010 (< $2).
Decisión (fijada antes de mirar): L6 entra si su AUC/κ contra humanos en los 6 hallazgos ≥ L5 + 0.03 y no empeora
ningún hallazgo > 0.02. Gold solo veta: L6 contra gold no cae > 0.02 frente a L5.

**P2 — 1 envío LB (16–17 oct, ~5 GPU-h ≈ $3).** Exp hijo de exp056 (consenso 5 + aug espacial, all-data, seeds
42 y 7). **Una variable**: L5 → L6 en los 6 hallazgos (o L6 como 6.ª fuente; elegir una en la SPEC). Mismo TTA,
misma inferencia, envío solo. `worked` si LB ≥ 0.943 + 0.004. Si da < +0.004, **se cierran las etiquetas**: se
registra como hallazgo del paper (C1: meseta del texto en 0.895) y todo el GPU va a R7/init.

**P3 — Opcional, solo si P2 da worked:** reentrenar el bloque que entra al ensamble final (el que use R7) con las
etiquetas nuevas.

**No hacer:** LLM local en pod, fine-tune de LLM, más relecturas del mismo modelo, usar JEV HYBRID (contaminado por
gold).

## 5. Fuentes

- Irvin 2019, CheXpert — https://arxiv.org/abs/1901.07031
- Smit 2020, CheXbert — doi:10.18653/v1/2020.emnlp-main.117
- Jain 2021, efecto de la calidad del labeler en el modelo de imagen — https://arxiv.org/abs/2104.00793
- Karimi 2020, etiquetas ruidosas en imagen médica — https://arxiv.org/abs/1912.02911
- Rolnick 2017, deep learning robusto a ruido masivo de etiqueta — https://arxiv.org/abs/1705.10694
- Dunnmon 2020, cross-modal data programming (informe → imagen) — doi:10.1016/j.patter.2020.100019
- Plaza-del-Arco 2023, agregación de LLMs como multitud — https://arxiv.org/abs/2307.12973
- Park 2024, etiquetas LLM en RM de columna — https://arxiv.org/abs/2410.17235
- Le Guellec 2024, LLM open-source sobre informes (Radiology: AI) — doi:10.1148/ryai.230364
- Mukherjee 2023, Vicuna para etiquetar informes (Radiology) — doi:10.1148/radiol.231147
- Dawid & Skene 1979 — doi:10.2307/2346806
- Kaggle: `riadmohamed42/jev-knee-labels`, `karttikjangid05/rsna-knee-public-llm-label-sets-compared` (en gold-58,
  11/12 hallazgos no distinguen la mejor fuente de la mediana); detalle en `docs/pasos/raptor_receta.md` §2.
- Interno: `docs/conocimiento/01_etiquetas_llm.md`, `RSNA_papers/analisis/A3_etiquetas_llm.md`,
  `docs/pasos/tablero_paso7.md`, notas_sesion6.md (costos de L5 y L5v2).
