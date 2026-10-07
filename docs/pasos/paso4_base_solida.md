# Paso 4 — base sólida: etiquetas y entrada

Fecha: 2026-10-06. Referencia actual: exp002 (gold 0.788). Notebooks públicos: ~0.94.

Cambiar backbone no movió nada (exp003–exp008 entre 0.70 y 0.77 de gold). Las dos palancas grandes son:

1. **Etiquetas.** Las de Qwen2.5-7B (v1) coinciden ~0.80 con gold y "no mencionado" quedó como 0.5. El modelo aprende ese ruido.
2. **Entrada.** 224 px sobre la imagen completa. Los públicos recortan la rodilla (130–154 mm), usan 336–384 px y alinean
   izquierda/derecha. El cache v2 de exp009 ya existe y nadie lo ha usado.

Van en paralelo: etiquetas es trabajo de texto y personas, entrada es GPU.

## Experimentos, en orden

Cada número tiene un solo propósito. Los números 010–012 ya están reservados en `/workspace/users/.claims`.

| exp | Propósito | Padre | Depende de | Quién | Costo |
|---|---|---|---|---|---|
| exp010_dev_informes | Referencia humana: 200 informes del fold 0 etiquetados desde el texto. Sirve para elegir etiquetador sin gold. | exp001 | — | Alejandra + José Javier anotan; Rodrigo resuelve | ≤ US$5, ~8 h-persona |
| exp011_etiquetas_v2 | Comparar etiquetadores contra el dev y producir `report_labels_v2.csv` | exp001 | exp010 | Rodrigo | ~US$40 API |
| exp012_entrada_v2 | Receta exp002 con cache v2 (recorte físico, 336 px, alineado, 6 slots). Etiquetas v1 sin cambio | exp002 | — | (GPU, arranca ya) | ~3–4 GPU-h screen |
| exp013 (por reservar) | Receta exp002 con etiquetas v2 | exp002 | exp011 | | ~1.5 GPU-h screen |
| exp014 (por reservar) | Nueva referencia: entrada v2 + etiquetas v2, 5 folds | exp012/exp013 | ambos | | ~15–20 GPU-h |

### exp011 — candidatos a etiquetador

1. v1 tal cual (Qwen2.5-7B, "no mencionado" = 0.5).
2. v1 con "no mencionado" = 0 (gratis).
3. Datasets públicos de Kaggle con más uso: `stevenleehans/rsna-knee-llm-report-labels`, `pilkwang/rsna-knee-llm-labels`,
   `lixin73/rsna-knee-llm-report-labels-sol56` (GPT-5.6-Sol), `dreaddevelopment/rsna-knee-labels` (blandas),
   `laymond/rsna-knee-abnormality-qwen3-8b-weak-labels` (Qwen2.5-32B).
4. GPT por API con prompt nuevo: definición oficial por etiqueta, estado `present/absent/uncertain/not_mentioned`,
   lateralidad, cita textual de evidencia (para detectar alucinación), salida JSON, 12 idiomas.
5. Consenso de los 3 mejores.

Criterio: mayor macro AUC contra el dev de exp010 (incierto excluido). Gold solo veta: acuerdo con gold < v1 − 0.02 → fuera.
Las reglas permiten usar LLMs comerciales para sacar etiquetas de los informes (respuesta del host en el foro, regla 2.6.b).
Los datasets públicos se declaran en el paper.

### exp013 — cómo se compara con etiquetas distintas

La pseudo AUC del padre se midió contra v1. Para comparar justo, se re-puntúan las predicciones OOF del padre (`oof_f0.csv`)
contra las etiquetas v2 y se compara con el hijo medido contra v2. Gold sigue como veto.

### exp012 — puntos a fijar en su SPEC

- Slot faltante: máscara en la atención, no relleno con ceros (slot 1 cubre 94 %, slots 4–6 menos).
- Mismas épocas, lr y semilla que exp002. Solo cambia el cache y lo mínimo para leerlo.

## Después del paso 4

Con la referencia exp014 se sigue con la cabeza por hallazgo (label-query attention), la poda de series y la eficiencia.
Ver `docs/10_PLAN_PAPER.md`.
