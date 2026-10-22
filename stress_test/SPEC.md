# SPEC — stress_test_v1

Estado: borrador de diseño; no se han creado manifests de casos ni se ha iniciado entrenamiento.

## Objetivo

Crear un benchmark privado, difícil y reproducible para medir generalización del tanque de AUC y de candidatos futuros. El objetivo no es maximizar una métrica artificial de corrupción: es averiguar qué modelos son estables en dominios y condiciones de imagen que no dominaron el entrenamiento.

## Hipótesis

1. Un panel de dominio no visto y un panel de estrés de imagen distinguirán fallas que el AUC limpio agregado oculta.
2. El ranking de robustez podría transferir al leaderboard, pero solo se usará para decidir si el análisis histórico fuera de familia y temporal lo respalda.
3. En empates públicos dentro del ruido observado, el modelo con mejor robustez OOD podrá ser preferible; una mejora en stress test, por sí sola, no prueba una mejora en el private leaderboard.

## Unidad de evaluación y protección de datos

- Fuente: estudios de entrenamiento del reto, con `StudyInstanceUID` y datos DICOM existentes en el volumen privado.
- No incluir test de Kaggle, duplicar DICOM, informes, etiquetas ni predicciones por paciente en repositorio, Kaggle o artefactos públicos.
- Los `StudyInstanceUID`, mapas de casos, predicciones individuales, etiquetas y manifests completos son datos restringidos: vivirán solo en el volumen privado de RunPod, fuera de Git. En el repo solo van código y documentación sin IDs; `metrics.json` versionable debe ser agregado y no reidentificable.
- Generar condiciones al vuelo desde la caché existente cuando sea posible. Guardar en el volumen privado un manifest determinista con IDs, transformación, parámetros/semilla y hashes de artefactos; en Git, solo métricas agregadas y configuración no sensible.
- Los 58 estudios gold nunca eligen casos difíciles, transformaciones, hiperparámetros, pesos ni modelos; solo conservan su rol de veto según las reglas del equipo.
- Etiquetas pseudo: usar el target acordado en la SPEC del experimento que se evalúa, registrar su fuente y no mezclar fuentes silenciosamente. Reportar por separado resultados sobre pseudo y gold.

## Paneles del benchmark

### A. Subgrupos de adquisición (histórico) y dominio no visto (futuro)

Crear un artefacto auxiliar `stress_domain_v1`, sin modificar `split_v1_seed42`. Sus grupos candidatos son `ManufacturerModelName`, fabricante y campo magnético (`vendor`, `field` del split canónico). Antes de elegir grupos se publican conteos por estudio y por clase; se incluyen solo dominios con soporte positivo/negativo suficiente para estimar AUC y se reportan los grupos excluidos y la razón.

Los OOF existentes de `split_v1_seed42` permiten análisis de subgrupos, no prueban generalización a un dominio jamás visto: ese split estratifica fabricante/campo/idioma. Para llamarlo OOD real, cada grupo evaluado debe estar excluido por completo del entrenamiento del checkpoint. La auditoría histórica debe marcar por separado `subgroup_only` y `true_group_holdout`; si no hay checkpoints apropiados, el panel OOD queda pendiente para entrenamiento futuro, sujeto al freeze vigente y a SPEC propia. Si el soporte obliga a combinar categorías, la regla se fija mirando solo metadatos/conteos, antes de inspeccionar métricas. Sexo e idioma son análisis secundarios, no shift del escáner.

### B. Casos difíciles

- Auditar primero qué OOF hay. Una predicción vale para dificultad solo si ese estudio estuvo fuera del entrenamiento del checkpoint. Excluir gold del descubrimiento.
- Construir conjuntos disjuntos por estudio (nunca por imagen/serie): `difficulty_discovery` para definir dificultad, `hard_dev_v1` para iterar y `hard_lockbox_v1` evaluado con modelos de familias que no participaron en descubrir/ajustar el lockbox.
- Definir el lockbox con una regla congelada y auditable: usar solo OOF de familias de descubrimiento para elegir casos del lockbox, y reservar familias evaluadoras distintas; congelar IDs y fórmula antes de abrir sus métricas. Si no hay suficientes familias/OOF, el panel se etiqueta provisional y no selecciona modelos.
- La dificultad se calcula sin gold y con outputs OOF únicamente. Registrar fórmula, familias de descubrimiento, familias evaluadoras y universo elegible antes de materializar IDs.
- Si no hay suficientes predicciones OOF de varias familias, no declarar que el panel mide dificultad general: reportarlo como benchmark provisional y no seleccionar modelos con él.

### C. Estrés de adquisición e imagen

Partir del catálogo ya ejecutado en `exp040_robustez`; no volver a inventar perturbaciones sin motivo. Las familias iniciales son ruido Rician/Gaussiano, gamma/contraste/bias field, resolución/encuadre, desplazamiento/submuestreo de cortes, ventanas/series ausentes y espejo anatómico válido. Etiquetar cada condición como `plausible` o `severe`; no mezclar los dos niveles en un solo promedio.

La severidad se parametriza en un manifiesto determinista. Transformaciones anatómicamente inválidas se excluyen. Cada condición se aplica consistentemente por estudio/serie y conserva las máscaras de cortes ausentes.

## Modelos y protocolo

- Registrar los candidatos por experimento, familia, padre, fecha, versión del kernel, score LB y disponibilidad de pesos, OOF y predicciones de estrés.
- Una predicción in-sample no es válida para el benchmark. Modelos sin checkpoint/predicción OOF se marcan `not_evaluable`; no se rellenan con resultados del submission CSV.
- Primera referencia: modelos con OOF reproducible y el miembro Raptor que se pretende robustecer. El componente R6/compact se mantiene fijo al evaluar el ensamble. Las entregas históricas sin checkpoint/OOF no se imputan ni se evalúan con predicciones del test.
- Evaluar limpio, grupos no vistos, hard-dev, lockbox y cada corrupción por separado. Usar bootstrap pareado por estudio y reportar intervalo, n y soporte de ambas clases.
- Métrica principal de robustez: vector de (AUC limpia, macro AUC por dominio no visto, peor dominio, delta por condición plausible, peor delta plausible). No fabricar una única cifra antes de validar su transferencia.

## Transferencia al leaderboard

Aplicar el protocolo de `TRANSFER.md` a las entregas históricas evaluables. Un candidato solo gana preferencia por robustez en un empate público dentro de ±0.003 si la mejora en lockbox/OOD es reproducible y el análisis de transferencia no contradice esa regla. El leaderboard privado no se consulta ni se usa para ajustar el benchmark.

## Gates de calidad del benchmark

1. Reproducibilidad: misma versión y semilla producen idénticos IDs/transformaciones y métricas, salvo tolerancia numérica documentada.
2. Integridad: cero IDs duplicados entre `hard_dev_v1` y `hard_lockbox_v1`; cero estudios entrenados incluidos en la evaluación de su propio modelo.
3. Cobertura: informe de conteos de clase y dominio por panel/condición; grupos sin soporte suficiente se muestran como no estimables.
4. Validez: la condición limpia reproduce la referencia exp040 del mismo checkpoint dentro de la tolerancia fijada en el plan.
5. Transferencia: estimar Spearman/Kendall entre robustez y score público con bootstrap agrupado por familia y una validación cronológica. Si el intervalo es compatible con correlación nula o hay muy pocas familias, el benchmark sirve para diagnóstico OOD, no para predecir LB.

## Experimento GroupDRO posterior

El entrenamiento es un experimento separado y requiere número global, SPEC confirmada y notebook propio. Una sola variable de entrenamiento: ERM → GroupDRO por grupos de adquisición predefinidos. Arquitectura, targets, augmentations, inicialización y calendario quedan fijados al padre. Screen en split válido, selección por pseudo, gold solo veta. Solo se prepara candidato all-data si pasa el gate pre-registrado. El freeze de entrenamiento vigente debe respetarse; este benchmark no lo levanta.

## Riesgos y límites

- La AUC de Kaggle tiene ruido y el score público es solo una fracción del test; ningún stress test sobre train demuestra victoria en el private leaderboard.
- Los subgrupos pequeños producen AUC inestable; informar soporte e intervalos, no ordenar dominios sin potencia.
- Seleccionar los casos difíciles usando muchos modelos crea sesgo adaptativo; por eso hay descubrimiento, lockbox y cross-fitting por familia.
- Un artefacto sintético extremo puede no representar un cambio clínico real. Las condiciones `severe` son diagnóstico, no criterio de selección por sí solas.
- Algunas versiones históricas quizá no tengan OOF/checkpoints recuperables. La cobertura real se debe medir antes de prometer evaluación de los ~70 envíos.
