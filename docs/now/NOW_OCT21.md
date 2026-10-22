# Handoff para Claude Code — 21-oct-2026

Este archivo permite retomar el trabajo sobre el benchmark privado `stress_test` sin depender del historial del chat. Está escrito el 21-oct-2026 (America/Mexico_City). **No reemplaza el NOW canónico**: `docs/now/NOW.md` tiene fecha 19-oct y debe revalidarse contra Kaggle y RunPod antes de confiar en sus estados.

## Resumen ejecutivo

Rodrigo quiere priorizar generalización para el reto RSNA Knee. Propuso un benchmark privado, difícil y reproducible, basado en estudios de entrenamiento existentes: generar perturbaciones al vuelo, guardar solo manifiestos restringidos y métricas agregadas, identificar casos difíciles y comprobar si la robustez predice el leaderboard público. No quiere crear un Kaggle Dataset ni duplicar/publicar DICOM.

Codex creó la carpeta `stress_test/` con una primera propuesta documental. **Todavía no existe un benchmark ejecutable:** no se materializaron manifests/IDs, no se auditaron los OOF/checkpoints disponibles, no se implementó código, no se corrió inferencia y no se entrenó nada. No decirle al usuario que ya se hizo el benchmark.

Siguiente paso recomendado: retomar como auditoría de evidencia, no empezar entrenando. Verificar el estado vivo, medir la cobertura real de modelos/OOF/checkpoints, concretar qué paneles sí son válidos y luego preparar un plan de implementación. Respetar SDD: no crear un experimento ni modificar notebooks/código compartido sin la SPEC requerida y confirmada.

## Qué existe en el repo

- `stress_test/README.md` — propósito, uso previsto y prevención de fuga/sobreajuste.
- `stress_test/SPEC.md` — borrador de especificación, aún pendiente de confirmación.
- `stress_test/TRANSFER.md` — protocolo propuesto para contrastar métricas OOF con scores públicos.
- `stress_test/RESULTS.md` — plantilla vacía de cobertura/resultados; no contiene resultados reales.
- `docs/now/NOW_OCT21.md` — este handoff y el prompt autocontenido de abajo.

El último `git status` observado por Codex fue rama `exp010_dev_informes` con `stress_test/` sin seguimiento; no había otros cambios en el status. No se hizo commit ni push. Revisa el status antes de editar. No borres ni reemplaces trabajo existente.

## Decisiones y límites que ya expresó Rodrigo

- Enfocarse en generalización del modelo de AUC, sin confundirla con la línea de eficiencia.
- Crear un benchmark privado basado en las mismas imágenes de train, con los casos más difíciles y transformaciones de estrés.
- Poder comparar experimentos históricos y estudiar si las métricas del benchmark se trasladan al public leaderboard.
- No crear Kaggle Dataset ni copias de DICOM; las transformaciones se generarían al vuelo.
- Tener un directorio y documentos propios para el benchmark.
- Dejar que Claude Code retome y orqueste; Rodrigo previamente pidió usar agentes para trabajo paralelo.
- Rodrigo rechazó revocar el token de GitHub que pegó en el chat. No repitas, leas, guardes, uses ni incluyas ese token en ningún prompt/log/archivo. Para este benchmark no se necesita token de GitHub.

## Hallazgos técnicos importantes ya establecidos

1. `experiments/exp040_robustez/SPEC.md` y sus scripts contienen un catálogo real de perturbaciones y un análisis de fold 0 + 58 gold. Entre lo observado: ruido Rician/Gaussiano de 5–10 %, menos cortes, resolución reducida y quitar series dañan; gamma/contraste/bias y pequeños encuadres fueron mucho más estables; espejo TTA aportó muy poco. Los números exactos deben copiarse de exp040 y no reconstruirse de memoria.
2. `split_v1_seed42` estratifica/mezcla fabricante, campo, idioma y otros grupos. Sus OOF sirven para análisis de subgrupos, pero **no demuestran OOD de un fabricante/campo nunca visto**. Para afirmar group-OOD, el dominio evaluado debe estar excluido del entrenamiento de ese checkpoint.
3. Solo OOF de un estudio que estuvo fuera del entrenamiento de ese modelo sirve para elegir casos difíciles/evaluar. Predicciones in-sample de modelos all-data no son válidas. `submission.csv` no tiene etiquetas de test y no sustituye OOF.
4. ~70 submissions no son ~70 observaciones independientes: agrupar reenvíos, variantes y descendientes por familia; reportar cobertura real y no prometer correlaciones predictivas si hay pocas familias/ventanas temporales.
5. El mismo conjunto de casos no debe servir indefinidamente para descubrir dificultad, ajustar el método y seleccionar entre decenas de modelos. La propuesta divide descubrimiento, desarrollo y lockbox. Ese lockbox es una barrera de proceso, no un enclave técnico para el dueño del volumen.
6. Public LB es una muestra parcial, redondeada y ya consultada repetidamente. Un vínculo con public LB no demuestra transferencia al private LB ni a hospitales externos.
7. En `docs/now/NOW.md` se declara freeze de entrenamiento desde 19-oct 12:00 UTC hasta el cierre de la competencia 22-oct 23:59 UTC. Verifícalo con Rodrigo/estado vivo antes de cualquier propuesta de entrenamiento; el benchmark de esta etapa debe ser auditoría e inferencia con artefactos existentes.

## Estado operativo que requiere revalidación

El NOW canónico del 19-oct lista `tank-v6`, `tank-v6soft` y `exp082` como pendientes/incompletos. Informes de auditorías posteriores en esta conversación indicaron:

- Los envíos históricos v1 revisados (`exp074-student224-fast`, `sub078-1-bigru224`, `exp080-ctrl224`, `sub075-1-solo`, `tank-seeds`, `tank-v6`, `tank-v6soft`) ya fueron enviados; no se encontró un candidato validado pendiente en la cola revisada.
- `exp082-bench-ens` aparece COMPLETE con runtime aproximadamente 1982 s frente a baseline 252 s; bajo su regla predefinida el umbral estimado de LB para justificarlo era 0.975. El log parecía técnicamente completo, pero no hay evidencia de que se haya enviado ni score LB.

Estas son salidas de agentes previas, no comprobaciones en vivo realizadas en esta sesión. Claude debe consultar estados actuales antes de cambiar `NOW.md`, encolar o enviar algo. El objetivo actual es `stress_test`, no gastar un envío ni hacer un submission.

## Orden de trabajo recomendado para Claude Code

1. Leer en el orden obligatorio de `AGENTS.md`: `docs/now/NOW.md`, `docs/now/CODEX.md`, `docs/00_EMPIEZA_AQUI.md`, `docs/02_REGLAS_DEL_EQUIPO.md`, `docs/06_SDD.md`, `docs/EXPERIMENTOS.md`, `docs/07_SPLIT.md`, `docs/04_DATOS.md`, `docs/05_PLAN_DE_MODELOS.md`, `docs/08_LO_QUE_HACEN_OTROS.md`, más SPECs relevantes (`exp040` y padres/candidatos que se encuentren).
2. Revalidar fecha, rama/status, freeze, resultados Kaggle y RunPod sin imprimir `.secrets.env` ni ninguna credencial.
3. Hacer una auditoría solo lectura de OOF/checkpoints/predicciones y scores disponibles. Priorizar evidencia local/RunPod y las reglas del proyecto; no descargar/ejecutar notebooks remotos no confiables. Usar varios agentes para preguntas independientes (inventario OOF; inventario submissions/model families; revisión metodológica), cada agente con alcance read-only.
4. Llenar `stress_test/RESULTS.md` solo con números respaldados por archivos concretos. Separar `evaluable`, `not_evaluable` y razón; no imputar resultados.
5. Revisar si la SPEC propuesta es implementable y si el protocolo de dificultad/lockbox es estadísticamente viable con la cobertura encontrada. Si hay preguntas que solo Rodrigo puede contestar, presentarlas juntas; no inventar aprobación.
6. Si hay un benchmark realizable dentro del freeze, proponer auditoría + smoke/inferencia únicamente, dejando claro el coste y el volumen privado. No crear pods GPU sin autorización/estado actual; no lanzar GroupDRO ni nuevo entrenamiento durante el freeze.
7. Para cambios de implementación, seguir SDD, TDD y el ciclo de revisión aplicable. Todo identificador de paciente, DICOM, etiqueta, predicción individual y manifest con UIDs vive solo en el volumen privado RunPod; nunca en Git/Kaggle. Solo métricas agregadas no reidentificables pueden versionarse conforme a las reglas.
8. Al terminar, actualizar este handoff o el NOW canónico según corresponda con hechos verificados, rutas, comandos/resultados, pendientes y costos; nunca declarar que el benchmark está listo si solo hay diseño.

## Prompt inicial largo para pegar en Claude Code

```text
Retoma el trabajo RSNA Knee desde este handoff: `docs/now/NOW_OCT21.md`.

Eres Claude Code y orquestas el trabajo. Rodrigo quiere avanzar con un benchmark privado de generalización llamado `stress_test`, enfocado en la línea de máximo AUC (separada de eficiencia). El benchmark debe aprovechar las mismas imágenes de entrenamiento existentes para analizar casos difíciles y cambios plausibles/severos de adquisición, generar transformaciones al vuelo, y estudiar si sus métricas ayudan a ordenar modelos respecto del public leaderboard. NO debe convertirse en Kaggle Dataset, no debe copiar/publicar DICOM y no debe usar el token de GitHub pegado en el chat.

ESTADO REAL: Codex solo creó documentos preliminares en `stress_test/README.md`, `SPEC.md`, `TRANSFER.md` y `RESULTS.md`. No hay dataset/manifests materializados, no hay código del benchmark, no hay auditoría completa de OOF, no hubo inferencia ni entrenamiento. La SPEC está en borrador. No presentes el benchmark como terminado.

PRIMERA FASE OBLIGATORIA — solo lectura:
1. Lee completamente los archivos del repo en el orden indicado por AGENTS.md: `docs/now/NOW.md` y `docs/now/CODEX.md`; `docs/00_EMPIEZA_AQUI.md`; `docs/02_REGLAS_DEL_EQUIPO.md`; `docs/06_SDD.md`; `docs/EXPERIMENTOS.md`; `docs/07_SPLIT.md`, `docs/04_DATOS.md`, `docs/05_PLAN_DE_MODELOS.md`, `docs/08_LO_QUE_HACEN_OTROS.md`; luego `experiments/exp040_robustez/SPEC.md`/scripts/resultados y las SPECs de candidatos relevantes.
2. Compara fechas: el NOW canónico visto por Codex estaba actualizado al 19-oct; hoy este handoff es del 21-oct. Revalida el estado de Kaggle, RunPod, freeze de entrenamiento, submissions, balance si es relevante y rama/status antes de cualquier acción con efectos externos. No confíes ciegamente en valores históricos de este archivo.
3. Inspecciona el árbol y los resultados existentes sin tocar nada todavía. No cargues ni imprimas `.secrets.env`, `.environment`, API keys o credenciales. No uses ni repitas el token de GitHub pegado anteriormente; no se necesita para esta auditoría.
4. Usa hasta tres agentes read-only en paralelo para preguntas realmente independientes: (A) inventario de OOF/checkpoints y cobertura por modelo/familia; (B) inventario de Kaggle submissions, fechas/scores/linajes y cola actual; (C) revisión metodológica de la SPEC/exp040 contra riesgos de fuga, selección adaptativa y métricas. Indícales expresamente que no editen, no entrenen, no envíen, no creen pods y no impriman secretos. Integra sus resultados solo con referencias a evidencia.

PREGUNTA CENTRAL:
¿Con los artefactos disponibles podemos crear un benchmark honesto que (i) evalúe robustez por familia de transformaciones, (ii) identifique estudios difíciles sin fuga y (iii) pruebe fuera de familia/tiempo si stress performance aporta señal predictiva sobre public LB? Contesta por separado para cada parte; no confundas diagnóstico de train con generalización a hidden/private leaderboard.

REGLAS METODOLÓGICAS QUE NO SE PUEDEN ROMPER:
- La dificultad/evaluación requiere predicciones OOF donde el estudio haya estado fuera del entrenamiento del checkpoint. Predicción in-sample/all-data no cuenta.
- El submission CSV de test no tiene etiquetas: nunca tratarlo como OOF o AUC etiquetada.
- `split_v1_seed42` no se modifica. Como sus folds mezclan fabricante/campo, esos OOF dan análisis de subgrupos, no una prueba de dominio jamás visto. Usa el rótulo `true_group_holdout` solo si el checkpoint dejó completamente ese dominio fuera del entrenamiento; si no, `subgroup_only`.
- Gold-58 no descubre casos, no elige transformaciones ni hiperparámetros; solo el rol limitado descrito en las reglas del equipo. Reporta gold por separado.
- Agrupa ~70 submissions por familia/linaje; no trates reenvíos/variantes relacionadas como 70 observaciones independientes. La transferencia a public LB es retrospectiva y el LB puede tener sesgo adaptativo; no afirma private LB/generalización clínica.
- Separa discovery/dev/lockbox por estudio. El lockbox es disciplina de proceso, no enclave técnico. Explica cómo se previene que las mismas familias descubran y validen su propio panel.
- Usa el catálogo comprobado de exp040, no inventes perturbaciones arbitrarias. Separa severidad plausible de severe; la severidad extrema es diagnóstica, no criterio de selección.
- Mantén UIDs, mappings de casos, etiquetas, predicciones por estudio y manifests completos solo en el volumen privado RunPod; no subas a Git/Kaggle ni los imprimas. En el repo, solo código/documentación y métricas agregadas no reidentificables.

LÍMITES OPERATIVOS:
- En NOW está declarado freeze de entrenamiento desde 19-oct 12:00 UTC hasta cierre 22-oct 23:59 UTC; confirma si sigue vigente. Por defecto, esta tarea es read-only más inferencia existente/auditoría, no entrenamiento ni GroupDRO.
- No crear ni encender pods GPU, no gastar API keys, no encolar kernels, no hacer submissions y no cambiar finales de Kaggle como parte del benchmark. Si una acción requiere gasto/estado externo o contradice el freeze, detente y presenta el bloqueo/opciones.
- No hacer push, no cambiar rama, no editar `src/knee/`, no crear un experimento numerado ni tocar notebooks hasta tener scope/SPEC compatibles con las reglas. Preserva el worktree y cualquier cambio de Rodrigo.

ENTREGABLE DE ESTA PRIMERA SESIÓN:
1. Auditoría respaldada de cobertura histórica: por experimento/familia indica si hay score LB, OOF usable y checkpoint; cita las rutas exactas y cuenta evaluable/no evaluable. Evita incluir UIDs o información sensible.
2. Dictamen claro por componente: hard cases, transformaciones de stress, subgrupos/OOD verdadero, y transferencia a public LB — viable ahora / provisional / no viable, con razón y límite.
3. Una SPEC ajustada de `stress_test_v1` si la existente es implementable con evidencia; si la cobertura la invalida, corrige el diseño a un benchmark más modesto y explica qué se pierde. No cambies gates después de ver resultados.
4. Un plan de implementación separado (no lo ejecutes todavía) con archivos/rutas, smoke, costes estimados, gates, outputs privados frente a versionables, y criterio para detenerse.
5. Actualiza `stress_test/RESULTS.md` únicamente con cifras comprobadas. Actualiza docs de estado solo con hechos revalidados; añade timestamp/fuente. No escribas secretos.

No preguntes de nuevo lo que ya está decidido. Si hace falta una elección que cambie materialmente el benchmark (por ejemplo, habilitar entrenamiento tras el freeze, usar una fuente de etiquetas diferente o abrir un lockbox), explica por qué no la puedes inferir y pregunta una sola vez con opciones concretas. Primero entrega a Rodrigo un resumen directo en español mexicano: qué sí se puede medir hoy, qué no, cuántos modelos son evaluables y cuál es el siguiente gate. No digas “ya quedó” hasta demostrarlo.
```
