# Paso 5 — síntesis de resultados de la sesión autónoma (exp012–exp022)

Fecha: 2026-10-09. Para el paper (ISBI 2027 + CMIG). Todos los números se copian de la sección 3 de cada `SPEC.md` y de
`results/expNNN_tag/metrics.json` en su worktree (`RSNA_wtNNN`, rama `expNNN_tag`), y del LB en `docs/pasos/envios_kaggle.md`.
Lo que no existe todavía dice "pendiente". Pseudo = pseudo AUC fold 0 contra etiquetas v1; gold = 58 estudios, fold 0.

## 1. Tabla maestra

| exp | Variable (una) | Padre | Pseudo f0 (v1) | Gold f0 | LB público | GPU-h | Veredicto |
|---|---|---|---|---|---|---|---|
| exp002 (ref.) | — | — | 0.827 | 0.777 | 0.812 (5 folds) | — | referencia |
| exp012_entrada_v2 | cache v2 (recorte físico, 0.5 mm, 336 px, 6 slots) | exp002 | 0.850 (+0.023) | 0.784 (+0.007) | pendiente (no enviado) | 3.65 | promising (5 folds pendientes) |
| exp013_cabeza_hallazgo | cabeza con 12 queries (una por hallazgo) | exp002 | 0.838 (+0.011)¹ | 0.789¹ | pendiente | pendiente | pendiente (5 folds en curso) |
| exp014_mnv4m_lr | lr backbone 1e-4 en MobileNetV4-M | exp003 | 0.737 (−0.006) | 0.633 (−0.068) | — | ~1.05 | killed |
| exp015_epocas16 | 16 épocas en vez de 8 | exp002 | 0.847 (+0.020) | 0.797 (+0.020) | 0.810 (solo fold 0) | 3.34 | promising (5 folds pendientes) |
| exp016_entrada_25d | 2.5D: vecinos como canales RGB | exp002 | 0.824 (−0.003) | 0.778 (+0.001) | — | 1.69 | didnt |
| exp017_atencion_dispersa | penalización de entropía λ = 0.05 | exp013 | 0.793 (−0.046) | 0.732 (−0.057) | — | 2.81 | killed |
| exp018_cabeza_hallazgo_v2 | cabeza por hallazgo sobre v2 | exp012 | 0.846 (−0.004) | 0.777 (−0.007) | — | 4.42 | didnt |
| exp019_poda_slots | poda de slots en inferencia, sin reentrenar | exp012 | 0.850 (6 slots; s1+s3+s5: 0.824) | 0.784 (s1+s3+s5: 0.785) | — | 0.18 | didnt |
| exp020_cabeza_epocas16 | exp013 con 16 épocas | exp013 | 0.850 (+0.012) | 0.802 (+0.013) | no enviado | 5.26 | killed por presupuesto (1.503×, GPU compartida); AUC promising, ganancias se suman |
| exp021_etiquetas_L2 | etiquetas de entrenamiento L2 en vez de v1 | exp002 | 0.789 (−0.038); contra L2: 0.817 (+0.035) | 0.812 (+0.035) | **0.836** (+0.024, solo fold 0) | 3.17 | worked (decide el LB; falta confirmación de Rodrigo) |
| exp022_entropia_suave | penalización de entropía λ = 0.005 | exp013 | 0.835 (−0.003) | 0.800 (+0.011) | no enviado | 2.91 | didnt (entropía 0.949 > 0.6: no concentra) |

¹ exp013 aún no tiene `metrics.json` ni verdict (worktree `RSNA_wt013`). 0.838 / 0.789 y el +0.011 se copian de la columna
"padre" de las SPEC de exp017, exp020 y exp022 (y del texto de exp018). **No verificados contra la fuente primaria**: confirmar
cuando exp013 cierre su verdict.

L2 = `stevenleehans/rsna-knee-llm-report-labels` (`llm_labels_v4_blend.csv`), candidato de exp011 con mayor acuerdo medio.
GPU-h con GPU compartida entre corridas (ver cada SPEC).

## 2. Hallazgos clave

- **Entrada v2 es la mayor ganancia en la vara interna:** +0.023 pseudo, gold +0.007 (exp012). Curva aún subiendo en 8/8.
- **La receta estaba subentrenada:** 16 épocas dan +0.020 pseudo y +0.020 gold (exp015); meseta desde la época 8, mejor la 12.
- **La vara interna v1 no predice el LB.** exp015 subió la pseudo pero quedó en 0.810 LB (exp002, 5 folds: 0.812). exp012 no se
  ha enviado, así que "v2 no mueve el LB" **no está medido**. En cambio exp021 (solo etiquetas L2) sube el LB +0.024 (0.836)
  mientras su pseudo v1 baja −0.038 y su pseudo contra L2 y gold suben +0.035. La señal del test está en las etiquetas.
- **La cabeza por hallazgo no es robusta:** +0.011 sobre v1 (exp013¹), −0.004 sobre v2 (exp018). La atención queda casi uniforme
  (entropía normalizada 0.992, coseno entre hallazgos 0.976 en exp018), con un sesgo débil pero clínicamente coherente
  (slots FS 1–3 reciben más masa; ACL, Baker y sinovitis al axial FS; meniscos y OA medial al coronal FS).
- **La entropía sola colapsa:** con λ = 0.05 los 12 hallazgos miran la misma serie coronal (entropía 0.011, coseno 1.000) y
  pierde −0.046 pseudo / −0.057 gold (exp017). λ = 0.005 (exp022) no pierde AUC pero tampoco concentra (entropía 0.949): la entropía sola salta de nada a colapso; hace falta otro mecanismo (top-k duro o diversidad entre hallazgos).
- **La poda sin reentrenar no paga:** quitar un slot cuesta 0.009–0.015 pseudo; 3 slots (61 % del costo) cuestan −0.026 (exp019).
  En la métrica de Kaggle, ahorrar todo el preprocesado v2 de 6 slots (~281 s, 4 vCPU ideal) vale ~0.004 AUC
  (cota pesimista medida en el pod: 997 s ≈ 0.014; `costo_inferencia.md`). Con runtime en minutos, manda la AUC.
- **Ni MobileNetV4-M con lr de CNN ni 2.5D sobre v1 sirven:** exp014 killed (gold −0.068), exp016 neutral.
- **Ensamble no paga en eficiencia:** 1 fold de 16 épocas (0.810) empata con 5 folds de 8 (0.812) con 1/5 del forward.

## 3. Implicaciones para el paper

**Se mantiene** (tesis de `docs/10_PLAN_PAPER.md`):
- Curva AUC–cómputo medida en hardware real, con desglose: ya hay costo DICOM/preprocesado por slot (`costo_inferencia.md`),
  Pareto de backbones (`analisis_eficiencia_paso1.md`) y la curva de 63 subconjuntos de slots (exp019).
- Entrada v2 (recorte físico + slots por tipo de secuencia) como base del método: es lo que más mueve la vara interna.

**Se reformula:**
- **Poda por hallazgo:** en su forma actual no funciona (exp018 no suma sobre v2; exp019 no tolera poda; exp017 colapsa). Para
  sostenerla hacen falta dos piezas nuevas: entrenamiento con **slot dropout** y una atención que **concentre sin colapsar**
  (entropía suave, exp022, o término de diversidad entre hallazgos). Si el 14-oct no hay evidencia, ISBI pasa al título 3
  (estudio Pareto) de `PLAN_ISBI.md`.
- **Etiquetas:** dejan de ser contribución "baja-media" solo para CMIG. El ruido de etiquetas y la alineación con las
  definiciones oficiales es una contribución **medible**: misma red, solo etiquetas → +0.024 LB, con la vara v1 apuntando al
  lado contrario. Candidato a entrar en ISBI como hallazgo de evaluación (validación interna desalineada con el test).
- **Lectura de eficiencia:** con la fórmula de Kaggle (0.01 AUC ≈ 697 s) y runtimes de minutos, el término de tiempo es pequeño;
  el argumento del paper debe ser costo clínico/hardware (s/estudio, GMACs), no solo el score de Kaggle.

**Figuras y tablas:**

| Pieza | Estado |
|---|---|
| Pareto backbones (GMACs, latencia) | existe: `docs/figs/paso1_pareto_gmacs.png`, `paso1_pareto_latencia.png` |
| Matriz atención 12 hallazgos × 6 slots (Fig. 7 CMIG / inset Fig. 1 ISBI) | datos en SPEC de exp018 (`slot_mass`); falta la figura |
| Curva costo–AUC por subconjunto de slots | datos en `metrics.json` de exp019 (`subsets`, 63); falta la figura y la versión con slot dropout |
| Desglose de latencia DICOM/preprocesado/red (Fig. 6 CMIG) | datos en `costo_inferencia.md` (rama exp013); falta la figura |
| Tabla de ablaciones (Tab. 4 CMIG) | filas exp012–exp021 en fold 0; faltan 5 folds y bootstrap |
| Etiquetas vs LB (pseudo v1, pseudo L2, gold, LB) | datos de exp021; falta tabla y más etiquetadores (L3/L4/L7/L8) |
| Tabla principal con OOF, gold con IC y LB privado | falta (no hay referencia nueva en 5 folds) |

## 4. Plan siguiente (priorizado)

1. **Referencia nueva** = entrada v2 + 16 épocas + etiquetas L2, 5 folds, con envío a Kaggle. Es la fila base de todas las tablas.
2. **Cambiar la vara interna:** pseudo contra L2 (o contra la etiqueta ganadora de exp011) para seleccionar; v1 solo informa.
   Re-puntuar los OOF existentes contra esa etiqueta.
3. **Slot dropout** sobre la referencia nueva, y repetir la curva de poda de exp019 (incluye poda aleatoria como control).
4. **Etiquetas L3/L4/L7/L8** como entrenamiento, decididas por LB y por el dev humano de exp010 (una variable cada una).
5. **Destilación a estudiante eficiente** desde la referencia nueva (no más ajuste de MobileNetV4 sobre la receta actual).
6. **Go/no-go ISBI 14-oct:** si la poda con slot dropout no da curva limpia (k = 2 retiene ≥ 95 % de la AUC), ISBI con el
   título 3 (Pareto + etiquetas) o directo a CMIG. Números congelados 20-oct.
