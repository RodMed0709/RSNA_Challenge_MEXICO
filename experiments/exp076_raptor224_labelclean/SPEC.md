# exp076_raptor224_labelclean

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp074_raptor224_distill |
| Fecha | 2026-10-17 |
| Rama | exp076_raptor224_labelclean |
| Estado | didnt |

Idea: exp074 + limpieza OOF exp045: celdas de consenso contradichas -> solo teacher

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** parte de las etiquetas de consenso (LLM sobre el informe) contradicen lo que se ve en la imagen. Si en las
celdas (estudio, hallazgo) donde el OOF de exp045 (Raptor 384, 5 folds, cada estudio predicho por el modelo que no lo vio)
contradice fuertemente al consenso quitamos la mitad "consenso" del objetivo y dejamos solo al teacher, el alumno 224
aprende menos ruido y la pseudo AUC fold 0 sube. Técnica del 2.º lugar RSNA Lumbar 2024 (brendanartley,
"confident learning": OOF ensemble + quitar los puntos cuya predicción difiere mucho de la etiqueta; su README no da umbral
ni fracción).

**Única variable cambiada:** el objetivo de entrenamiento en las celdas marcadas como ruidosas.

| Parámetro o celda | Padre (exp074) | Este experimento |
|---|---|---|
| Objetivo en celdas marcadas (celda 2) | 0.5 × teacher + 0.5 × consenso | teacher (se enmascara el consenso) |
| Objetivo en el resto | 0.5 × teacher + 0.5 × consenso | igual |

**Regla exacta (fijada antes de entrenar, elegida solo con estudios de folds 1–4; nunca fold 0 ni gold):**
1. OOF = `oof_swa_f{k}.csv` de exp045 (worktrees `wt_exp045_raptor_aug_espacial`, `wt_exp045_f1..f4`), 4348 estudios del
   pool, cada uno con la predicción del fold que lo tuvo en validación (se comprueba que el fold coincide con `split_v1`).
2. Calibración por hallazgo (el OOF crudo está inflado por `pos_weight`: media 0.39–0.52 contra 0.15–0.50 del consenso):
   mapeo por cuantiles OOF → distribución del consenso, ajustado **solo con los estudios de folds 1–4**
   (`oof_cal(x) = quantile(consenso_f1-4, ECDF_oof_f1-4(x))`) y aplicado igual a todos los estudios.
3. Celda marcada si `|oof_cal − consenso| > LC_TAU = 0.7`, el consenso existe y el teacher puntuó el estudio.
   En celdas marcadas el objetivo = teacher; si el teacher no puntuó el estudio (11 estudios) la celda no se toca.
4. Elección de `LC_TAU` (análisis `orq/labelclean/analysis_f1to4.json`, solo folds 1–4, rejilla 0.5/0.6/0.7/0.8 calibrada):
   regla previa = el umbral más bajo con **fracción tocada ≤ 5 %** y donde el teacher (modelo de imagen independiente del
   OOF) da la razón al OOF en ≥ 50 % de las celdas marcadas. 0.5 → 11.1 % (fuera), 0.6 → 7.0 % (fuera),
   **0.7 → 3.43 % de las celdas (1431 celdas, 1111 estudios, teacher con el OOF 51.6 %)**, 0.8 → 1.6 % (teacher 48 %).
5. Enmascarar el consenso (no suavizar hacia el OOF): en las celdas marcadas el teacher queda ~50/50 entre OOF y consenso,
   así que no se mete la opinión de exp045 (más débil que el teacher) en el objetivo; se deja al modelo de imagen más fuerte.
   Es lo más parecido al "remove" del 2.º lugar sin perder la celda.
6. Guarda en el notebook: la fracción marcada en folds 1–4 tiene que caer en [3 %, 4 %] o se aborta.

Todo lo demás igual a exp074: split `split_v1_seed42`, semilla 42, Raptor 224, 3 slots, 16 épocas, SWA top-3 por pseudo,
aug espacial, teacher.csv de exp074. Entrena en folds 1–4 y evalúa en fold 0 (pseudo L2 y gold de siempre). En fold 0 la
limpieza no cambia nada de la evaluación (son estudios de validación); en all-data los estudios de fold 0 se limpian con el
OOF de exp045 fold 0 (fuera de fold).

**Efecto esperado:** Δ pseudo fold 0 ≈ +0.002 a +0.006; mismo costo de entrenamiento e inferencia.

**Criterios (fijados por Rodrigo el 17-oct, no cambian después de correr):**
- `worked` si pseudo SWA fold 0 ≥ 0.9037 (exp074 0.8987 + 0.005).
- `killed` si gold SWA < 0.8997 (exp074 0.9197 − 0.02), diga lo que diga pseudo.
- Si no: `didnt`. Gold nunca elige época, checkpoint, umbral ni pesos.
- Si `worked`: kernel = `exp074-student224` (medellinrodrigo) con solo el checkpoint cambiado por el all-data de exp076;
  pregunta del envío "¿la limpieza mejora el alumno eficiente?", worked si LB ≥ 0.950 (exp074 all-data 0.948).

**Presupuesto:** fold 0 ≈ 0.9 GPU-h + all-data especulativo en paralelo ≈ 1 GPU-h · 2× RTX 4090 · ≤ $7.

**Riesgos:**
- Fuga leve: los modelos exp045 de folds 1–4 entrenaron con fold 0, así que sus OOF de folds 1–4 llevan algo de información
  de las etiquetas de fold 0; y el teacher (exp074) vio fold 0. La pseudo fold 0 puede salir algo inflada; gold y LB verifican.
- v1 (también LLM sobre informe) da la razón al consenso en ~80 % de las celdas marcadas: buena parte de los desacuerdos
  pueden ser errores del modelo y no del informe. Por eso no se reemplaza por el OOF, solo se quita el consenso.
- Las celdas marcadas se concentran en PF OA (367), Medial/Lateral OA (194/173) y Lateral Meniscus (133).

## 2. PLAN

- Celda 0: `OOF045 = [...]` (5 rutas), `LC_TAU = 0.7`, `LC_FRAC_RANGE = [0.03, 0.04]`, `NOTES`.
- Celda 2: tras armar el objetivo de exp074, carga OOF exp045, calibra por cuantiles con folds 1–4, marca celdas
  `|oof_cal − consenso| > LC_TAU` (con teacher puntuado) y ahí pone objetivo = teacher; imprime fracción por fold y por
  hallazgo. Ninguna otra celda cambia.
- Cadena en pod 2× 4090 (`orq/labelclean/chain.sh`): smoke → fold 0 (GPU0) ∥ all-data especulativo (GPU1, SWA épocas
  [15, 14, 13] como exp074) → verdict con los criterios de arriba → commit local metrics.json → export si worked →
  quita guard_skip → podStop.

## 3. VERDICT (números de `results/exp076_raptor224_labelclean/metrics.json`)

**Resultado:** didnt · **Fase alcanzada:** screen (fold 0)

| Métrica | Padre (exp074) | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 (SWA [14, 15, 13]) | 0.8987 | 0.8985 | −0.0002 (criterio ≥ 0.9037) |
| pseudo AUC fold 0 (mejor época 14) | 0.8988 | 0.8987 | −0.0001 |
| gold AUC (SWA, solo veto) | 0.9197 | 0.9226 | +0.0029 (veto < 0.8997: no) |
| v1 AUC fold 0 (SWA) | 0.8540 | 0.8508 | −0.0032 |
| OOF pseudo 5 folds | — | — | no se corre |
| GPU-h (fold 0) | 0.866 | 0.878 | +1 % |

Limpieza aplicada: 1760 celdas en 1372 estudios (3.42 % de las celdas de folds 1–4; PF OA 451, Medial OA 241, Lateral OA 210,
Lateral Meniscus 160). All-data especulativo cancelado por la cadena (no worked): no hay kernel ni CANDIDATE.

**Qué aprendimos (2-4 líneas):** con el objetivo ya destilado (0.5 teacher), quitar el consenso en el 3.4 % de celdas donde
el OOF de exp045 lo contradice no mueve la pseudo fold 0 (−0.0002, por hallazgo todo dentro de ±0.005). El teacher ya
amortigua ese ruido; en esas celdas el teacher estaba ~50/50 entre OOF y consenso, así que la "limpieza" cambia poco el
objetivo (media 0.550 → 0.589). El +0.003 de gold es ruido de 58 estudios y no decide nada.

**Siguiente paso sugerido:** no seguir por esta vía con el alumno destilado. Si se reintenta, que sea sobre un objetivo sin
teacher (exp057/exp045, consenso puro) o con OOF de un ensamble más fuerte, como experimento nuevo.
