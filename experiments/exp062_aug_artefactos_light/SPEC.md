# exp062_aug_artefactos_light

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | **por decidir con la regla fija de abajo** (ganador de la Fase 1 de preprocesado, o exp045_raptor_aug_espacial) |
| Fecha | 2026-10-13 |
| Rama | exp062_aug_artefactos_light (creada desde exp045 607644c; si el padre es otro brazo, el notebook se copia del padre al materializar) |
| Estado | spec |

Idea: Fase 2 de p7c — aug de artefactos de adquisición de RM en GPU, preset **light** (`AUG_STRENGTH = "light"`),
sobre el padre elegido por la Fase 1. Borrador original: `orq/p7c_augs/README.md` (SPEC borrador, cambios de celdas,
presets, harness exp040). Hermano: exp063_aug_artefactos_strong (preset strong), mismo padre, misma vara.

## 1. SPEC (antes de tocar el notebook)

**Regla de elección del padre (fijada el 2026-10-13 ~04:00 UTC, antes de cualquier resultado de la Fase 1; también en `orq/ESTADO.md`):**
padre = el brazo de la Fase 1 — P1 `exp058_prepro_clahe`, P2 `exp059_prepro_nyul` o `exp061_prepro_kneecrop` (este último solo
si su decisión final está disponible cuando lo estén las de P1 y P2; si no, no se considera) — que sea **promising** según
los criterios de su SPEC (media de 2 semillas: Δ pseudo ≥ +0.003 vs referencia = media(exp045 s42 0.8898899, sub045_5_seed7)
y gold ≥ ref_gold − 0.02) con la **mayor media de pseudo de 2 semillas**; si un brazo no tiene las 2 semillas no es elegible;
si ninguno es promising → padre = **exp045_raptor_aug_espacial** (corpus crudo). P3 (N4) no entra en la regla.

**Hipótesis:** simular artefactos de adquisición de RM por serie (bias de bobina, menor resolución in-plane, movimiento y
ghosting en k-space, ruido Rician, gamma; p baja por transformación, parámetros por (estudio, slot)) sube la pseudo L2
fold 0 frente al padre y mejora la robustez (harness exp040) sin degradar gold.

**Origen:** Pérez-García 2021 TorchIO (doi:10.1016/j.cmpb.2021.106236); Zhang 2020 BigAug (doi:10.1109/TMI.2020.2973595);
Shaw 2019 MIDL (PMLR 102:427–436); internos `docs/conocimiento/02_augmentations.md` §3, exp040, exp046.

**Única variable cambiada:**

| Parámetro o celda | Padre | Este experimento |
|---|---|---|
| `AUG_STRENGTH` (bloque `mri_artifact_aug` en train) | sin aug de artefactos (`"off"`) | `"light"` (presets de `orq/p7c_augs/mri_aug.py`) |

Preset light: bias 0.10·c[0.05,0.15]; aniso 0.10·f[1.25,2.0]; motion 0.05·E{1,2}·±1.5 px; ghost 0.05·n{4..10}·[0.10,0.30]; rician 0.10·σ[0.005,0.02]; gamma 0.10·γ[0.86,1.16] (P(unidad tocada) 41 %).
El plumbing del slot por ventana no cambia el cómputo del padre (con `"off"` la salida es idéntica). Todo lo demás igual al
padre: corpus (si el padre usa corpus preprocesado se reconstruye en el pod con el mismo constructor y las mismas
comprobaciones), split `split_v1_seed42`, 16 épocas, K=12, B=8, lr, aug espacial, SWA top-3, fold 0, `SELECT_ON = "pseudo"`.

**Efecto esperado:** Δ pseudo ≈ −0.002…+0.004 (prior baja in-distribution; el valor esperado está en robustez). Coste de
entrenamiento × ~1.01; coste de inferencia 0.

**Criterios (fijos antes de correr; no cambian después de ver resultados):**
- **Referencia** = media de pseudo L2 fold 0 (mejor época) de las 2 semillas (42 y 7) del padre; ref gold = media de sus gold.
- Corridas: semillas 42 y 7 de este brazo (fold 0). Δ = media(2 semillas) − referencia.
- **promising** si Δ ≥ **+0.003** y gold medio ≥ ref_gold − 0.02 · **killed** si gold medio < ref_gold − 0.02 · si no **didnt**.
  (+0.003 en vez de +0.005 solo porque es media de 2 semillas; ruido entre semillas ~0.0013–0.0015.)
- La decisión automática de la celda 5 (+0.005 vs padre, 1 semilla) es informativa; mandan estos criterios.
- **Secundaria obligatoria (no decide):** harness de robustez exp040 sobre `best_f0.pt` semilla 42 de este brazo y del padre,
  11 condiciones (`ref,mirror,gauss5,rician5,rician10,bias,gamma70,gamma140,res192,cortes50,drop_s0`), salida en
  `results/exp062_aug_artefactos_light/robustez_s42/` (RES en subcarpeta: nunca pisa el metrics.json del experimento); Δ por condición y
  subgrupo `lang` (croata) vs la carpeta del padre.
- Gold (58) solo veta.

**Presupuesto:** 2 semillas × ~2.6 GPU-h + harness ~0.3 GPU-h (+ reconstrucción de corpus si toca) ≈ 5.5–6 GPU-h (~$4–4.5).

**Riesgos:** ver `orq/p7c_augs/README.md` §Riesgos (prior baja, regularizador extra con 16 épocas, FFT circular con el
relleno del afín, aug sobre uint8 p2–98, tiempo GPU solo extrapolado: se mide en el smoke).

## 2. PLAN

- Al conocerse el padre, la cadena (`orq/p7c_chain/chain.sh`) copia el notebook del padre a
  `experiments/exp062_aug_artefactos_light/exp062_aug_artefactos_light.ipynb` y aplica `orq/p7c_chain/patch_p2_nb.py` (probado en los 4 padres posibles): celdas
  **0** (`EXP`, `PADRE`, `AUG_STRENGTH`, `NOTES`), **3** (pega `mri_aug.py`; `StudyWindows` devuelve el slot por ventana;
  `to_input` aplica `mri_artifact_aug` tras `spatial_aug`, solo en train), **4** (desempaquetado de 5 elementos) y
  **6** (`mri_aug_preset` en metrics.json). Ninguna otra. Commit del notebook en esta rama antes de correr.
- Smoke (`SMOKE = True`) y fold 0 semilla 42 en `wt_exp062_aug_artefactos_light`; semilla 7 en `wt_exp062_aug_artefactos_light_seed7` → `metrics_seed7.json`.
- Harness exp040 (scripts copiados a esta carpeta como auxiliares, CKPT/OUT/RES/CORPUS/CACHE fijados por sed).

## Etapa 3 — confirmación 5 folds (criterio fijado el 2026-10-13 ~04:00 UTC, antes de ver resultados; aprobado por Rodrigo)

Solo aplica si este experimento resulta el **ganador final** de la cadena p7c (regla en `orq/ESTADO.md`, sección "Fase 2 + Etapa 3"):
mejor brazo `promising` de la Fase 2 (media de 2 semillas); si ninguno, el brazo `promising` de la Fase 1; si nada es
`promising`, la Etapa 3 no corre.
- Se entrenan los folds 1–4 con semilla 42 (el fold 0 s42 ya existe) → OOF pseudo de 5 folds concatenados y gold con el
  ensamble (media de probabilidades) de los 5 modelos `best_f{k}.pt`.
- Preprocesado reajustado por fold donde aprende algo: Nyúl → landmarks ajustados solo con el train de cada fold
  (`nyul_landmarks_f{k}.json`, un corpus por fold); recorte de rodilla y CLAHE no ajustan nada (un solo corpus).
- Referencia: exp045 5 folds — OOF pseudo **0.8844**, gold del ensamble de 5 **0.9244**.
- **worked (confirmado)** si Δ OOF pseudo ≥ **+0.003** vs 0.8844 **y** gold del ensamble ≥ 0.9244 − 0.02; si gold cae más
  de 0.02 → **killed**; si no → **didnt** (se queda en fold 0).
- Si se confirma: dataset privado con los 5 pesos + kernel solo privado + kernel ensamble blend-mine w0.15, con el
  preprocesado inyectado igual que en entrenamiento; se corren hasta COMPLETE y se valida `submission.csv`. No se envía a
  Kaggle desde la cadena: los slugs quedan en ESTADO para las colas del 15/16-oct.

## 3. VERDICT (números de `results/exp062_aug_artefactos_light/metrics.json` y `metrics_seed7.json`)

Pendiente.

**Padre elegido por la regla (cadena p7c, 2026-10-13T07:47:25Z):** `exp045_raptor_aug_espacial` — referencia 2 semillas pseudo 0.88803 / gold 0.922301 (`orq/p7c_chain/state/phase1.json`).
