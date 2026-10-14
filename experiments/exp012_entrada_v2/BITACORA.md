# Bitácora — exp012_entrada_v2

Padre: `exp002_dinov2s_ref` · Rama: `exp012_entrada_v2` · Fase SDD actual: **IMPLEMENT terminado, falta RUN (smoke)**.

## 1. Qué es este experimento

Repetimos exactamente la receta de exp002 (mismo modelo, mismas épocas, mismo split), pero cambiando **lo que el modelo ve**.
Antes veía la imagen completa a 224 px y hasta 5 series elegidas con una regla simple. Ahora ve el cache v2 de exp009:
un recorte centrado en la rodilla (0.5 mm por píxel), rodillas izquierdas volteadas para que todas queden igual, y 6
"slots" fijos, uno por tipo de secuencia. La idea: meniscos y ligamentos ocupan más píxeles y cada slot trae siempre la
serie que le corresponde, así que la pseudo AUC debería subir.

## 2. Registro cronológico

### 2026-10-08 — SPEC escrita
- **Qué:** se llenaron las secciones 1 (SPEC) y 2 (PLAN) de `SPEC.md`.
- **Por qué:** regla SDD: sin SPEC confirmada no se toca el notebook. Fija la única variable (la entrada), los criterios
  de éxito/kill y el presupuesto antes de ver resultados.
- **Evidencia:** commit `4403713` ("Add exp012 spec").

### 2026-10-08 — Notebook adaptado al cache v2
- **Qué:** copia del notebook de exp002 con cambios solo en las celdas del PLAN:
  - Celda 0 (parámetros): `CACHE = "v2_05mm"`, `IMG_SIZE = 336`, `N_SLOTS = 6`, `BATCH_SIZE = 1`, `GRAD_ACCUM = 8`,
    `STUDY_META_CSV = /workspace/data/cache/meta_v2/study_meta_v2.csv`.
  - Celda 2 (datos): lector nuevo `KneeV2Dataset`. Lee volúmenes `[cortes, 320, 320]` uint8 del cache, toma los slots de
    `study_meta_v2.csv`, muestrea 16 cortes por serie (igual que exp002) y espeja las rodillas izquierdas al leer.
    Revisa cuántos estudios no tienen ningún slot y se detiene si son ≥ 0.5 %.
  - Celda 3 (modelo): `KneeV2Model` con embedding por id de slot y máscara (`key_padding_mask`) en la atención entre series.
  - Celdas 1 y 4–7: sin cambios.
- **Por qué:** es lo mínimo para que la receta de exp002 lea la entrada nueva (una sola variable).
- **Evidencia:** commit `3e789f5` ("Adapt exp012 notebook to cache v2 input"), 1 archivo, +196 / −12 líneas.
  Verificación del worker (en local, sin GPU ni datos reales):
  - prueba sintética PASS: el dataset devuelve `(6, 4, 3, 28, 28)`, el forward devuelve `(1, 12)`, y cambiar el contenido
    de los slots enmascarados no cambia la salida (diferencia 0.0);
  - JSON del notebook válido, 7/7 celdas de código compilan;
  - `pytest`: 8 passed, 1 skipped.
  - **No se ha entrenado nada todavía.**

### 2026-10-08 — Revisión del orquestador
- **Qué:** se revisó el diff contra la copia de exp002.
- **Por qué:** comprobar que el espejo no se aplica dos veces (si el cache ya viniera espejado, voltear al leer lo
  desharía).
- **Evidencia:** `experiments/exp009_preparacion_v2/SPEC.md` dice que el cache v2 está centrado y que "el espejado con
  `study_side` se hace al cargar (sin espejar los `unknown`)". El notebook espeja solo si `study_side == "L"`, así que
  es correcto.

## 3. Cómo correrlo (en el pod)

1. Abre el pod (volumen `RSNA-KNEE`, template `RSNA-KNEE`) y en la terminal de Jupyter:
   ```bash
   cd /workspace/users/<tu_nombre>
   bash setup_pod.sh
   git fetch && git switch exp012_entrada_v2 && git pull
   ```
2. Comprueba que existen el cache y los metadatos de exp009:
   `/workspace/data/cache/v2_05mm/` y `/workspace/data/cache/meta_v2/study_meta_v2.csv`.
3. Abre `experiments/exp012_entrada_v2/exp012_entrada_v2.ipynb` con el kernel **Python (rsna)**.
4. **Smoke:** deja `SMOKE = True` en la celda 0 y corre todo. Hace 1 época con pocos estudios. Fíjate en:
   - la línea `Estudios sin slots: ...` (si es ≥ 0.5 %, el notebook se para: avisa, no lo "arregles");
   - que no haya error de memoria (CUDA OOM) a 336 px.
5. **Screen fold 0:** en la celda 0 pon `SMOKE = False` (deja `FOLDS = [0]`), guarda y lánzalo en segundo plano:
   ```bash
   bash run_notebook.sh experiments/exp012_entrada_v2/exp012_entrada_v2.ipynb
   ```
   El script imprime el PID y el comando para ver el avance (`tail -f results/exp012_entrada_v2/run_<fecha>.log`).
6. Si el fold 0 promete (Δ pseudo ≥ +0.005), los 5 folds:
   ```bash
   bash run_notebook.sh experiments/exp012_entrada_v2/exp012_entrada_v2.ipynb -p FOLDS '[0,1,2,3,4]'
   ```
7. Dónde queda todo: `results/exp012_entrada_v2/`
   - pesos: `best_f<fold>.pt`; predicciones: `oof_f<fold>.csv`, `gold_f<fold>.csv`; curvas: `history_f<fold>.csv`;
   - notebook ejecutado y log: `executed_<fecha>.ipynb`, `run_<fecha>.log`;
   - `metrics.json`: lo único que se sube a git.
8. Apaga el pod al terminar.

## 4. Decisiones de diseño y por qué

- **Máscara de slots faltantes.** Muchos estudios no tienen todos los slots (cobertura: slot 1 94 %, slot 4 57 %,
  slot 6 47 %). Un slot vacío se marca en `key_padding_mask` y la atención lo ignora. No se rellena con ceros, porque
  el modelo vería esos ceros como si fueran una imagen negra real. La prueba sintética confirmó que los slots
  enmascarados no influyen en la salida.
- **Embedding por id de slot.** Sustituye al embedding de plano + secuencia de exp002: en v2 el número de slot ya dice
  qué plano y qué secuencia es.
- **336 px.** DINOv2 corta la imagen en parches de 14 px; 336 = 24 × 14, así no se pierde ni se rellena borde. El cache
  guarda 320 × 320 y se redimensiona a 336.
- **Batch 1 × acumulación 8.** A 336 px cada imagen cuesta ~2.3× más memoria que a 224. Con batch 1 y acumulando
  gradiente 8 pasos, el batch efectivo es 8, igual que en exp002 (que usa batch 2). La SPEC dice que
  esto no cuenta como segunda variable.
- **Espejo al leer.** El cache v2 está guardado sin espejar. Al cargar, si `study_side == "L"`: en los slots sagitales
  (1, 4, 6) se invierte el orden de los cortes; en coronal y axial (2, 3, 5) se invierten las columnas. Los `unknown`
  no se espejan. Es lo mismo que definió exp009.

## 5. Pendientes y riesgos

- **Smoke en RunPod:** no se ha corrido. Es lo siguiente.
- **Memoria:** 336 px con 6 slots × 16 cortes puede no caber; si pasa, bajar batch / subir acumulación (ya está en
  batch 1, así que la siguiente opción sería gradient checkpointing; anotarlo aquí si se usa).
- **Tiempo:** el estimado del worker es ~30 min/época contra ~11 min/época de exp002 (≈2.7×), más que el ×2.3 de la
  SPEC. Vigilar el presupuesto: screen ~4 GPU-h, kill si GPU-h > 1.5 × presupuesto.
- **Estudios sin slots:** si son ≥ 0.5 %, el notebook se detiene; hay que avisar, no cambiar el umbral.
- **Inferencia en Kaggle:** hay que replicar el preprocesado v2 (recorte, 0.5 mm, slots, espejo) en
  `inferencia_kaggle.ipynb`. No hace falta para el screen, sí para enviar.
- **Detalle menor:** la celda markdown de título del notebook todavía dice "exp002 — Referencia DINOv2-small"; no afecta
  al entrenamiento.

## 6. Resultados

Números **solo** desde `results/exp012_entrada_v2/metrics.json`. No se copian de logs ni de memoria.

| Métrica | Padre (exp002) | exp012 | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | | | |
| gold AUC fold 0 | | | |
| GPU-h (fold 0) | | | |
