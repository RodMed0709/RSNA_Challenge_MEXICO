# exp032_raptor_alineado_LR

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp029_raptor_entreno |
| Fecha | 2026-10-11 |
| Rama | exp032_raptor_alineado_LR |
| Estado | didnt |

Idea: entrenar y evaluar la receta Raptor d96 de exp029 con todas las rodillas alineadas a la misma lateralidad
(izquierdas espejadas a derechas), como hacía nuestro cache v2 (exp009). Raptor no alinea; solo aplica TTA espejo en inferencia.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si las rodillas izquierdas se espejan a derechas al cargar el corpus (entrenamiento, validación y gold),
el modelo no tiene que aprender dos versiones especulares de la anatomía medial/lateral y la **pseudo AUC L2 de fold 0
sube ≥ +0.005** contra exp029, sobre todo en las etiquetas con lado (menisco y artrosis mediales/laterales, MCL).

**Única variable cambiada:**

| Parámetro o celda | Padre (exp029) | Este experimento |
|---|---|---|
| Orientación L/R al cargar del corpus (celda 0 `ALIGN_LR`, celdas 2–3) | nativa (como sale de `build_corpus96.py`) | estudios con lado `L` espejados por slot a "derecha"; `R` y `unknown` nativos |

Todo lo demás es idéntico: corpus `raptor96_384` (no se reconstruye), etiquetas blandas de Raptor, CoAtNet-2 384 +
MIL por hallazgo, 16 épocas, semilla 42, recorte central 320, selección por pseudo L2, SWA top-3, sin flips de
aumento. **La evaluación del padre no usa TTA** (la función `predict` de exp029 hace una sola pasada sin espejo), y aquí
tampoco: la única diferencia entre los dos experimentos es la alineación.

### Investigación previa (hecha antes de entrenar)

**Lado por estudio.** Columna `study_side` de `/workspace/data/cache/meta_v2/study_meta_v2.csv` (exp009,
`01_meta_v2.ipynb`, celda 5): cascada tag DICOM `Laterality`/`ImageLaterality` → texto L/R de la descripción de la serie →
geometría (centro físico x del FOV en LPS a más de ±15 mm del isocentro: x > 15 → L, x < −15 → R) → voto del estudio →
`unknown`. Validada en exp009 contra el lado que dice el informe: **97.04 % de precisión (n = 1,890), 98.62 % de
cobertura**. En los 4,407 estudios del corpus: **R 2,305 (52.3 %) · L 2,041 (46.3 %) · unknown 61 (1.4 %)**; fuente
tag 2,202 · geometría 1,768 · texto 376. Fold 0: L 411 · R 453 · unknown 7; gold: L 23 · R 33 · unknown 2.

**Orientación del corpus de exp029.** `build_corpus96.py` (= `build_study` de nartaa) guarda los píxeles **en
orientación nativa** (sin transponer ni voltear según IOP) y ordena los cortes ascendente por IPP·(r×c). Se leyó la IOP
de las 20,792 series que usa el corpus (`lateralidad_corpus.py`, reproduce `_pick_series_for_slot`): el eje x del
paciente cae **siempre** en el mismo eje y con el **mismo sentido** en el 100 % de las series de cada slot (sagitales:
orden de cortes, el índice crece hacia −x = derecha del paciente; coronales y axiales: columnas, el índice crece hacia
+x = izquierda del paciente; |cos| mínimo 0.76). Por tanto todas las rodillas derechas ya están en la misma orientación
y la alineación se reduce exactamente a **espejar las izquierdas**: sagital → invertir el orden de cortes del slot;
coronal y axial → voltear columnas (igual que el cache v2). Ningún estudio derecho ni `unknown` se toca.

**Implementación.** Tabla `align_lr_exp032.{npy,csv,json}` en `/workspace/data/cache/raptor96_384/` (y copia en
`/root/corpus96_384/`), `(N, 5)` int8 alineada con `ids.npy`: 0 nativo · 1 invertir cortes del slot · 3 voltear columnas
(2 = voltear filas, no aparece). En el notebook, `StudyWindows` permuta la máscara y los índices de corte por slot y
voltea columnas por corte antes de formar las ventanas 2.5D (el recorte central 320 es simétrico y conmuta con el
volteo). Espejos por slot: Sag fluid 2,041 · Sag 1,993 · Cor fluid 2,041 · Cor 1,511 · Ax 2,041 (slots ausentes no cuentan);
2,041 estudios con algún espejo = exactamente los `L`.

**Verificación del espejo** (`verifica_espejo.py`): plantilla = media de 300 derechas; correlación de Pearson del bloque
del slot (recorte 320 → 40×40 por corte) con la plantilla en 20 izquierdas, nativo → alineado (control: 20 derechas):
Sag fluid 0.537 → 0.561 (alineado > nativo en 18/20; control R 0.575) · Cor fluid 0.613 → 0.651 (19/20; R 0.655) ·
Ax 0.579 → 0.608 (19/20; R 0.624). Tras alinear, las izquierdas se parecen a la plantilla tanto como las derechas.
Figura revisada a mano (corte coronal, axial y sagital lateral, nativo | alineado). Los 61 `unknown` quedan nativos
(0 con espejo).

**Nota sobre el sagital.** El modelo ve una bolsa de ventanas sin posición (atención MIL), así que invertir el orden de
cortes sagitales solo cambia el orden de los canales dentro de cada ventana 2.5D (c−1, c, c+1 → c+1, c, c−1); el efecto
fuerte de la alineación viene de coronal y axial. Es parte de la misma variable (alinear la anatomía), no se separa.

**TTA espejo en Kaggle.** Si exp032 gana, la TTA espejo anatómica de Raptor no tiene sentido tal cual (espejar una
rodilla ya alineada produce una "izquierda", fuera de la distribución de entrenamiento); la inferencia debe alinear igual
que aquí y, si se quiere TTA, usar otra (p. ej. ganancia o desplazamiento de ventanas). Se decide en el experimento de envío.

**Lateralidad en la inferencia de Kaggle.** Todo sale de cabeceras DICOM, disponibles en el test: tag
`Laterality`/`ImageLaterality`, `SeriesDescription`/`ProtocolName` (texto L/R) y centro físico x del FOV (IPP + IOP ·
spacing · (filas, columnas)/2). Hay que portar la cascada de `01_meta_v2.ipynb` (celda 5) al kernel y aplicar el espejo
por slot con la misma regla por IOP de `lateralidad_corpus.py` (no suponer la orientación: calcular el eje x y su sentido
desde la IOP de cada serie, como aquí; en train fue constante al 100 %); en los 3
estudios de test públicos el lado sale (L por tag, R y R por geometría). Lado desconocido → nativo, como aquí.

**Criterios (no cambian después de correr)** — métrica: pseudo AUC macro contra L2 binarizada (> 0.5) en val fold 0,
**mejor época** (el `pseudo_auc` de `per_fold["0"]`, como en el padre); gold-58 solo veta:
- `worked`: pseudo L2 f0 ≥ exp029 + 0.005 **y** gold no cae más de 0.02 contra exp029.
- `killed`: pseudo L2 f0 ≤ exp029 − 0.005 **o** gold cae más de 0.02.
- `didnt`: cualquier otro caso (|Δ| < 0.005 sin veto).
- Se reportan también SWA top-3, v1 y AUC por etiqueta (pares medial/lateral) como información.

**Presupuesto:** fold 0 ≈ 2.5 GPU-h (igual que el padre: 434 pasos/época × 16 épocas, ~1.15 s/paso) en rsna-d2 tras
terminar exp029; smoke ≤ 10 min con batch reducido mientras exp029 corre. Tabla de espejo: 1 min de CPU (hecha).

**Riesgos:** ~3 % de lados erróneos (precisión 97 %) quedan espejados al revés; 61 `unknown` nativos mezclan
orientación; la ganancia puede ser pequeña porque CoAtNet ya ve ambas orientaciones con ~50/50 de L/R; varianza de
semilla entre runs (no medida para esta receta) del orden del umbral.

## 2. PLAN

- `lateralidad_corpus.py` (auxiliar, hecho): tabla de espejo desde IOP + `study_side`.
- `verifica_espejo.py` (auxiliar, hecho): correlación con la plantilla de derechas y figura.
- Notebook `exp032_raptor_alineado_LR.ipynb` = copia de exp029 con cambios solo en: celda 0 (`EXP`, `PADRE`, `ALIGN_LR`,
  `ALIGN_CODES`, `NOTES`), celda 2 (carga y comprobación de la tabla, `align_index`), celda 3 (`StudyWindows.__getitem__`
  aplica la alineación), celda 5 (texto de la decisión) y celda 6 (`align_lr` en `metrics.json`).
- Smoke: `SMOKE=True`, `BATCH_SIZE=2`, `N_WORKERS=4`, corpus completo `/root/corpus96_384` (cabe en la GPU junto a exp029).
- Run real: `FOLDS=[0]`, `SMOKE=False`, rsna-d2, cuando muera el PID de exp029; symlink `results/exp029_raptor_entreno`
  en el clon del pod para que `leer_metricas(PADRE)` calcule los Δ.

## 3. VERDICT (números de `results/exp032_raptor_alineado_LR/metrics.json`)

**Resultado:** didnt (Δ pseudo L2 +0.0022 < +0.005; gold no cae) · **Fase alcanzada:** screen fold 0 (rsna-d2, 20:12–22:45 UTC)

| Métrica | exp029 (padre) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8682 (ep15) | 0.8704 (ep13) | +0.0022 |
| pseudo L2 fold 0 (SWA top-3) | 0.8679 [15,14,13] | 0.8703 [13,15,14] | +0.0024 |
| v1 fold 0 (mejor época) | 0.8329 | 0.8299 | −0.0030 |
| gold (veto, mejor época) | 0.9115 | 0.9143 | +0.0028 |
| gold (SWA) | 0.9111 | 0.9154 | +0.0043 |
| GPU-h | 2.51 | 2.50 | −0.01 |

Decisión automática de la celda 5: `neutral` → `didnt` según los criterios de la sección 1.

pseudo L2 por etiqueta (mejor época, este − padre): MCL **+0.040** (0.8164 vs 0.7764) · Contusion +0.008 ·
Synovitis +0.005 · Lateral Meniscus +0.002 · Medial Meniscus +0.002 · Lateral OA +0.002 · Effusion +0.001 ·
Baker's −0.002 · Medial OA −0.004 · Fracture −0.004 · PF OA −0.006 · ACL **−0.017**.

**Qué aprendimos:** alinear L/R no cambia la AUC macro de la receta Raptor por encima del umbral (+0.002 pseudo,
+0.003 gold, −0.003 v1): CoAtNet con atención MIL ya aprende ambas orientaciones con ~50/50 de L/R. La ganancia
grande y aislada es MCL (+0.04, estructura claramente medial), compensada por ACL (−0.017, estructura central); el resto
es ruido de semilla. El costo es nulo (mismo tiempo de entrenamiento, tabla de espejo de 1 min).

**Siguiente paso sugerido:** no promover la alineación como base. Como fuente de diversidad, el modelo alineado puede
entrar en un ensamble con exp029/exp030 (con su propia inferencia alineada, sin TTA espejo). Si se quiere confirmar
el efecto en MCL, hacerlo con 5 folds solo si se usa el modelo en el envío.
