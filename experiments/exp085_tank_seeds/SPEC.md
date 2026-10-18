# exp085_tank_seeds

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp044_raptor_ft_nartaa (all-data: fine-tune 6 ép. de los pesos SWA de nartaa 384 con consenso exp031; gold SWA 0.9192; seed 7 = sub044_1, 1.04 GPU-h) |
| Fecha | 2026-10-18 |
| Rama | exp085_tank_seeds |
| Estado | spec |

Idea: "tanque" para la final A (r6flip-repro, LB 0.958): un Raptor 384 propio entrenado con el 100 % de train, incluidos
los 58 gold con su etiqueta gold dura, ensamble de varias semillas, metido en la mitad Raptor de r6flip-repro.

## Decisión de Rodrigo (18-oct, autorizada explícitamente) — excepciones a las reglas del repo

- **Regla 6 suspendida para este experimento:** los 58 estudios gold (fold = −1) **entran al entrenamiento**, y en ellos la
  etiqueta de entrenamiento es la **gold dura (0/1)** en lugar del consenso blando. Por tanto gold no veta ni selecciona nada
  (las AUC gold que imprime el notebook son sobre estudios de entrenamiento: solo sanidad de que el código corre).
- **Sin validación, sin pseudo, sin fold 0:** entrenamiento all-data (4407 estudios) directo. Decide **solo el LB público**.
- **Más de una variable contra el padre (regla 2):** el "paquete tanque" lo fija Rodrigo: (a) aug espacial de exp045,
  (b) gold dentro con etiqueta dura, (c) ensamble de semillas. No se pretende atribuir el efecto a una sola de ellas.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** un Raptor 384 "init nartaa + nuestras etiquetas" entrenado con todo train (gold incluido con etiqueta dura)
y la aug espacial de exp045, promediado sobre varias semillas, es un miembro Raptor más robusto que nartaa384 solo; meterlo
al 50 % dentro de la parte 384 de la mitad Raptor de r6flip-repro sube el LB público por encima de 0.958.

**Cambios contra el padre (paquete fijado por Rodrigo):**

| Parámetro o celda | Padre (exp044) | Este experimento |
|---|---|---|
| Aug espacial de entrenamiento (celdas 0, 3, 4) | ninguna (solo ganancia ×U(0.9,1.1)) | la de exp045 verbatim: espejo L/R anatómico p 0.5 + afín por estudio (rot ±10°, escala 0.9–1.1, tras ±6 %) |
| Estudios de entrenamiento (celda 2) | 4349 no-gold | 4407 = 4349 no-gold + 58 gold (`GOLD_IN_TRAIN = True`) |
| Etiqueta de los 58 gold (celda 2) | — (fuera) | gold dura 0/1 de `train.csv` (`GOLD_HARD = True`); resto = consenso exp031 |
| `SEED` | 42 | una corrida por semilla: 42, 7, 123, 2024 (objetivo 4; las que terminen antes del congelamiento) |
| Nombres de salida (celdas 1, 6) | `swa_alldata.pt`, `alldata/metrics.json` | `swa_alldata_s<SEED>.pt`, `alldata_s<SEED>/metrics.json` |

Todo lo demás idéntico a exp044: init SWA nartaa `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (sha 7e5315da…,
carga estricta), 6 épocas, BB_LR 1e-5 / HEAD_LR 1e-4, AdamW wd 0.02, OneCycle pct 0.15, bs 8 × K 12 (K_EVAL 24), bf16,
clip 3.0, recorte central 320, corpus d96/384 (`/workspace/data/cache/raptor96_384`, copiado a `/root/corpus96_384`),
SWA sobre las épocas fijas [3, 4, 5], pos_weight sobre las etiquetas de entrenamiento. Código = notebook de exp071 (que es el
de exp044 + la aug de exp045 verbatim) con el calendario de exp044.

**Efecto esperado:** LB +0.000 a +0.002 sobre r6flip-repro (0.958). Coste ≈ 1.1 GPU-h por semilla en RTX PRO 4500.

**Kernel `tank-seeds` (pregunta de envío, fijada antes de correr):** r6flip-repro (build local, dataset R6 `-w1`) con UNA
variable: en la mitad Raptor, la parte 384 pasa de `rank(nartaa384)` a `0.5·rank(nartaa384) + 0.5·rank(media de
probabilidades de nuestras semillas)`. 224 queda igual (A224 = 0.35); pesos por hallazgo Raptor/ours iguales (Syn 0.8,
LatOA 0.2, PFOA 0.6, Baker 0.35, MedOA y resto 0.5). Inferencia de las semillas = la de nartaa 0949 (mismo script, TTA
espejo), en un proceso aparte que corre **después** de todo lo existente; si falla o no cabe en el tiempo, por estudio se
cae a nartaa384 solo (la submission nunca depende de las semillas). Pesos fijados ahora; no se ajustan con el LB.

**Criterios (no cambian después de correr):**
- **Decide el LB público** del kernel `tank-seeds` (un envío): `worked` si LB ≥ **0.959**; `didnt` si LB < 0.959.
- Para que el kernel cuente: ≥ 2 semillas terminadas, validación COMPLETE sin Traceback y las semillas puntuaron (sin
  fallback) en la corrida de validación.
- Sanidad (no decide): pérdida de entrenamiento finita y decreciente; carga estricta del init; gold-58 impreso solo como
  sanidad (es entrenamiento).
- Congelamiento: una semilla que no termine de entrenar antes del **19-oct 12:00 UTC** se descarta.

**Presupuesto:** ≤ 1.5 GPU-h por semilla, ≤ 6 GPU-h en total; 2 pods RTX PRO 4500 ($0.72/h); tope $14.

**Riesgos:** sin validación no hay ninguna señal interna (el LB público tiene ruido ±0.004 con ~390 estudios); sobreajuste a
gold no medible; tiempo de inferencia del kernel (4 Raptor 384 extra sobre el test oculto: se mide en la validación y hay
corte por deadline con fallback); colisión de salidas entre pods (nombres por semilla).

**Licencia / legal:** pesos de nartaa (supervisión OAI/MOAKS) como init, igual que exp044/exp071. Dataset de pesos y kernel privados.

## 2. PLAN

- Notebook = exp071 con: celda 0 (parámetros: `EXP`, `PADRE`, `EPOCHS 6`, `BB_LR 1e-5`, `HEAD_LR 1e-4`,
  `SWA_EPOCHS [3, 4, 5]`, `GOLD_IN_TRAIN`, `GOLD_HARD`, `NOTES`), celda 1 (`TAG` por semilla), celda 2 (`train_lab` =
  consenso con gold duro; gold dentro del pool all-data), celda 4 (`train_lab` para Y y pos_weight; el smoke incluye 8 gold),
  celda 6 (ruta de metrics por semilla). Ninguna otra. Generado por `orq/tank_seeds/make_nb.py` (fuera del repo).
- Cadena autónoma por pod (`orq/tank_seeds/chain.sh`, fuera del repo): copia corpus → setup → smoke (solo pod A) →
  semillas en serie (`-p SEED n -p SMOKE False`) → export `kaggle_export/exp085_tank_seeds/swa_alldata_s<SEED>.pt` →
  commit local de `alldata_s<SEED>/metrics.json` → quita guard_skip → podStop. Pod A: 42, 123; pod B: 7, 2024.
- Kernel `tank-seeds` y dataset privado `rsna-knee-exp085-tank-seeds` cuando haya ≥ 2 semillas.

## 3. VERDICT (números de `results/exp085_tank_seeds/alldata_s*/metrics.json` y LB público de Kaggle)

pendiente LB
