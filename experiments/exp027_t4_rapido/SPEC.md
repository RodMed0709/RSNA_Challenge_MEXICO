# exp027_t4_rapido

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp026_t4_variantes (V2: Raptor/0.949 sin TTA + channels_last, 13.7 min estimados @1300 en T4 ×2) |
| Fecha | 2026-10-11 |
| Rama | exp027_t4_rapido |
| Estado | didnt |

Idea: con el mismo marco de medición de exp026 (kernel privado en Kaggle, 2 × T4, 150 estudios de train =
58 gold + 92 fold 0, mismo pipeline Pool de preparación + un hilo por GPU), bajar el runtime de V2 con
cambios de cómputo en la T4: sin autoajuste de cudnn, el modelo de 224 px que publicó el mismo autor,
fp16 puro y `torch.compile`. Alimenta el envío al premio de eficiencia
(`AUC/(0.5−maxAUC) + RuntimeSeconds/32400`: 1 min ≈ 0.00086 de AUC).

## 1. SPEC

**Hipótesis (H):** modelo 224 px + `model.half()` + sin autoajuste de cudnn baja el runtime extrapolado a
1 300 estudios ≥ 40 % contra V2 (re-medido en la misma corrida) con ΔAUC gold ≥ −0.005.

**Variantes (mismos estudios; todas sin TTA, 94 ventanas, una pasada):**

| Var | Qué es | Cambio contra |
|---|---|---|
| R | V2 de exp026: modelo 384 (`…_r384_swa.pt`), recorte central 320, `channels_last`, autocast fp16, trozos de 16, `cudnn.benchmark = True` | referencia (re-medida) |
| A | R con `cudnn.benchmark = False` (el autoajuste es por hilo: hacerlo antes de lanzar los hilos no sirve). El modelo ya se carga una vez por GPU en el notebook (no hay nada que cambiar ahí) | R |
| B | A con el modelo de eficiencia del autor: `…_c08_r224_swa.pt` (`coatnet_rmlp_2_rw_224`), ventanas como en `nartaa/rsna-knee-0945-efficient-224crop`: recorte central 0.8 del corte de 384 (307 px) → bilinear a 224 → ImageNet; trozos de 16 | A |
| B94 | B con las 94 ventanas en un solo lote (como el notebook de 224) | B |
| C | A con `model.half()` completo + ventanas en half (sin autocast) | A |
| D | A + `torch.compile(backbone, mode="max-autotune-no-cudagraphs")`; compilación y autoajuste en el hilo principal antes de lanzar los consumidores; tiempo de compilación aparte y sumado a los costes fijos | A |
| E1 | B + C (224 + half) | combinación |
| E2 | B + C + D (224 + half + compile) | combinación |
| E2d | B + C + `torch.compile` en modo `default` (compilación barata; añadida tras el smoke, antes de correr) | combinación |
| F | si la imagen trae `torch_tensorrt` o `onnxruntime` con CUDA/TensorRT EP: backbone 224 half por ese runtime. Si no existe, se reporta y no se instala | B+C |

`reduce-overhead` (CUDA graphs) se descarta de antemano: los grafos de CUDA se graban por hilo y el
notebook consume desde hilos; se usa `max-autotune-no-cudagraphs`. Si D/E2 fallan al compilar, se
reportan como fallidos (el kernel sigue).

**Qué se mide (por variante, pipeline del notebook desde disco con page cache caliente, 150 estudios):**
reloj, s/estudio, "steady" (reloj sin el exceso del primer estudio de cada hilo), autoajuste, espera de
cada GPU, tiempo de alimentación (preparación), y lo mismo desde RAM (techo si la preparación no limitara).
Costes fijos medidos en el kernel: imports, carga del modelo por variante, compilación.
R corre primero en frío (primera lectura de disco) y otra vez en caliente; la mejor combinación corre al
final en frío sobre 150 estudios de fold 0 nuevos (rol `cold`, no se usan para AUC) para ver si la
preparación en CPU pasa a mandar.

**Minutos @1300** = steady × 1 300 / 60 + (imports + carga de modelo + autoajuste + compilación) / 60.
**Score de eficiencia relativo** = ΔAUC gold − 0.00086 · Δmin (contra R; también con ΔAUC pseudo-150).

**Métricas de predicción contra R:** Pearson de las probabilidades (150 × 12), media y máx. de |Δp|; AUC
macro gold (58) y pseudo AUC vs L2 > 0.5 (150 y 92 de fold 0); ΔAUC gold y pseudo con bootstrap pareado
(1 000). AUC absolutas infladas (los dos modelos se entrenaron con todo train): vale la diferencia.

**Criterios (fijados antes de correr):**
- H confirmada si E1 (o E2, la mejor de las dos que cumpla AUC) tiene minutos @1300 ≤ 0.60 × R **y**
  ΔAUC gold ≥ −0.005 (estimación puntual). Rechazada si ninguna cumple ambas.
- A, C, D: "ayuda" si baja los minutos @1300 ≥ 3 % con |ΔAUC gold| < 0.002 y Pearson ≥ 0.999.
- B: se reporta su ΔAUC gold / pseudo con IC; si ΔAUC gold < −0.005 la H falla por AUC aunque baje el tiempo.
- Recomendación: (1) para AUC ≈ R, la variante más rápida con |ΔAUC gold| < 0.002 y Pearson ≥ 0.999;
  (2) para eficiencia, la de mayor score relativo (gold y pseudo deben ser ambos ≥ 0; si discrepan se
  dice).
- Sanity: AUC gold de R > 0.85 y R ≈ V2 de exp026 (0.9219 gold).

**Excepción a la regla 4 (notebook):** igual que exp025/exp026: `t4_fast.py` (arnés que corre en
Kaggle), `make_kernel.py` (arma el kernel inyectando las celdas de los dos notebooks públicos, que no se
copian al repo) y `analyze.py` (AUC fuera de Kaggle). No entrena nada ni toca `src/knee/`.

**Riesgos:** compilar CoAtNet-2 con max-autotune en T4 puede tardar muchos minutos (se reporta tal cual,
cuenta en el runtime); el modelo 224 es otro entrenamiento (otra semilla de SWA y otra receta de recorte):
su ΔAUC mezcla tamaño de entrada y pesos, pero es el único 224 que existe con esa receta; 58 gold es
poco (IC del bootstrap). Fallo de una variante → se registra y la corrida sigue.

## 2. PLAN

- `make_kernel.py`: baja (o reusa) los dos notebooks públicos (`nartaa/rsna-knee-0949-anatomical-mirror`
  y `nartaa/rsna-knee-0945-efficient-224crop`), incrusta sus celdas de código en base64 + `t4_fast.py` +
  la lista de estudios (sin etiquetas); kernel privado `medellinrodrigo/exp027-t4-rapido`, fuentes
  `nartaa/rsna-knee-publication-swa-weights-20261007` + competición, T4 ×2, sin internet, imagen sin fijar
  (como exp026).
- Estudios: los 150 `main` + 2 `warmup` de exp026 y 150 de fold 0 nuevos (rol `cold`, semilla 1 sobre
  `split_v1_seed42`, excluidos los de exp026) → `exp027_studies.csv` (solo en scratchpad, con etiquetas).
- Orden en el kernel: entorno (versiones, ORT/TensorRT) → R frío → R, A, B, B94, C, E1, Rb en caliente
  (disco + RAM) → la más rápida de B/B94/E1 en frío sobre `cold` + preparación sola en caliente → D, E2,
  E2d, F (compiladas al final) → E1b (= E1 otra vez, control de carga residual tras compilar).
  Cambio tras el smoke (antes de correr en Kaggle): en el smoke todo lo que corrió después de compilar fue
  más lento, así que las compiladas van al final, se apagan los workers de inductor tras compilar y E1b lo
  controla; la variante en frío pasa a ser la 224 sin compilar más rápida.
- `analyze.py` → `results/exp027_t4_rapido/metrics.json` y la tabla; kernels de envío (sin enviar) en
  `scratchpad/kaggle_exp027/` para la mejor variante con AUC ≈ R y la más eficiente.

## 3. VERDICT (números de `results/exp027_t4_rapido/metrics.json`)

**Resultado:** didnt · **H:** rechazada por el veto de AUC gold (el tiempo sí baja: E1 = 0.56 × R, pero
ΔAUC gold −0.0072 < −0.005; el pseudo dice lo contrario, +0.0088). Lo que sí sirve sin coste de AUC:
`model.half()` (C, −8.9 % de minutos, predicciones idénticas).

Kernel `medellinrodrigo/exp027-t4-rapido` v1: 2 × Tesla T4, 4 vCPU, 33.7 GB RAM, torch 2.11.0+cu128, cuDNN
9.19, timm 1.0.29, triton 3.6.0; **sin** torch_tensorrt, tensorrt, onnxruntime ni onnx en la imagen (F no
se pudo medir; no se instaló nada). 2 299 s en total. Imports 17.2 s; carga de modelos (2 GPU, en serie)
8.3 s (384) / 4.5 s (224). nproc = 4.

Minutos @1300 = steady × 1 300/60 + fijos (imports + carga + exceso del primer estudio por hilo +
compilación). Score relativo = ΔAUC − 0.00086 · Δmin (contra R). 150 estudios (58 gold + 92 fold 0),
pipeline del notebook desde disco (page cache caliente); "RAM" = misma cola sin la preparación:

| Var | steady s/est. | RAM | GPU s/est./GPU | fijos s (compil.) | min @1300 | × R | AUC gold | pseudo 150 | Pearson vs R | media \|Δp\| | ΔAUC gold [IC95] | Δ pseudo 150 [IC95] | score gold | score pseudo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | 0.561 | 0.534 | 1.099 | 48.5 | **12.96** | 1.00 | 0.9219 | 0.9282 | 1 | 0 | 0 | 0 | 0 | 0 |
| A | 0.565 | 0.538 | 1.097 | 41.5 | 12.93 | 1.00 | 0.9219 | 0.9282 | 1.00000 | 0.0000 | +0.0000 | −0.0000 | +0.0000 | −0.0000 |
| C | 0.514 | 0.483 | 0.997 | 40.0 | **11.81** | 0.91 | 0.9219 | 0.9283 | 1.00000 | 0.0001 | +0.0000 | +0.0000 | **+0.0010** | **+0.0010** |
| D | 0.430 | — | 0.787 | 335.4 (295.7) | 14.91 | 1.15 | 0.9219 | 0.9282 | 1.00000 | 0.0000 | +0.0000 | −0.0000 | −0.0017 | −0.0017 |
| B | 0.335 | 0.251 | 0.544 | 37.3 | 7.89 | 0.61 | 0.9149 | 0.9370 | 0.96816 | 0.0473 | −0.0070 [−0.0195, +0.0046] | +0.0088 [+0.0020, +0.0156] | −0.0027 | +0.0132 |
| B94 | **0.289** | 0.243 | 0.547 | 30.4 | **6.77** | **0.52** | 0.9150 | 0.9371 | 0.96816 | 0.0473 | −0.0069 [−0.0194, +0.0049] | +0.0088 [+0.0020, +0.0156] | −0.0016 | **+0.0142** |
| E1 | 0.307 | 0.227 | 0.476 | 37.6 | 7.29 | 0.56 | 0.9148 | 0.9371 | 0.96815 | 0.0473 | −0.0072 [−0.0199, +0.0046] | +0.0088 [+0.0020, +0.0156] | −0.0023 | +0.0137 |
| E2 | 0.252 | 0.174 | 0.387 | 264.5 (227.7) | 9.86 | 0.76 | 0.9150 | 0.9371 | 0.96816 | 0.0473 | −0.0070 [−0.0197, +0.0049] | +0.0088 | −0.0043 | +0.0115 |
| E2d | 0.305 | 0.226 | 0.486 | 39.2 (0.9) | 7.26 | 0.56 | = E1 | | | | | | | |
| Rb | 0.564 | 0.533 | 1.101 | 49.1 | 13.04 | 1.01 | 0.9219 | 0.9282 | 1.00000 | 0.0000 | +0.0000 | −0.0000 | −0.0001 | −0.0001 |
| E1b | 0.312 | 0.226 | 0.485 | 37.7 | 7.39 | 0.57 | = E1 | | | | | | | |

(AUC infladas: los dos modelos se entrenaron con todo train. D perdió su corrida RAM por una carrera en el
muestreador de nvidia-smi, ya corregida en el código. E2d **no es válida**: compiló en 0.5 s y da lo mismo
que E1 → casi seguro dynamo llegó al límite de recompilaciones (`cache_size_limit` = 8: D y E2 ya ocupaban
2 modelos × 2 GPU × 2 formas) y cayó a eager. Rb ≈ R y E1b ≈ E1: no hubo deriva ni carga residual tras compilar.)

En frío y techo de la preparación: R en frío (primera lectura de disco) 0.559 s/est. = caliente. B94 sobre
150 estudios de fold 0 nuevos (primera lectura) 0.286 s/est. = caliente (0.289): el disco no importa. La
preparación sola (Pool de 4 procesos, caché caliente) da **0.210 s/est.** (4.6 min @1300): con el modelo
224 la GPU ya no manda — B94 va a 0.289 en el pipeline contra 0.243 desde RAM y 0.27 de GPU pura (2 GPU):
los 4 vCPU se reparten entre la preparación y los dos hilos de GPU.

**Criterios:**
- H: E1 = 0.562 × R (≤ 0.60 ✓) pero ΔAUC gold −0.0072 < −0.005 ✗; E2 = 0.761 × R ✗ en tiempo. **Rechazada.**
  El IC del gold incluye 0 y el pseudo-150 mejora +0.0088 con IC que excluye 0 (y +0.0137 en los 92 de
  fold 0): las dos validaciones discrepan; el gold (58) es la que veta según la SPEC.
- A (sin `cudnn.benchmark`): −0.03 min (−0.2 %) → **no ayuda** (< 3 %). El exceso del primer estudio
  baja de 23 a 16 s, pero queda init por hilo + contención de CPU; el steady no cambia.
- C (`model.half()`): −1.15 min (−8.9 %), Pearson 1.00000, |Δp| 1e-4, ΔAUC 0 → **ayuda**.
- D (`torch.compile` max-autotune): steady −24 % (0.430 vs 0.565) pero 296 s de compilación en serie
  (183 + 113 s) → +1.95 min → **no ayuda** a 1 300 estudios (empata a ≈ 2 200 estudios).
- B: ΔAUC gold −0.0070 → por debajo de −0.005: la H falla por AUC aunque baja el tiempo 39 %.
- Sanity: AUC gold de R 0.9219 = V2 de exp026 (0.9219) ✓; steady 0.561 vs 0.594 de exp026.

**Qué aprendimos:** en T4 con el modelo 384 lo único gratis es fp16 puro (−9 %); sin autoajuste no se gana
nada; compile solo compensa con ≫ 2 000 estudios o con caché de compilación. El modelo 224 del mismo autor
corta el runtime a la mitad (6.8 min @1300 con un solo lote de 94) y a partir de ahí la preparación en
CPU (0.21 s/estudio con 4 vCPU) es el siguiente cuello. Su efecto en AUC no está resuelto con nuestros
datos: gold −0.007 (IC ±0.012), pseudo +0.009. En el LB público el autor da 0.945 (224, sin TTA) contra
0.949 (384 con TTA); exp026 mostró que la TTA vale ≈ 0 en gold, así que el 384 sin TTA debería estar en
≈ 0.947–0.949: si es así, B94 pierde ≈ 0.002–0.004 de LB y ahorra 6.2 min (≈ +0.0053) → ganaría en
eficiencia. Eso solo lo decide un envío.

**Recomendación:**
1. AUC ≈ R: **C** (384 sin TTA + `model.half()` + channels_last + sin benchmark): 11.8 min estimados, mismas
   predicciones que V2. Kernel listo: `scratchpad/kaggle_exp027/submit_C` (probado en rsna-f con los 3
   estudios de test: escribe `submission.csv`).
2. Eficiencia: por la regla de la SPEC (score gold y pseudo ≥ 0) solo C cumple (+0.0010). La variante más
   eficiente medida es **B94** (6.77 min, score pseudo +0.0142, gold −0.0016): enviarla como apuesta y
   decidir con el LB. Kernel listo: `scratchpad/kaggle_exp027/submit_B94`; alternativa sin medir en tiempo
   `submit_E1_94` (= B94 + `model.half()`, GPU −13 % en E1 vs B, probado en los 3 estudios de test).
3. Siguiente (exp028): la preparación en CPU ya es el cuello con el 224 (0.21 s/estudio, 4 vCPU): menos
   hilos/procesos que compitan con la GPU, o mover percentiles/resize a la GPU.
