# exp027_t4_rapido

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp026_t4_variantes (V2: Raptor/0.949 sin TTA + channels_last, 13.7 min estimados @1300 en T4 ×2) |
| Fecha | 2026-10-11 |
| Rama | exp027_t4_rapido |
| Estado | spec |

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

## 3. VERDICT

(pendiente)
