# exp041_t4_sin_slots

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp027_t4_rapido (variante C: Raptor 384/320 sin TTA + `model.half()` + channels_last + sin `cudnn.benchmark`, 11.8 min @1300) |
| Fecha | 2026-10-11 |
| Rama | exp041_t4_sin_slots |
| Estado | didnt |

Idea: exp040 midió en el Raptor fold 0 (RunPod) que quitar el slot Axial (4) y el Coronal no fluido (3) cuesta
≈ 0 de pseudo AUC (Δ 0.0000 y −0.0003), que la TTA espejo aporta +0.0016 y que más ventanas sí importan. Aquí se
mide en la T4 real de Kaggle (mismo marco que exp027) cuánto runtime ahorra **no leer ni procesar** esos dos slots
y cuánto AUC cuesta con los pesos públicos.

## 1. SPEC

**Hipótesis (H):** C3 (C sin los slots 3 y 4) baja los minutos @1300 ≥ 25 % contra C con ΔAUC gold ≥ −0.005 y
Δ pseudo-150 ≥ −0.003.

**Cómo se quita un slot (igual que exp040):** el corpus representa una serie ausente como cortes a cero con
máscara 0. Aquí la preparación usa el mismo lector `study_lut` del notebook con `SLOTS[:3]` (Sag fluido 26,
Sag no fluido 22, Cor fluido 18 = 66 cortes): **no se listan ni se leen** los DICOM del Cor no fluido ni del
Axial, y la elección de series de los slots 0–2 no cambia (se eligen en el mismo orden). El volumen se rellena
con ceros hasta 96 cortes (máscara 0) en el consumidor GPU, que es la representación de exp040. Las ventanas
salen de `_eval_centers` sobre la máscara: con 66 cortes válidos hay 64 centros; C3 usa **K = 64** (una
ventana por centro, la misma densidad que las 94 de C en los slots que quedan; con K = 94 solo habría
ventanas duplicadas). El modelo no tiene posición por ventana (MIL con atención), así que no "espera" los 5
slots más allá de las ventanas que ve.

**Variantes (mismos 150 estudios):**

| Var | Qué es | Contra |
|---|---|---|
| C | exp027 C: 384, recorte 320, 94 ventanas, sin TTA, half, channels_last, trozos de 16, sin benchmark | referencia (re-medida) |
| Ch | C con el modelo 224 del autor en un lote de 94 + half (= `submit_E1_94` de exp027, no medido allí) | C (ancla del 224 con 5 slots) |
| C3 | C sin slots 3 y 4: lectura de 3 slots, K = 64 | C |
| C3h | C3 con el modelo 224 (recorte 0.8 → 224, un lote de 64, half) | Ch y C |
| C3t | C3 + TTA espejo anatómica (64 + 64 ventanas, media de sigmoides como el notebook) | C3 y C |
| Cb | C repetida al final (deriva) | C |

**Qué se mide:** pipeline del notebook (Pool de 4 procesos + un hilo por GPU) desde disco con caché caliente y
desde RAM; preparación sola (cola sin GPU) con 5 y con 3 slots sobre los mismos estudios; y en frío (primera
lectura) la variante de 3 slots más rápida sobre 150 estudios nuevos de fold 0 y su par de 5 slots sobre otros
150 nuevos (son estudios distintos: se compara el orden de magnitud, no décimas).

**Minutos @1300**, ΔAUC gold y pseudo (150 y 92 de fold 0) con bootstrap pareado (1 000), Pearson y |Δp|,
y score de eficiencia relativo = ΔAUC − 0.00086 · Δmin, todo **contra C** (y C3h también contra Ch).

**Criterios (fijados antes de correr):**
- H confirmada si C3 ≤ 0.75 × C en minutos @1300 **y** ΔAUC gold ≥ −0.005 **y** Δ pseudo-150 ≥ −0.003.
- C3h: se reporta contra C y contra Ch con los mismos umbrales de AUC.
- C3t: "vale la pena" si su Δ pseudo-150 contra C3 ≥ +0.002 y su score de eficiencia contra C es mayor que el de C3.
- Recomendación: (1) AUC ≈ C: la más rápida con |ΔAUC gold| ≤ 0.003 y |Δ pseudo| ≤ 0.002 contra C;
  (2) eficiencia: la de mayor score (gold y pseudo; si discrepan se dice).
- Sanity: C reproduce exp027 (AUC gold 0.9219; ~0.51 s/estudio steady).

**Excepción a la regla 4 (notebook):** como exp025–exp027: `t4_slots.py` (arnés en Kaggle), `make_kernel.py`
(inyecta las celdas de los notebooks públicos, que no van al repo) y `analyze.py`. No entrena ni toca `src/knee/`.

**Riesgos:** algunos estudios no tienen Cor fluido o lo tienen con otra serie; con slots faltantes el rango
válido cambia y `_eval_centers` repite centros (igual que el notebook). 58 gold es poco. El frío compara
estudios distintos.

## 2. PLAN

- `exp041_studies.csv` (scratchpad): 150 main + 2 warmup de exp026/027, y 2 × 150 de fold 0 nuevos (`cold5`,
  `cold3`; semilla 2; excluidos los usados en exp026 y exp027).
- Kernel privado `medellinrodrigo/exp041-t4-sin-slots` (T4 ×2, sin internet, mismas fuentes que exp027).
- Orden: C (primera lectura de los main, guarda volúmenes de 5 slots) → C, Ch, C3 (guarda volúmenes de
  3 slots), C3h, C3t en caliente (disco + RAM) → Cb → prep sola 5 y 3 slots → frío 3 slots (`cold3`) y su par
  de 5 slots (`cold5`).
- `analyze.py` → `results/exp041_t4_sin_slots/metrics.json`; kernels de envío (sin enviar) en
  `scratchpad/kaggle_exp041/` para la mejor.

## 3. VERDICT (números de `results/exp041_t4_sin_slots/metrics.json`)

**Resultado:** didnt · **H:** rechazada. Quitar los slots 3 y 4 sí ahorra tiempo (C3 = 0.71 × C), pero con los
pesos públicos **sí cuesta AUC**: gold −0.0092 [−0.0175, −0.0024] (el IC excluye 0) y pseudo-150 −0.0032.
Lo de exp040 (≈ 0) no se traslada a este modelo.

Kernel `medellinrodrigo/exp041-t4-sin-slots` v1: 2 × T4, 4 vCPU, torch 2.11.0+cu128, 1 152 s. Volumen de 3
slots idéntico a los primeros 66 cortes del de 5 (comprobado en el kernel). Imports 13 s; carga de modelos
8.9 s (384) / 6.1 s (224).

Minutos @1300 = steady × 1 300/60 + fijos (imports + carga + exceso del primer estudio por hilo). Todo contra C
(misma corrida). Score = ΔAUC − 0.00086 · Δmin:

| Var | steady s/est. | RAM | GPU s/est./GPU | min @1300 | × C | AUC gold | pseudo 150 | Pearson vs C | ΔAUC gold [IC95] | Δ pseudo 150 [IC95] | score gold | score pseudo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C | 0.536 | 0.517 | 1.034 | **12.25** | 1.00 | 0.9219 | 0.9283 | 1 | 0 | 0 | 0 | 0 |
| Ch (224, 94 ventanas, half) | 0.282 | 0.238 | 0.533 | **6.57** | 0.54 | 0.9148 | 0.9371 | 0.96815 | −0.0072 [−0.0199, +0.0046] | +0.0088 [+0.0020, +0.0156] | −0.0023 | **+0.0137** |
| C3 (sin slots 3/4, K 64) | 0.379 | 0.348 | 0.741 | 8.72 | 0.71 | 0.9127 | 0.9250 | 0.99068 | −0.0092 [−0.0175, −0.0024] | −0.0032 [−0.0067, +0.0002] | −0.0062 | −0.0002 |
| C3h (224 sin slots 3/4) | **0.187** | 0.167 | 0.353 | **4.51** | **0.37** | 0.9088 | 0.9331 | 0.96045 | −0.0132 [−0.0270, −0.0010] | +0.0048 [−0.0023, +0.0118] | −0.0065 | +0.0115 |
| C3t (C3 + TTA espejo) | 0.728 | 0.712 | 1.417 | 16.28 | 1.33 | 0.9139 | 0.9241 | 0.98938 | −0.0080 [−0.0168, −0.0011] | −0.0042 [−0.0080, −0.0004] | −0.0115 | −0.0076 |
| Cb (C repetida) | 0.557 | 0.524 | 1.074 | 12.71 | 1.04 | 0.9219 | 0.9283 | 1.00000 | 0 | 0 | −0.0004 | −0.0004 |

C3h contra Ch (efecto de quitar los slots con el 224): ΔAUC gold −0.0060 [−0.0146, +0.0014], Δ pseudo −0.0040
[−0.0074, −0.0007], −2.06 min. C3t contra C3: gold +0.0012, pseudo −0.0010, ×1.87 de tiempo.
(AUC absolutas infladas: los modelos se entrenaron con todo train. Cb salió un 4 % más lenta que C: ruido o
deriva de la corrida; el orden de las variantes no cambia.)

Lectura y preparación: la preparación sola (Pool de 4, caché caliente) baja de **0.172 a 0.122 s/est.** con 3
slots (−29 %: menos cabeceras, píxeles y decodificación). En frío sobre estudios nuevos: C3h 0.201 s/est.
(150 de `cold3`) contra Ch 0.279 (150 de `cold5`), ≈ los mismos valores que en caliente (0.187 / 0.282): el
disco no limita. Con el 224 el ahorro de quitar slots se reparte entre GPU (64 en vez de 94 ventanas) y CPU.

**Criterios:**
- H: C3 = 0.712 × C (≤ 0.75 ✓) pero ΔAUC gold −0.0092 < −0.005 ✗ y Δ pseudo −0.0032 < −0.003 ✗. **Rechazada.**
- C3h: contra C gold −0.0132 ✗; contra Ch gold −0.0060 ✗, pseudo −0.0040 ✗.
- C3t: Δ pseudo contra C3 −0.0010 (< +0.002) y score peor que C3 → **no vale la pena**: la TTA no recupera lo
  perdido.
- Sanity: C gold 0.9219 = exp027; steady 0.536 (exp027 C 0.514) ✓.

**Qué aprendimos:** con los pesos públicos el Axial y el Cor no fluido sí aportan (sobre todo en gold; exp040
lo midió con nuestro modelo fold 0, otra receta y otros datos de evaluación). El ahorro de quitarlos (−29 % de
preparación, −32 % de ventanas) no compensa: C3 pierde lo mismo en AUC que lo que gana en minutos (score pseudo
−0.0002, gold −0.0062), y C3h queda dominada por Ch en pseudo (+0.0115 vs +0.0137). Ch (= `submit_E1_94` de
exp027) queda medida: **6.57 min @1300**, la mejor en score pseudo.

**Recomendación:**
1. AUC ≈ C: **C** (ninguna variante sin slots cumple |ΔAUC gold| ≤ 0.003). Kernel: `scratchpad/kaggle_exp041/submit_C`.
2. Eficiencia: **Ch** (224 + half + un lote de 94, todos los slots): score pseudo +0.0137, gold −0.0023 (las
   dos validaciones discrepan, como en exp027; el gold veta según la regla pero su IC incluye 0). Kernel:
   `scratchpad/kaggle_exp041/submit_E1_94`. C3h solo si el LB mostrara que el 224 no pierde nada con menos slots.
3. No seguir por la vía de quitar slots con estos pesos; si se quiere recortar lectura, mejor 50 % de cortes
   por slot (exp040: −0.007) o la preparación en GPU.
