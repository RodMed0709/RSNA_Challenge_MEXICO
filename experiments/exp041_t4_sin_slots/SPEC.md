# exp041_t4_sin_slots

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp027_t4_rapido (variante C: Raptor 384/320 sin TTA + `model.half()` + channels_last + sin `cudnn.benchmark`, 11.8 min @1300) |
| Fecha | 2026-10-11 |
| Rama | exp041_t4_sin_slots |
| Estado | spec |

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

## 3. VERDICT

(pendiente)
