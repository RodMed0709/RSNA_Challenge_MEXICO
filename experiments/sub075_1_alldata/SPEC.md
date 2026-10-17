# sub075_1_alldata

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp075_raptor_ssl_dino (fold 0 SWA: pseudo L2 0.8903, gold 0.9219; LB solo f0 0.941) |
| Fecha | 2026-10-17 |
| Rama | sub075_1_alldata |
| Estado | pendiente LB |

Idea: sonda LB autorizada por Rodrigo (17-oct). exp075 (init DINO) sacó 0.941 en LB con solo el fold 0, igual que
exp045 con 5 folds. ¿El init DINO en all-data supera al init nartaa en all-data (exp071 solo 0.944)?

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** el init DINO ayuda más en LB que lo que mide la pseudo fold 0. Entrenado con todos los datos (como
exp071) debería quedar al nivel o por encima del init nartaa sin datos externos.

**Única variable cambiada:**

| Parámetro o celda | Padre (exp075) | Este experimento |
|---|---|---|
| Modo (`ALL_DATA`, `SWA_EPOCHS`, celda 0) | screen fold 0 (train folds 1–4, SWA = top-3 por pseudo) | all-data: 4349 estudios no-gold, sin validación, SWA sobre épocas fijas [13, 14, 15] (las del all-data de exp045/exp071) |

El init es el **mismo** backbone DINO de exp075 (`ssl_backbone.pt`, sha del run del 16-oct; se carga con `SSL_CKPT`, no se
re-entrena el SSL; ese SSL solo vio folds 1–4, sin fold 0 ni gold). Receta exp045 tal cual (16 épocas, aug espacial,
consenso, LRs, semilla 42). La celda 6 copia `swa_alldata.pt` para el kernel (cambio de ruta del export, no de receta).

**Criterios (no cambian después de correr):**
- **Decide el LB público** del kernel solo `sub075-1-solo` = `exp075-solo` con un cambio (checkpoint = SWA all-data de
  sub075_1).
  - `worked` si LB ≥ 0.946 (exp071 0.944 + ~0.002; Δ < 0.004 es ruido de ~390 estudios, por eso el umbral es el de exp071).
  - `didnt` si 0.941 ≤ LB < 0.946.
  - `killed` si LB < 0.941 (peor que el propio fold 0).
- **Sanidad (no decide):** gold del SWA all-data ≥ 0.90; si no → `killed` y no se envía.
- Pseudo fold 0 no existe (fold 0 dentro del train).

**Presupuesto:** ≤ $6. Fine-tune all-data ≈ 2.8 GPU-h en RTX PRO 4500 ($0.72/h) ≈ $2.1 (exp045 all-data: 2.81 GPU-h).

**Riesgos:** ninguno nuevo; el kernel ya corre (exp075-solo COMPLETE).

## 2. PLAN

- Celdas que cambian contra exp075: 0 (EXP, PADRE, `ALL_DATA = True`, `SWA_EPOCHS = [13, 14, 15]`, `SSL_CKPT`, NOTES) y
  6 (export: ruta del SWA all-data). Ninguna otra.
- Cadena en el pod: smoke (`SMOKE True`) → all-data → export → quita guard_skip → para el pod. Finisher en rsna-cron:
  dataset privado + kernel privado + validación COMPLETE sin Traceback → `orq/ssl/CANDIDATE.json` → cola del 18-oct.

## 3. VERDICT (números de `results/sub075_1_alldata/alldata/metrics.json`)

**Resultado:** pendiente LB · **Fase alcanzada:** all-data

| Métrica | exp045 all-data | exp071 all-data (init nartaa) | sub075_1 |
|---|---|---|---|
| gold SWA [13,14,15] (sanidad ≥ 0.90) | 0.9186 | 0.917 (contaminado) | 0.9195 |
| GPU-h | 2.81 (PRO 4500) | 1.49 (5090) | 2.78 (PRO 4500) |
| LB público solo | 0.941 (5 folds) | 0.944 | pendiente LB |

Sanidad gold OK (0.9195 ≥ 0.90) → se construye el kernel `medellinrodrigo/sub075-1-solo` (exp075-solo con un cambio:
checkpoint `sub075_1_swa_alldata.pt`, sha256 1a9e32e9…), dataset privado `medellinrodrigo/rsna-knee-sub075-1-weights`.
El verdict lo decide el LB (worked ≥ 0.946 · didnt 0.941–0.945 · killed < 0.941). Kernel v1 COMPLETE (17-oct 20:29 UTC, log sin Traceback) → `orq/ssl/CANDIDATE.json` y en la cola del 18-oct (4.ª entrada). LB: pendiente LB.
