# exp006_dinov3s

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp002_dinov2s_ref |
| Fecha | 2026-09-26 |
| Rama | paso1-backbones |
| Estado | killed |

Idea: paso 1 de `docs/Hoja_de_Ruta_Eficiencia.docx` (barrido de backbones). Backbone DINOv3 ViT-S/16 (licencia DINOv3).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si cambio el backbone de DINOv2-S/14 a DINOv3 ViT-S/16 (licencia DINOv3), la pseudo AUC se mantiene o sube con menos o igual
costo por estudio. Sucesor directo del padre, ~21 % menos GMACs por corte que DINOv2-S/14.

**Única variable cambiada:**

| Parámetro o celda | Padre | Este experimento |
|---|---|---|
| Backbone (celdas 0 y 3) | `dinov2-small` (HF, local) | `vit_small_patch16_dinov3.lvd1689m` (timm) |

Todo lo demás igual: split `split_v1_seed42`, semilla 42, 224 px, 5 series × 16 cortes, 8 épocas (1 congelada),
lr cabeza 1e-4 / backbone 1e-5, etiquetas v1. La receta se afinó para DINOv2: es una desventaja conocida para los
demás backbones y se revisa en el paso 2.

**Efecto esperado:** Δ pseudo fold 0 entre −0.02 y +0.02; el costo lo dice `metrics.json["cost"]`.

**Criterios (no cambian después de correr):**
- Éxito → 5 folds: Δ pseudo fold 0 ≥ +0.005.
- Kill: Δ pseudo fold 0 ≤ −0.005, o gold < padre − 0.02, o GPU-h > 1.5 × presupuesto.
- Neutral: cuenta como `didnt`, salvo ahorro de cómputo ≥ 30 % (GMACs por estudio contra el padre).
- Regla del paso 1 (para escoger 3 backbones que pasan al paso 2): se ordenan por pseudo AUC fold 0 entre los no
  `killed`; empate (|Δ| < 0.005) lo gana el de menos GMACs por estudio.

**Presupuesto:** screen 2.0 GPU-h (A100 80GB o RTX PRO 6000) · 5 folds 10.0 GPU-h.

**Riesgos:** lr del backbone pensado para ViT (las CNN suelen querer más); memoria con 160 imágenes por paso;
licencia de pesos para el premio.

## 2. PLAN

- Celda 0: `PADRE = "exp002"`, `TIMM_NAME = "vit_small_patch16_dinov3.lvd1689m"` en lugar de `BACKBONE_PATH`.
- Celda 3: `KneeModel(timm_name=TIMM_NAME, gradient_checkpointing=True)`.
- Celda 6: añade `metrics["cost"]` con `src/knee/cost.measure_cost` (solo medición, no cambia el modelo).
- Smoke con `SMOKE = True`; después `FOLDS = [0]`, `SMOKE = False`.

## 3. VERDICT (números de `results/exp006_dinov3s/metrics.json`)

**Resultado:** killed · **Fase alcanzada:** screen (fold 0)

| Métrica | Padre | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | 0.8266 | 0.7793 | −0.0473 |
| gold AUC fold 0 | 0.7765 | 0.7453 | −0.0312 |
| GMACs por estudio | 489.9 | 375.9 | −23 % (×0.77) |
| ms por estudio (A100, `cost_a100`) | 14.6 | 16.7 | +15 % |
| GPU-h | 1.46 (f0, RTX PRO 6000) | 0.78 (RTX 4090) | — |

Δ gold es fold 0 contra fold 0 del padre. La celda 5 lo comparó contra el gold promedio de 5 folds del padre
(0.7884) y dio `killed`; corregido en `metrics.json` (ver `note`).
Costo de `results/paso1_cost_a100.json`, medido aparte en la misma A100; el `cost` del run se mantiene.

**Qué aprendimos:** mismo tamaño que DINOv2-S y peor en todo: −0.047 pseudo, −0.031 gold (veto). 375.9 GMACs (−23 %),
16.7 vs 14.6 ms. Corrió en RTX 4090 (los demás en A100): sus GPU-h no se comparan. Con patch 16 a 224 px ve 196 tokens
por corte contra 256 de DINOv2/14; puede influir, no se probó.

**Siguiente paso sugerido:** archivar.
