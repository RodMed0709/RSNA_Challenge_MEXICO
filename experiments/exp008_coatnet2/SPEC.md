# exp008_coatnet2

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp002_dinov2s_ref |
| Fecha | 2026-09-26 |
| Rama | paso1-backbones |
| Estado | didnt |

Idea: paso 1 de `docs/Hoja_de_Ruta_Eficiencia.docx` (barrido de backbones). Backbone CoAtNet-RMLP-2 224 (híbrido, Apache-2.0).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si cambio el backbone de DINOv2-S/14 a CoAtNet-RMLP-2 224 (híbrido, Apache-2.0), la pseudo AUC se mantiene o sube con menos o igual
costo por estudio. Familia del mejor notebook público (~0.94); techo de costo del barrido.

**Única variable cambiada:**

| Parámetro o celda | Padre | Este experimento |
|---|---|---|
| Backbone (celdas 0 y 3) | `dinov2-small` (HF, local) | `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k` (timm) |

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

**Presupuesto:** screen 4.0 GPU-h (A100 80GB o RTX PRO 6000) · 5 folds 20.0 GPU-h.

**Riesgos:** lr del backbone pensado para ViT (las CNN suelen querer más); memoria con 160 imágenes por paso;
licencia de pesos para el premio.

## 2. PLAN

- Celda 0: `PADRE = "exp002"`, `TIMM_NAME = "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k"` en lugar de `BACKBONE_PATH`.
- Celda 3: `KneeModel(timm_name=TIMM_NAME, gradient_checkpointing=True)`.
- Celda 6: añade `metrics["cost"]` con `src/knee/cost.measure_cost` (solo medición, no cambia el modelo).
- Smoke con `SMOKE = True`; después `FOLDS = [0]`, `SMOKE = False`.

## 3. VERDICT (números de `results/exp008_coatnet2/metrics.json`)

**Resultado:** didnt · **Fase alcanzada:** screen (fold 0)

| Métrica | Padre | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | 0.8266 | 0.8141 | −0.0125 |
| gold AUC fold 0 | 0.7765 | 0.7734 | −0.0031 |
| GMACs por estudio | 489.9 | 1201.7 | +145 % (×2.45) |
| ms por estudio (A100, `cost_a100`) | 14.6 | 65.5 | +350 % |
| GPU-h | 1.46 (f0, RTX PRO 6000) | 3.62 (A100 80GB PCIe) | — |

Δ gold es fold 0 contra fold 0 del padre. La celda 5 lo comparó contra el gold promedio de 5 folds del padre
(0.7884) y dio `didnt`; corregido en `metrics.json` (ver `note`).
Costo de `results/paso1_cost_a100.json`, medido aparte en la misma A100; el `cost` del run se mantiene.

**Qué aprendimos:** el más cercano al padre (−0.012 pseudo, −0.003 gold) pero el más caro: 1201.7 GMACs (+145 %) y
65.5 vs 14.6 ms. Aun empatando en AUC quedaría dominado por el padre en el frente de Pareto.

**Siguiente paso sugerido:** pasa al paso 2 por la regla del barrido (1.º no `killed`), aunque para el track de eficiencia está dominado.
