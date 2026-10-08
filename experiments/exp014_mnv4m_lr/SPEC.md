# exp014_mnv4m_lr

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp003_mnv4m |
| Fecha | 2026-10-08 |
| Rama | exp014_mnv4m_lr |
| Estado | spec |

Idea: MobileNetV4-M con lr de backbone 1e-4 (receta CNN) en lugar de 1e-5 (receta pensada para DINOv2).

Origen: `docs/pasos/analisis_eficiencia_paso1.md`. exp003 se mató (gold −0.076), pero con el lr del ViT. MobileNetV4-M es
el único backbone no dominado en latencia junto con DINOv2-S y tiene −86 % GMACs: es el candidato natural a estudiante
eficiente. Antes de descartarlo hay que darle su receta.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si cambio `BACKBONE_LR` de 1e-5 a 1e-4, la pseudo AUC de MobileNetV4-M sube, porque las CNN con BatchNorm
preentrenadas en ImageNet necesitan pasos más grandes para adaptarse a RM; con 1e-5 el backbone casi no se movió.

**Única variable cambiada:**

| Parámetro o celda | Padre (exp003) | Este experimento |
|---|---|---|
| `BACKBONE_LR` | 1e-5 | 1e-4 |

Todo lo demás igual: split, semilla, épocas, etiquetas v1, cache v1_256, 224 px, cabeza, `SELECT_ON = "pseudo"`.

**Efecto esperado:** Δ pseudo fold 0 vs exp003 ≈ +0.03 a +0.06; costo igual.

**Criterios (no cambian después de correr):**
- Contra el padre (exp003): éxito Δ pseudo fold 0 ≥ +0.005; kill Δ ≤ −0.005 o gold < exp003 − 0.02.
- Lectura de eficiencia (no decide el veredicto, se reporta): distancia a exp002 en pseudo y gold con −86 % GMACs.
  Si queda a ≤ 0.02 de exp002 en pseudo, MobileNetV4-M pasa a ser el estudiante candidato del paso de eficiencia.

**Presupuesto:** screen ~1 GPU-h (RTX PRO 4500).

**Riesgos:** lr alto puede desestabilizar al descongelar; se ve en `history_f0.csv`.

## 2. PLAN

- Copia del notebook de exp003. Celda que cambia: 0 (`BACKBONE_LR`). Ninguna otra.
- Smoke con `SMOKE = True` antes del run real.

## 3. VERDICT (números de `results/exp014_mnv4m_lr/metrics.json`)

**Resultado:** worked / didnt / killed · **Fase alcanzada:** screen

| Métrica | Padre (exp003) | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | 0.743 | | |
| gold AUC fold 0 | 0.701 | | |
| Distancia a exp002 (pseudo f0 0.827) | | | |
| GPU-h | | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
