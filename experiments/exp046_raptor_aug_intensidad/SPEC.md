# exp046_raptor_aug_intensidad

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | exp046_raptor_aug_intensidad |
| Estado | spec |

Idea: paso 7, track T2a. La receta Raptor (exp031) solo tiene ganancia ×U(0.9,1.1) como augmentation de intensidad.
exp040 (robustez de exp029): ruido 5 % → −0.011; sitio croata 0.817 vs inglés 0.880. Augmentation de intensidad
(gamma, contraste, ruido, blur, campo de sesgo) apunta a ese gap de sitio/escáner.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** añadir en entrenamiento augmentation de intensidad por estudio (gamma, contraste, ruido gaussiano, blur
gaussiano y campo de sesgo suave) sube la pseudo AUC L2 en fold 0 en ≥ +0.003 contra exp031, porque hace al modelo
robusto a diferencias de escáner/protocolo entre sitios, sin que gold caiga más de 0.02.

**Única variable:** augmentation de intensidad de entrenamiento (un único bloque "intensidad", pedido así en el plan
del paso 7 como una sola variable). Se mantiene la ganancia ×U(0.9,1.1) original.

| Componente (train, por estudio = mismos parámetros para las K ventanas) | Padre (exp031) | Este experimento |
|---|---|---|
| Ganancia | ×U(0.9, 1.1), clamp 0–1 | igual |
| Gamma | no | x^γ, γ ~ U(0.7, 1.5), p 1 |
| Contraste | no | (x − μ)·c + μ, c ~ U(0.8, 1.2), μ = media del estudio, p 1 |
| Campo de sesgo | no | multiplicativo 1 + a·P(u,v)/max\|P\|, P polinomio de grado 2 en coordenadas [−1,1] con coeficientes N(0,1), a ~ U(0, 0.15) (amplitud ≤ ±15 %), mismo campo para las K ventanas, p 0.3 |
| Blur gaussiano | no | σ ~ U(0.5, 1.0) px, kernel separable 7×7, p 0.25 |
| Ruido gaussiano | no | aditivo N(0, σ²) por píxel, σ ~ U(0, 0.05), p 0.5 |

Orden en GPU (`to_input`, solo con `train=True`): uint8 → /255 → ganancia (padre) → clamp → gamma → contraste → sesgo →
blur → ruido → clamp 0–1 → normalización ImageNet. Validación, gold e inferencia: idénticas al padre (sin aug).

Todo lo demás idéntico al padre: etiquetas de consenso (Raptor + L2 + L3 + L4), corpus 96×384 en `/root/corpus96_384`,
recorte central 320, `RaptorClassifier`, 16 épocas, bs 8 × K 12 (K_EVAL 24), AdamW/OneCycle/LRs, bf16, semilla 42,
`split_v1_seed42` fold 0, selección por pseudo L2 (L2 binarizada > 0.5), SWA top-3 por pseudo L2.

**Métrica principal:** pseudo L2 fold 0 de la mejor época (como el resto del paso 7); SWA top-3 se reporta.

**Criterios (fijos, no cambian después de correr):**
- `worked` (promete): pseudo L2 fold 0 ≥ exp031 + 0.003 (= **≥ 0.8866**) **y** gold ≥ 0.8944 (no cae > 0.02).
- `killed`: pseudo L2 fold 0 ≤ exp031 − 0.005 (= **≤ 0.8786**) **o** gold < 0.8944, o GPU-h > 1.5 × 2.49.
- `didnt`: cualquier otro caso.
(Padre exp031, `results/exp031_raptor_etiquetas_consenso/metrics.json`: pseudo L2 0.8836 (ep 15), SWA 0.8835, gold 0.9144.)
Ruido de semilla medido (sub031_2): 0.0013. La `decision_auto` del notebook usa ±0.005; el verdict se aplica con estos criterios.

**Si promete:** run all-data en el mismo pod (`ALL_DATA True`, `SWA_EPOCHS` = 3 mejores épocas por pseudo L2 del fold 0).
**Si queda por debajo pero cerca** (0.8836 ≤ pseudo < 0.8866): sub046_1 con una sola variable de intensidad (versión
suave o fuerte del bloque).

**Presupuesto:** ≈ 2.5 GPU-h fold 0 en rsna-i (RTX PRO 4500 32 GB); la aug en GPU cuesta < 5 % extra.

**Riesgos:** gamma U(0.7,1.5) es asimétrico (oscurece en promedio); contraste < 1 levanta el fondo negro; la
validación (sin aug) no mide robustez de sitio directamente — eso lo mediría el harness de exp040.

## 2. PLAN

- Notebook `exp046_raptor_aug_intensidad.ipynb` = copia del de exp031; cambios:
  - celda 0: `EXP`, `PADRE`, parámetros `AUG_GAMMA = (0.7, 1.5)`, `AUG_CONTRAST = (0.8, 1.2)`, `AUG_NOISE_P = 0.5`,
    `AUG_NOISE_SD = 0.05`, `AUG_BLUR_P = 0.25`, `AUG_BLUR_SIGMA = (0.5, 1.0)`, `AUG_BIAS_P = 0.3`, `AUG_BIAS_AMP = 0.15`, `NOTES`.
  - celda 3: `to_input(x, gain, train=False)` aplica la cadena de intensidad en GPU solo si `train`.
  - celda 4: el bucle de entrenamiento llama `to_input(x, g, train=True)`; `predict` sin cambios.
- Smoke: `SMOKE=True` (64 train / 48 val, 1 época).
- Run real: `FOLDS=[0]`, `SMOKE=False`, en rsna-i.

## 3. VERDICT (números de `results/exp046_raptor_aug_intensidad/metrics.json`)

**Resultado:** pendiente · **Fase alcanzada:** spec

| Métrica | Padre (exp031) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8836 (ep 15) | | |
| pseudo L2 fold 0 (SWA top-3) | 0.8835 | | |
| v1 fold 0 | 0.8368 | | |
| gold (veto) | 0.9144 | | |
| GPU-h | 2.49 | | |

**Qué aprendimos:**

**Siguiente paso sugerido:**
