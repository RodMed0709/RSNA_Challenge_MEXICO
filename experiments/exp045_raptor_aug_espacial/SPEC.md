# exp045_raptor_aug_espacial

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | exp045_raptor_aug_espacial |
| Estado | worked · all-data hecho |

Idea: paso 7, track T2a. La receta Raptor (exp031) casi no tiene augmentation (solo ganancia ×U(0.9,1.1) y muestreo
aleatorio de ventanas). Añadir augmentation espacial por estudio: espejo L/R anatómico + afín leve.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** añadir en entrenamiento (a) espejo L/R anatómico con p 0.5 por estudio y (b) afín aleatorio por estudio
(rotación ±10°, escala 0.9–1.1, traslación ±6 %) sube la pseudo AUC L2 en fold 0 en ≥ +0.003 contra exp031, porque
duplica efectivamente los datos (rodilla izquierda ↔ derecha) y quita sensibilidad a la posición/escala del recorte
físico de 140 mm, sin que gold caiga más de 0.02.

**Única variable:** augmentation espacial de entrenamiento (un único bloque "espacial": espejo + afín, pedido así en el
plan del paso 7 como una sola variable).

| Parámetro o celda | Padre (exp031) | Este experimento |
|---|---|---|
| Espejo L/R (train) | no | p 0.5 por estudio, el mismo eje que el TTA espejo de nartaa (`_mirror_tri`): ventanas cuyo corte central cae en slots sagitales (0, 1: cortes 0–47) → invertir el orden de los 3 cortes-canal; coronal/axial (slots 2–4) → volteo del ancho (W) |
| Afín (train) | no | por estudio (misma matriz para las K ventanas del estudio): rotación U(−10°, 10°), escala U(0.9, 1.1), traslación U(−6 %, 6 %) del lado en x e y; `F.affine_grid` + `F.grid_sample` bilineal, relleno 0, en GPU sobre (B·K, 3, H, W), p 1 |

Orden en GPU (`to_input`, solo con `train=True`): uint8 → /255 → espejo → afín → ganancia ×U(0.9,1.1) del padre → clamp
0–1 → normalización ImageNet. Validación, gold e inferencia: idénticas al padre (sin aug).

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
**Si queda por debajo pero cerca** (0.8836 ≤ pseudo < 0.8866): sub045_1 con una sola variable de intensidad (versión
suave o fuerte del bloque espacial), decidido antes de verlo según la pista de la curva.

**Presupuesto:** ≈ 2.5 GPU-h fold 0 en rsna-h (RTX PRO 4500 32 GB); el afín en GPU cuesta < 5 % extra.

**Riesgos:** el espejo sagital (invertir canales) es exacto solo dentro de cada ventana; la bolsa MIL no usa posición, así
que equivale al espejo del volumen. Ventanas que cruzan frontera de slot usan el slot del corte central (igual que el
TTA). El afín interpola (suaviza ligeramente) siempre; puede retrasar la convergencia en 16 épocas.

## 2. PLAN

- Notebook `exp045_raptor_aug_espacial.ipynb` = copia del de exp031; cambios:
  - celda 0: `EXP`, `PADRE`, parámetros `AUG_FLIP_P = 0.5`, `AUG_ROT_DEG = 10`, `AUG_SCALE = (0.9, 1.1)`,
    `AUG_TRANS = 0.06`, `NOTES`.
  - celda 3: `StudyWindows.__getitem__` devuelve además la máscara sagital por ventana (slot del corte central);
    `to_input(x, gain, sag=None, train=False)` aplica espejo + afín en GPU solo si `train`.
  - celda 4: el bucle de entrenamiento pasa `sag` y `train=True`; `predict` ignora `sag` (sin aug).
- Smoke: `SMOKE=True` (64 train / 48 val, 1 época) + comprobación visual/numérica del espejo contra `_mirror_tri`.
- Run real: `FOLDS=[0]`, `SMOKE=False`, en rsna-h.

## 3. VERDICT (números de `results/exp045_raptor_aug_espacial/metrics.json`)

**Resultado:** `worked` (promete) · **Fase alcanzada:** screen fold 0 → all-data hecho en rsna-h

All-data (`results/exp045_raptor_aug_espacial/alldata/metrics.json`): 4349 estudios no-gold, 16 épocas, SWA sobre épocas
[13, 14, 15] (las 3 mejores por pseudo L2 del fold 0), gold 0.9186 (solo informativo), 2.81 GPU-h. Pesos en el volumen:
`/workspace/users/rodrigo/wt_exp045_raptor_aug_espacial/results/exp045_raptor_aug_espacial/swa_alldata.pt`.

| Métrica | Padre (exp031) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8836 (ep 15) | 0.8899 (ep 15) | **+0.0063** |
| pseudo L2 fold 0 (SWA top-3) | 0.8835 | 0.8895 (ep 15, 14, 13) | +0.0060 |
| v1 fold 0 | 0.8368 | 0.8447 | +0.0079 |
| gold (veto) | 0.9144 | 0.9202 | +0.0058 |
| GPU-h | 2.49 | 2.51 | +1 % |

Criterio: pseudo ≥ 0.8866 y gold ≥ 0.8944 → se cumplen ambos. Δ ≈ 4.8× el ruido de semilla (0.0013).

**Qué aprendimos:** la augmentation espacial (espejo L/R anatómico p 0.5 + afín leve por estudio) es la primera mejora
de regularización sobre la receta Raptor: +0.0063 pseudo L2, y suben también v1 (+0.008) y gold (+0.006), así que no es
sesgo de la vara L2. Por etiqueta (pseudo) sube sobre todo ACL +0.021, MCL +0.024, menisco lateral +0.015, menisco
medial y contusión +0.007; bajan levemente PF OA −0.004 y fractura −0.005. La curva va por encima del padre desde la
época 7 y sigue subiendo en la 15 (0.8896 → 0.8899): con más épocas podría dar algo más. Coste nulo: 501 s/época como
el padre (aug en GPU ≈ 8 ms/batch).

**Siguiente paso sugerido:** pesos all-data (`swa_alldata.pt`, ruta arriba) al ensamble (inferencia con TTA espejo igual que exp031);
apilar en exp056 combo; probar sub045_1 con afín más fuerte (rot ±15°, escala 0.85–1.15) o 20 épocas, una variable.
