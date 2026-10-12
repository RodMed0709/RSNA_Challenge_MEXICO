# exp047_raptor_slot_dropout

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | exp047_raptor_slot_dropout |
| Estado | didnt |

Idea (paso 7, track T2 estructura): la receta Raptor casi no tiene augmentation; en train solo baraja ventanas.
exp040 (robustez de exp029) mide −0.028 sin sagital y −0.007 con 50 % de cortes; exp043 mostró que un modelo de
5 slots inferido con 3 conserva gold. Enseñar al modelo a decidir con slots ausentes debería regularizarlo (y habilita
la poda de slots para el track de eficiencia).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** el dropout de slot completo + jitter de centros en train sube la pseudo AUC L2 en fold 0 en ≥ +0.003
contra exp031 sin que gold caiga más de 0.02.

**Única variable (augmentation estructural, solo `StudyWindows` con `train=True`):**
- El corpus 96×384 guarda los slots en posiciones fijas (`build_meta.json["slots"]`, en orden): Sag fluid 0–25,
  Sag no-fluid 26–47, Cor fluid 48–65, Cor 66–77, Ax 78–95; `masks.npy` (4407×96) marca los cortes válidos.
  Slot presente = ≥ 1 corte válido (presencia medida: 1.00 / 0.973 / 1.00 / 0.745 / 1.00; 1239 estudios con 4 slots,
  3166 con 5, 2 con 3).
- Por estudio y muestra: cada slot presente se descarta con p **0.15**; nunca quedan menos de **2** vivos (si pasa, se
  reviven descartados al azar). Se quitan los centros cuya ventana (c−1, c, c+1) toque un slot descartado; las K = 12
  ventanas se sacan del resto con el mismo muestreo del padre (rellena remuestreando de los slots vivos).
- Jitter: cada centro muestreado se mueve ±1 corte (uniforme {−1, 0, +1}); si cae fuera de los centros vivos, se queda.
- Validación, gold e inferencia (K_EVAL 24 linspace) idénticas al padre.

**Todo lo demás idéntico al padre:** etiquetas de consenso, corpus `/root/corpus96_384`, recorte 320, `RaptorClassifier`,
16 épocas, bs 8 × K 12, AdamW / OneCycle / LRs, bf16, semilla 42, `split_v1_seed42` fold 0, selección por pseudo L2
(L2 binarizada > 0.5), SWA top-3.

**Métrica principal:** pseudo L2 fold 0 de la mejor época (como el padre). SWA top-3 se reporta.

**Criterios (fijos, no cambian después de correr):** padre exp031 pseudo L2 0.8836, gold 0.9144.
- `worked` (promete): pseudo L2 ≥ **0.8866** (+0.003) **y** gold ≥ 0.8944.
- `killed`: pseudo L2 ≤ **0.8786** (−0.005) **o** gold < 0.8944.
- `didnt`: cualquier otro caso.
La `decision_auto` usa ±0.005; el verdict se aplica a mano con estos criterios (ruido de semilla medido 0.0013).

**Presupuesto:** fold 0 ≈ 2.5 GPU-h en rsna-d2 (el muestreo no lee más cortes: mismo coste). Si promete: sub de
intensidad (p 0.3) y all-data con las 3 mejores épocas del fold 0.

**Riesgos:** p 0.15 por slot deja ~56 % de las muestras sin algún slot: puede restar señal en hallazgos que dependen de
un solo plano y bajar la pseudo con 16 épocas; el jitter ±1 es casi neutro porque el padre ya muestrea centros al azar
(se declara como parte de la misma variable estructural, según el plan del paso 7).

## 2. PLAN

- Notebook `exp047_raptor_slot_dropout.ipynb` = copia del de exp031; cambios: celda 0 (`EXP`, `PADRE`, `SLOT_DROP_P`,
  `MIN_SLOTS`, `CENTER_JITTER`, `NOTES`), celda 3 (`SLOT_OF`, `StudyWindows._slot_dropout`, jitter en `_centers` solo
  en train, comprobación del muestreo sin leer imágenes).
- Smoke: `SMOKE=True` → `metrics_smoke.json`.
- Run real: `FOLDS=[0]`, `SMOKE=False`, en rsna-d2 (tras sub031_3).

## 3. VERDICT (números de `results/exp047_raptor_slot_dropout/metrics.json`)

**Resultado:** `didnt` · **Fase alcanzada:** screen fold 0 (rsna-d2, 2026-10-12 02:45–05:16 UTC)

| Métrica | Padre (exp031) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8836 (ep 15) | 0.8844 (ep 14) | +0.0007 |
| pseudo L2 fold 0 (SWA top-3) | 0.8835 | 0.8849 (ep 14, 15, 11) | +0.0014 |
| v1 fold 0 | 0.8368 | 0.8398 | +0.0030 |
| gold (veto) | 0.9144 | 0.9139 (SWA 0.9147) | −0.0005 |
| GPU-h | 2.49 | 2.50 | +0.01 |

Criterio: worked exige ≥ 0.8866; killed ≤ 0.8786 o gold < 0.8944 → `didnt` (Δ dentro del ruido de semilla 0.0013).
Δ pseudo por hallazgo (mejor época): MCL +0.014, Medial Meniscus +0.008, Contusion +0.008, Effusion +0.005,
Lateral Meniscus +0.005; PF OA −0.011, Baker's −0.007, Fracture −0.007, Lateral OA −0.005.

**Qué aprendimos:** quitar slots completos en train (p 0.15, ~66 % de las bolsas pierden ≥ 1 slot presente contra
~29 % en el padre) no daña ni ayuda la pseudo L2 con 16 épocas: la curva es casi idéntica a la del padre época a
época (ep 11–15: 0.8837–0.8844 contra 0.8811–0.8836). Coste igual. No es una augmentation que suba AUC en
distribución; su posible valor es robustez a slots ausentes (inferencia con 3 slots para el track de eficiencia),
que este screen no mide.

**Siguiente paso sugerido:** no hace falta sub de intensidad ni all-data por AUC. Si el track de eficiencia quiere
podar slots, evaluar (solo inferencia, ~0.2 GPU-h) `best_f0.pt` de exp047 contra el de exp031 en val fold 0 con
3 slots / sin Cor no-fluid / sin sagital, como exp040/exp043; si exp047 pierde menos, usarlo como base del modelo
podado.
