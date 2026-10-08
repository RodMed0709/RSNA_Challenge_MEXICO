# exp019_poda_slots

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp012_entrada_v2 |
| Fecha | 2026-10-08 |
| Rama | exp019_poda_slots |
| Estado | didnt |

Idea: solo inferencia (no se entrena). Con el checkpoint de exp012 fold 0, medir cuánta AUC se pierde y cuánto
tiempo de Kaggle se gana si en inferencia se usan menos slots. Insumo directo para el track de eficiencia y el paper.

Contexto: `docs/pasos/costo_inferencia.md` (rama `exp013_cabeza_hallazgo`). Para ~1 300 estudios con 4 vCPU (ideal),
el preprocesado v2 cuesta 281 s con 6 slots, 182 s con slots 1–3 y 78 s con 1 slot; domina la normalización por
percentiles, no la decodificación.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** el modelo de exp012 concentra casi toda la señal en pocos slots (sobre todo los de líquido con
supresión grasa, 1–3). Quitando slots en inferencia, el costo total en Kaggle baja ≥ 35 % perdiendo ≤ 0.01 de pseudo
AUC macro.

**Única variable cambiada:** qué slots ve el modelo en inferencia. Los slots fuera del subconjunto se enmascaran con
`key_padding_mask` (exactamente igual que una serie que falta). Pesos, cache, cortes, tamaño, split: idénticos a exp012.

| Parámetro | Padre (exp012) | Este experimento |
|---|---|---|
| Entrenamiento | 8 épocas fold 0 | ninguno (se carga `best_f0.pt` de exp012) |
| Slots en inferencia | los 6 | subconjuntos (abajo) |

**Subconjuntos evaluados** (fold 0 pseudo = validación de exp012; gold = 58 estudios, fold −1):
1. Los 6 subconjuntos de 1 slot.
2. Eliminación hacia atrás greedy desde los 6: en cada paso se quita el slot cuya ausencia menos baja la pseudo AUC
   macro, hasta 1 slot (6 subconjuntos en el camino, el primero son los 6 slots).
3. Slots 1–3.
4. Por hallazgo (insumo para poda por hallazgo, no decide el verdict): la mejor combinación de ≤ 2 slots según la
   pseudo AUC de ese hallazgo.

**Estudio sin ningún slot del subconjunto:** se usa el slot presente de menor número (misma regla que el escenario d de
`costo_inferencia.md`); su costo se cuenta. Así todo estudio recibe predicción.

**Costo estimado en Kaggle** (test oculto ≈ 1 300 estudios, 4 vCPU ideal):
- Preprocesado: `281 s × costo_rel(S)`, con `costo_rel(S) = media_estudios Σ_{slot usado} t_slot / (lo mismo con 6 slots)`
  y `t_slot` = segundos v2 por serie (1 hilo) de la tabla por slot de `costo_inferencia.md`
  (0.325, 0.191, 0.254, 0.314, 0.245, 0.181 s para slots 1–6). La media es sobre los estudios evaluados (fold 0).
  Se imprime la comprobación contra lo medido para slots 1–3 (182 s).
- Modelo: tiempo GPU por serie medido en el notebook (backbone + pooling de cortes, `cuda.synchronize`, mediana)
  × series usadas por estudio × 1 300. GPU del pod (compartida), no T4: se reporta como tal.
- Total = preprocesado + modelo (secuencial, cota conservadora). No incluye carga de pesos ni lectura de disco.

**Score de eficiencia** (menor es mejor): `AUC / (0.5 − MAX_AUC) + RuntimeSeconds / 32 400`, con `MAX_AUC = 0.965`
(supuesto; da 1 min ≈ 0.00086 AUC) y AUC = pseudo AUC macro del subconjunto. Se reporta también con gold.

**Elección del subconjunto final (solo pseudo):** entre los subconjuntos de los grupos 1–3 con costo ≤ 65 % del de 6
slots y pseudo ≥ pseudo(6 slots) − 0.01, el de menor score de eficiencia (pseudo). Gold no participa en la elección.

**Criterios (no cambian después de correr):**
- `worked`: existe el subconjunto elegido (costo ≤ 65 % del de 6 slots y pseudo ≥ 6 slots − 0.01) **y** su gold no cae
  más de 0.02 contra gold con 6 slots. Si el elegido cae > 0.02 en gold, es `didnt` (no se prueba el siguiente: gold no elige).
- `didnt`: ningún subconjunto cumple.
- `killed`: solo si el pipeline falla (no hay kill por AUC: es inferencia).

**Efecto esperado:** slots 1–3 o un subconjunto de 2–3 slots cumple; ahorro ~35–45 % del tiempo de Kaggle.

**Presupuesto:** ~0.3 GPU-h (una pasada de extracción de features por slot sobre ~870 + 58 estudios; los subconjuntos se
evalúan sobre features cacheadas). GPU compartida con otros runs.

**Riesgos:**
- Distribution shift: el modelo se entrenó siempre con los slots disponibles; enmascarar slots presentes puede degradar
  más de lo que degradaría un modelo entrenado con poda (eso sería otro experimento: entrenar con dropout de slots).
- Costos de preprocesado extrapolados de 60 estudios de train con ±30 % de ruido; costo del modelo medido en GPU
  compartida. Los costos relativos (65 %) son más fiables que los absolutos.
- Se elige y se evalúa sobre la misma validación de fold 0 (sesgo optimista pequeño; ~870 estudios, ≤ 40 candidatos).

## 2. PLAN

- Notebook nuevo `exp019_poda_slots.ipynb` con las celdas fijas; dataset y modelo copiados de exp012 (sin cambios).
- Celda 0: `CKPT`, `FOLD = 0`, `SMOKE`, constantes de costo (`T_SLOT_V2`, `PRE_6SLOTS_S = 281`, `N_TEST = 1300`, `MAX_AUC`).
- Celda 3: carga `best_f0.pt` de exp012.
- Celda 4 (no hay entrenamiento): extracción de features por slot (backbone + slice_pool + slot_embedding) una vez por
  estudio, con tiempo GPU por serie; evaluación de subconjuntos solo con `series_pool` + cabeza y la máscara. Equivale a
  correr el modelo completo con esa máscara (los slots no interactúan antes de `series_pool`); se comprueba numéricamente
  en el smoke contra el forward completo.
- Celda 5: tabla de subconjuntos, figura Pareto costo vs AUC en `results/exp019_poda_slots/`, elección y decisión.
- Celda 6: `metrics.json` (schema de los otros + tabla de subconjuntos).
- Smoke con `SMOKE = True` antes del run real.

## 3. VERDICT (números de `results/exp019_poda_slots/metrics.json`)

**Resultado:** didnt · **Fase alcanzada:** screen (subconjunto elegido = los 6 slots)

| Métrica | 6 slots (padre) | Subconjunto elegido | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | 0.850 | 0.850 (mejor candidato barato: s1+s3+s5 0.824) | 0 (−0.026) |
| gold AUC (58) | 0.784 | 0.784 (s1+s3+s5: 0.785) | 0 (+0.001) |
| costo Kaggle estimado (s) | 344 (281 prepro + 63 modelo) | 344 (s1+s3+s5: 211, 61 %) | 0 (−133) |
| score de eficiencia (pseudo) | −1.817 | −1.817 (s1+s3+s5: −1.775) | 0 (peor) |

**Qué aprendimos (2-4 líneas):** podar slots en inferencia sin reentrenar no sirve: el modelo entrenado con 6 slots usa todos.
Quitar uno cuesta 0.009–0.015 de pseudo (sin s2: −0.009 pseudo, −0.017 gold); 3 slots (61 % del costo) cuestan −0.026. Por
hallazgo, ≤2 slots pierden 0.02–0.08 salvo Medial OA, Contusion y Fracture (≤0.006). Y lo que cambia la lectura de eficiencia:
con runtime en minutos, ahorrar todo el preprocesado (~280 s) vale solo ~0.004 AUC en la métrica; la poda solo paga si la AUC
cae menos que eso. 63 subconjuntos evaluados, 0.18 GPU-h.

**Siguiente paso sugerido:** entrenar con slot dropout (apagar slots al azar) para que la red tolere la poda y medir otra vez
la curva costo–AUC; la poda por hallazgo del paper necesita ese entrenamiento. Para Kaggle, priorizar AUC (etiquetas) sobre runtime.
