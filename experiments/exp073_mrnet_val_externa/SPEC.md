# exp073_mrnet_val_externa

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial (checkpoint all-data swa_alldata.pt, SWA 13-15) |
| Fecha | 2026-10-14 |
| Rama | exp073_mrnet_val_externa |
| Estado | spec |

Idea: V3a del plan de preentreno externo: validación externa (para el paper) del modelo de la receta exp045 en MRNet
(Stanford), sin entrenar nada. V3b (MRNet como tarea auxiliar/preentreno) sería otro experimento y solo si este muestra que el
dominio encaja; nunca se mezclan en el mismo modelo evaluado.

## 1. SPEC (antes de tocar el notebook)

**Pregunta / hipótesis:** el modelo all-data de exp045 (CoAtNet-2 Raptor, 12 hallazgos RSNA, nunca vio MRNet) generaliza a
otro hospital, otro protocolo y otras etiquetas: su AUC cero-shot en MRNet para anormal / LCA / menisco, con el mapeo fijo de
abajo, es ≥ 0.85 / 0.85 / 0.75 en el conjunto de validación público de MRNet (120 exámenes).

**Única variable (frente a exp045):** el conjunto evaluado (MRNet en vez de pseudo fold 0 / gold RSNA). Mismo checkpoint
(`swa_alldata.pt` de exp045, SWA épocas 13–15), misma inferencia (K_EVAL 24, recorte central 320, normalización ImageNet,
TTA espejo nartaa; también sin TTA), sin entrenamiento, sin ajuste de umbral ni calibración.

| Parámetro o celda | Padre (exp045) | Este experimento |
|---|---|---|
| Datos evaluados (celda 2) | RSNA (pseudo fold 0, gold-58) | MRNet v1.0 train (1130) + valid (120), convertido a nuestro corpus 96×384 |
| Celda 4 | entrenamiento | solo inferencia con `swa_alldata.pt` (`MODELS` admite más checkpoints, p. ej. exp071, en la misma corrida) |

**Conversión MRNet → corpus (fija antes de ver resultados):** slot 0 (Sag fluido, 26) ← sagital T2 FS; slot 1 (Sag no fluido)
vacío; slot 2 (Cor fluido) vacío; slot 3 (Cor no fluido, 12) ← coronal T1; slot 4 (Ax, 18) ← axial PD FS. Por serie: cortes
por linspace en el 2–98 % de la serie, p2/p98 → uint8, 256 → 384 (MRNet no trae tamaño de píxel; se asume campo ≈ 140 mm como
nuestro recorte físico). Se revisa la orientación con `orientation_check.png` (MRNet frente a RSNA) antes de la corrida real;
si hay que corregirla, se fija con `FLIP_FIX` antes de calcular ninguna AUC.

**Mapeo de etiquetas (fijo):** LCA ← p(ACL); menisco ← max(p(Medial Meniscus), p(Lateral Meniscus)); anormal ← max de las 12
(segunda lectura: noisy-OR 1 − ∏(1 − p)).

**Métrica principal:** AUC con IC 95 % bootstrap (2000, semilla 42) en MRNet valid (120, el conjunto de comparación estándar;
el test oficial de MRNet es privado). Se reportan también train (1130) y todo (1250). Referencia del paper de MRNet (test, modelo
entrenado en MRNet): anormal 0.937, LCA 0.965, menisco 0.847 (solo contexto; no es comparable 1:1).

**Criterios (no cambian después de correr):**
- `worked` (el dominio encaja; habilita proponer V3b): en valid, AUC anormal ≥ 0.85 **y** LCA ≥ 0.85 **y** menisco ≥ 0.75
  (lectura principal: max, con TTA).
- `didnt`: alguno por debajo (resultado igualmente reportable en el paper como límite de generalización).
- `killed`: solo si la conversión falla (orientación/slots irrecuperables) o AUC LCA < 0.60 (indica error de canal, no de dominio).
- Gold y pseudo RSNA no se usan ni se tocan.

**Presupuesto:** construir el corpus MRNet ≈ 10 min CPU; inferencia ≈ 1250 × 24 ventanas × 2 (TTA) ≈ 15 min GPU (RTX 5090,
pod `rsna-ext1`, en paralelo con otro entrenamiento) ≈ $0.3.

**Licencia / legal:** copia de terceros en Kaggle (`cjinny/mrnet-v1`, 5.7 GB zip, extraído en `/workspace/data/external/mrnet/`,
7.3 GB) del MRNet Dataset de Stanford AIMI (Research Use Agreement, no comercial, sin redistribución). Para el paper Rodrigo
firmará el RUA de Stanford AIMI y se descargará/citará la fuente oficial.

**Fuentes:** Bien N, Rajpurkar P, Ball RL, et al. Deep-learning-assisted diagnosis for knee magnetic resonance imaging:
development and retrospective validation of MRNet. PLOS Med 2018;15(11):e1002699. doi:10.1371/journal.pmed.1002699.
https://stanfordmlgroup.github.io/competitions/mrnet/

**Riesgos:** slots vacíos (1 y 2) fuera de distribución para el modelo; MRNet "anormal" incluye hallazgos fuera de nuestros 12;
orientación/campo de visión asumidos; población distinta (Stanford, 2001–2012).

## 2. PLAN

- Celda 0 (parámetros): `MRNET_DIR`, `MRNET_LOCAL`, `MODELS`, `TTA_MIRROR`, `N_BOOT`, `FLIP_FIX`, `NOTES`.
- Celda 2 (datos): MRNet → vols/masks 96×384 en `/root/mrnet96_384` (lógica de `build_corpus96.py`), `orientation_check.png`.
- Celda 3 (modelo): igual que exp045 (sin aug de train).
- Celda 4: inferencia por checkpoint (sin entrenamiento), `preds_<modelo>.csv`.
- Celda 5: AUC + IC bootstrap por tarea y conjunto; AUC de cada uno de los 12 hallazgos contra cada etiqueta MRNet (informativo).
- Celda 6: `metrics.json` con bloque `mrnet`.
- Smoke (`SMOKE = True`, 16 exámenes por conjunto) → revisar orientación → corrida completa.

## 3. VERDICT (números de `results/exp073_mrnet_val_externa/metrics.json`)

**Resultado:** worked / didnt / killed · **Fase alcanzada:** screen / 5 folds

| Métrica | Padre | Este | Δ |
|---|---|---|---|
| pseudo AUC fold 0 | | | |
| gold AUC (screen: fold 0 contra fold 0 del padre) | | | |
| OOF pseudo 5 folds | | | |
| GPU-h | | | |

**Qué aprendimos (2-4 líneas):**

**Siguiente paso sugerido:**
