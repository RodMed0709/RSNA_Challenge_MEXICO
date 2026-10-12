# 01 — Etiquetas desde informes con varios LLM

Fecha: 2026-10-12. Fuentes: `RSNA_papers/02_llm_report_labels/RESUMEN.md` (27 PDFs), `RSNA_papers/analisis/A3_etiquetas_llm.md`,
`A2 §2.9`, `A5 C1 y ★3`. Números propios: verdicts vía A3/A5.

## 1. Cómo se construye hoy el consenso (A3 §1)

- Celda 2 de exp031 → exp054 → exp056: **media aritmética por celda** (estudio × hallazgo) de 5 fuentes: Raptor, L2, L3, L4, L5. Peso 1/5. NaN se enmascara en la BCE. Sin recalibración por fuente.
- Objetivo blando (BCE con target continuo); `pos_weight = clip((1−prev)/prev, 1, 10)`.
- L5 (`generar_L5.py`) = media de 3 prompts de gpt-4.1-mini (estado discreto mapeado a probabilidad + dos prompts de probabilidad 0–100 calibrada).
- **Vara de selección:** pseudo L2 (`v4_blend` > 0.5) en val fold 0. **L2 está a la vez en el objetivo y en la vara** → sesgo al cambiar la agregación (ya pasó en exp030; A5 supuestos).

## 2. Lo que ya medimos

| Exp | Cambio | Δ pseudo L2 f0 | Verdict |
|---|---|---|---|
| exp021 | etiquetas L2 | LB 0.812 → 0.836 | worked (A1) |
| exp031 | Raptor → media de 4 | +0.0155 (LB 0.930 → 0.939) | worked |
| exp052 | fuente muestreada por estudio y época | +0.0014 | didnt — **nulo por construcción** (ver §3) |
| exp054 | media de 4 → 5 (+L5) | +0.0060 | worked |
| sub056_2 | L5 → L5v2 (r≈0.998 con L5) | +0.0017 | didnt (ruido de semilla 0.0013) |

Lectura: **diversidad de fuente > relectura** (L5 nueva +0.006 vs L5v2 correlacionada +0.0017). Coincide con Plaza-del-Arco 2023, Flanders 2025 y Yamagishi 2026 (ensembles de LLM distintos ganan) [RESUMEN 02]. Hallazgo de **no aditividad**: aug espacial (exp045) y L5 (exp054) no suman en el combo exp056 (A5 C3) — hipótesis: ambas regularizan contra ruido de etiqueta [inferencia].

## 3. Lección matemática (exp052)

La BCE es lineal en y: `E_s[ℓ(z, y_s)] = ℓ(z, E_s[y_s])`. Muestrear una fuente da el **mismo gradiente esperado** que el consenso, más varianza (A2 §2.9). Las alternativas con fundamento modelan a cada anotador: Tanno 2019 (matriz de confusión por anotador), Raykar 2010 (EM), CROWDLAB (Goh 2022) [RESUMEN 02].

## 4. Datos del análisis CPU (A3 §3, pool no-gold 4349 × 12)

- Unanimidad binaria en 81 % de celdas; por hallazgo de 63 % (Effusion) a 96 % (Fracture).
- L2 y L4 casi idénticas (Pearson por hallazgo 0.87–0.985; κ 0.975): **la media cuenta dos veces a GPT-5.6-Sol** (L2 `v4_blend` incluye L4).
- Raptor–L2 en Synovitis: Pearson 0.479 (el resto 0.65–0.92). Fuentes "estrictas" (Raptor, L5) vs laxas en Effusion y PF OA.
- Dawid-Skene y MACE sin gold ya ajustados por hallazgo (A3 §3.4); mayores cambios de objetivo en Effusion (0.038), PF OA y Synovitis (0.022).

## 5. Ideas implementables (todas con SPEC de una variable)

| id | Idea | Cita | Implementación | Costo | Nota |
|---|---|---|---|---|---|
| LAB-1 | **Consenso dedup**: L2 y L4 como una sola fuente (pesos 1/4: Raptor, (L2+L4)/2, L3, L5) | Ratner 2017 (Snorkel, arXiv 1711.10160) | SPEC completa lista en A3 §4: padre exp054 (pseudo 0.8896, gold 0.9190); `worked` ≥ 0.8926 y gold ≥ 0.8990 | 2.5 GPU-h | 703 celdas cambian de lado de 0.5 |
| LAB-2 | **Dawid-Skene por hallazgo** con posterior templada (T fijada antes) | Dawid & Skene 1979 (doi:10.2307/2346806); Raykar 2010 | Fase 1 CPU contra dev-200 (exp010); fase 2 screen f0 solo si gana en dev | 0 + 2.5 | decide dev-200 y LB, no la pseudo (sesgada) |
| LAB-3 | **Cabezas por fuente** (4–5 lineales sobre el mismo `pooled`, BCE contra su fuente; inferencia = media de sigmoides) | Tanno 2019; Raykar 2010 | sustituto con fundamento de exp052 (A2 §2.9) | 2.5 | parámetros ×5 en la cabeza: despreciable |
| LAB-4 | **Limpieza/reponderación por OOF** (\|y − p_OOF\| ≥ 0.8 → peso bajo) o CROWDLAB con el modelo como anotador | Lumbar 2.º Kaggle (+1 % LB); Goh 2022 | `cleanlab.multiannotator` sobre OOF existentes | 2.5 | riesgo: quita positivos raros reales |
| LAB-5 | **6.ª fuente de otra familia** (Qwen-72B / Llama-70B / DeepSeek) | Plaza-del-Arco 2023; Flanders 2025; Yamagishi 2026 | mismo prompt que L5 | API + 2.5 | punto extra en la curva "AUC vs nº fuentes" (paper C1) |
| LAB-6 | Estado "incierto" explícito (3 clases → 0.5 o máscara) | CheXpert (Irvin 2019); MAPLEZ 2024 | re-prompt | API + 2.5 | |
| LAB-7 | Pre-entrenar con consenso, afinar solo con celdas unánimes | CheXbert (Smit 2020); Di Noto 2022 | fine-tune corto | ~1 | no usa gold |
| LAB-8 | Pérdida no lineal en y (ponderar celda por acuerdo entre fuentes) | Peterson 2019 (soft labels) [inferencia de aplicación] | peso = 1 − var entre fuentes | 2.5 | sí cambia el gradiente esperado (a diferencia de exp052) |

## 6. Reglas

- Gold (58) **nunca** estima pesos de fuentes (regla 6). Para pesos supervisados, usar el dev-200 de exp010.
- Cambiar la agregación sesga la pseudo L2 → declarar en la SPEC que decide el dev-200 + LB (A5 supuestos).
- Para el paper (C1): falta exactitud de cada fuente y del consenso vs humanos (dev-200), κ por hallazgo, y la curva AUC de imagen vs nº y decorrelación de fuentes (A5 A.2).
