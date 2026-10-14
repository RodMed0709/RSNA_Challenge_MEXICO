# 04 — Robustez, ensambles, weight averaging y lectura del LB

Fecha: 2026-10-12. Fuentes: `RSNA_papers/04_robustness_aug_ensemble/RESUMEN.md`, `analisis/A4_interno_vs_LB.md`, `A6_revision_colas.md`, A2 §2.12 y §2.14.

## 1. Ruido del LB (A4 §2)

- Test ≈ 1300 estudios; **público 30 % ≈ 390**, privado ≈ 910. Métrica: media de AUC de 12 hallazgos, redondeo a 3 decimales.
- SD de la media de 12 AUC ≈ **0.004 en público**, ≈ 0.003 en privado (simulación binormal; cota inferior).
- SD de la **diferencia pareada** en público: 0.0046 (ρ 0.5), 0.0032 (ρ 0.8), 0.0022 (ρ 0.9), 0.0016 (ρ 0.95).
- **Regla: Δ público < 0.003 = empate.** Maldición del ganador: elegir el máximo de K=5 envíos parecidos infla ≈ +0.0023; K=10 ≈ +0.003.
- Offsets en régimen Raptor: **LB ≈ gold + 0.020–0.025** y **LB ≈ pseudo L2 + 0.055–0.065**. Ordenan bien cuando Δ > ~0.005.
- Divergencia real única: v1 vs LB (ρ −0.40), por **definición de etiqueta** distinta (Kaggle usa umbrales estrictos).
- Literatura: Roelofs 2019 (NeurIPS; >100 competiciones, poco sobreajuste adaptativo, shakeup = ruido muestral); Maier-Hein 2018 (Nat Commun, doi:10.1038/s41467-018-07619-7; rankings frágiles, reportar bootstrap); Varoquaux & Cheplygina 2022 (npj Digit Med, doi:10.1038/s41746-022-00592-y).

## 2. Pesos de ensamble (A4 §4)

- Modelo de juguete binormal: con bloque propio ≈ 0.939 y ρ 0.90–0.95 respecto a blend-mine (0.954), **peso útil ≤ 0.10–0.15, ganancia ≈ 0**. 0.35 era demasiado (medido: 0.952 y 0.950).
- El bloque justifica 0.20–0.25 solo si su LB solo ≈ 0.945.
- **El valor de un bloque crece con la diversidad, no con su AUC**: bajar ρ de 0.95 a 0.85 vale más que +0.005 propio (Fort 2019, arXiv 1912.02757; DiWA, Ramé 2022, arXiv 2205.09739).
- Reglas a priori: media de **rangos** con pesos fijos; **no optimizar pesos en OOF** (Müller 2022, IEEE Access, doi:10.1109/ACCESS.2022.3182399: pooling simple ≥ aprendido); pesos iguales entre folds/semillas.
- ρ entre nuestro bloque y nartaa está **inferida** de 2 envíos; medirla con OOF exp045 vs Raptor nartaa sobre train (A4 pendiente).

## 3. Estrategia de envíos y finales (A4 §3, A6)

- 2 finales: **A (AUC)** = blend-mine + bloque propio 0.15 si ≥ 0.954; **B (eficiencia)** = el de menor score medido con runtime real repetido (hoy Ch 0.945 ~8 min; candidato exp057).
- Fórmula de eficiencia ≈ −AUC/0.46 + t/32400 → **0.001 AUC ≈ 70 s** (benchmark ≈ 0.5, deducido).
- Cada familia candidata necesita ≥ 2 envíos coherentes. No barrer pesos en LB (solo 0.15 y 0.25 a priori).
- Regla del 14-oct endurecida (A6 §3): w25 solo si exp045-5f-solo ≥ 0.945 **y** w15 ≥ 0.953; si no, sustituir por ens-exp045-5f-054-w15.
- exp042b (pesos nartaa con supervisión OAI): enviar sí; **no elegible como Final** sin verdict de exp044 y Δ ≥ +0.003 (A6 §4).

## 4. Ideas implementables

| id | Idea | Cita | Costo | Riesgo | Sirve a |
|---|---|---|---|---|---|
| ENS-1 | **Fold soup** de exp045 (uniforme; greedy si falla). Antes: barrera de pérdida al interpolar 2 folds sobre pseudo | Wortsman 2022 Model Soups (arXiv 2203.05482); DiWA; FissionFusion (Sanjeev 2024, arXiv 2403.13341, advierte soups subóptimos en médico) | ≈0.5 GPU-h (inferencia) | medio | eficiencia ★, paper C2 |
| ENS-2 | **Ensamble de rangos exp045 5f × exp054 5f**, 0.5/0.5 fijado ahora (si no suman como modelo, deberían como ensamble) | Fort 2019; DiWA | ≈0.5 + 1 envío | bajo | LB; teacher de EFF-1 |
| ENS-3 | **SWA real**: `update_bn` sobre `swa_f0.pt` de exp045 (hoy se promedian buffers BN sin recalcular) | Izmailov 2018 (arXiv 1803.05407): *"run one additional pass over the data … to compute the running mean"* | 0.2 GPU-h | bajo | corrige pesos ya enviados si cambia la AUC |
| ENS-4 | **Cola de LR constante** (16 ép OneCycle + 4 ép a ~0.25× LR, promedio por época + `update_bn`) | Izmailov 2018 | 3.2 | bajo | da además un padre largo |
| ENS-5 | **SWAD** (promedio denso con ventana por pérdida de pseudo-val, nunca gold) | Cha 2021 (arXiv 2102.08604) | 2.5 | bajo-medio | LB marginal |
| ENS-6 | **WiSE-FT**: interpolar pesos finales con checkpoint temprano, α ∈ {0.5, 0.7, 0.9} elegido en pseudo | Wortsman 2022 (doi:10.1109/CVPR52688.2022.00780) | ≈0 + 1 envío | medio | LB |
| ENS-7 | **Semillas** de exp045 all-data (2–3) | Fort 2019; medido: 2 semillas exp031 +0.0042 pseudo (0.8878 vs 0.8836) | 3 por semilla | bajo | LB (no eficiencia); sirve a R3 |
| ENS-8 | **Auditoría TTA**: con/sin espejo por hallazgo sobre `swa_f0.pt` de exp045 | Shanmugam 2021 (doi:10.1109/ICCV48922.2021.00125): la TTA voltea aciertos | 0.2 | bajo | crítico para exp057 (sin TTA) |
| ENS-9 | **BN-adapt / TENT** en test | Wang 2021 TENT (arXiv 2006.10726); Karani 2021 (doi:10.1016/j.media.2020.101907) | inferencia | **alto** (batch por estudio, ruidoso) | descartar antes del 22-oct |
| ENS-10 | **Harness de robustez exp040** (ruido, sitio croata) sobre exp046/sub046_1/exp047 | Varoquaux 2022 | 0.6 | nulo | decide prior de augs de intensidad |
| ENS-11 | **Calibración** (ECE, Brier; temperature scaling). ECE medida en exp040 = 0.20 | Ovadia 2019 (arXiv 1906.02530); Minderer 2021 (arXiv 2106.07998) | CPU | bajo | paper R7 |
| ENS-12 | Medir **ρ** entre OOF exp045 y Raptor nartaa; si < 0.85, subir peso a priori a 0.25 | A4 pendiente | CPU | bajo | LB |
