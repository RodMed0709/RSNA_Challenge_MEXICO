# Stress Test — cobertura y resultados

Estado (22-oct-2026): auditoría de cobertura hecha (21-oct) y benchmark modesto `exp091_stress_v1` ejecutado (solo inferencia).
Spec que rige: `SPEC_v1_modesto.md`. Código, SPEC y `metrics.json`: rama `exp091_stress_v1` (PR #38). Los números salen de
`results/exp091_stress_v1/metrics.json`; no hay IDs, predicciones por estudio ni etiquetas en este repo.

## Auditoría histórica (21-oct, lectura de Kaggle en vivo y del volumen)

| Métrica | Resultado |
|---|---|
| Submissions con score público | 48 (`privateScore` vacío) |
| Familias independientes | 3 a 5 (solo DINOv2-s es claramente independiente; nartaa/Raptor y heliosli comparten pesos base) |
| Envíos con OOF limpio y LB del mismo artefacto | 9 |
| Envíos solo con gold-58 (all-data, sin OOF de train) | 11 (5 sin reserva; 6 con maestro que vio el fold) |
| No evaluables | 28 (tanques por entrenar con 100 % de train, pesos externos heliosli/nartaa, exp069/exp071 con init que vio todo train) |
| Familias con OOF y LB | 2 (DINOv2-S, Raptor) — la SPEC original pedía 8 |
| OOD verdadero (`true_group_holdout`) | no viable: `split_v1` mezcla fabricante/campo/idioma y el freeze impide entrenar un `split_v2` |

Dictamen: transformaciones de estrés viables; casos difíciles provisionales (etiquetas pseudo, misma familia); OOD verdadero no
viable; transferencia al LB solo descriptiva.

## Resultados de `exp091_stress_v1`

- Cobertura 45/45 (Panel S 17, Panel R 18, Panel F 10 folds). Costo 11.58 USD, 11.2 GPU-h.
- Gate de reproducción exp029: AUC 0.868175 vs 0.868178 esperado, Δ −2.2e-6, **pasa** (|Δ| < 0.002).
- H1 (AUC limpia vs robustez plausible, 18 modelos): rho −0.172, IC95 [−0.568, 0.301]; sin relación clara.
- H2 (aug vs padre): 2 de 6 confirmadas. exp045 y exp046 vs exp031 mejoran `R_sev` +0.0036 y +0.0044 (IC excluye 0) y la AUC limpia
  (+0.0063, +0.0021). exp062/exp063 no confirman.
- H3 (10 % más difícil, n=435, misma familia, provisional): ΔBCE exp054−exp045 = −0.0083 [−0.0153, −0.0028].
- Correlación con LB: no calculada (< 8 familias); tabla de 6 scores en `metrics.json`.

### Ranking de estabilidad (Panel S; MAD media de |p_cond − p_ref|, menor = más estable; val fold 0, `in_sample`, descriptivo)

| Modelo | S_plaus | S_sev |
|---|---|---|
| exp090_all | 0.0234 | 0.0583 |
| exp088_ens4 (tank-v6) | 0.0239 | 0.0495 |
| exp074_all | 0.0240 | 0.0585 |
| exp088 semillas | 0.0250–0.0257 | 0.0510–0.0518 |
| exp080_all | 0.0269 | 0.0637 |
| exp089_ens2 (tank-v6soft) | 0.0284 | 0.0573 |
| exp085_ens4 (tank-seeds) | 0.0290 | 0.0588 |
| sub075_1_all | 0.0314 | 0.0657 |

Regla de subida a Kaggle (SPEC_v1 §5b): reducción ≥ 20 % en `S_plaus` y `S_sev` vs tank-seeds. El mejor (exp088_ens4) reduce
17.6 % y 15.8 %; tank-v6 ya tiene LB 0.957, igual que tank-seeds. **Ningún candidato cumple; no se envió nada.**

## Límites
Panel S es descriptivo (modelos entrenados con esos estudios); el stress no demuestra mejora en private LB ni en hospitales externos.

## Transferibilidad (22-oct, análisis post-hoc, descriptivo)

Pregunta: ¿alguna métrica de estrés predice el LB? **Respuesta: no hay evidencia; nada queda validado como transferible.**

| Modelo | S_plaus (menor = más estable) | LB público |
|---|---|---|
| exp090 | 0.0234 | 0.947 |
| tank-v6 (exp088) | 0.0239 | 0.957 |
| exp080 | 0.0269 | 0.949 |
| tank-v6soft (exp089) | 0.0284 | 0.957 |
| tank-seeds (exp085) | 0.0290 | 0.957 |

- El más estable (exp090) tiene el LB más bajo; los tres tanques difieren hasta 17 % en estabilidad y empatan en 0.957.
- Spearman informal (n=5, calculado a mano, no está en `metrics.json`): ≈ +0.67 (más estable ↔ peor LB). No significativo y
  confundido (exp090/exp080 no son tanques). Con 3 a 5 familias independientes no se calculó correlación formal (SPEC pedía ≥ 8).
- H1: AUC limpia y robustez son ejes distintos (rho −0.17, IC [−0.57, 0.30]).
- Lo más sólido: exp045/exp046 mejoran `R_sev` sin perder AUC (2 de 6 augmentations, solo fold 0).
- Por qué no se puede concluir más: el LB público no resuelve diferencias de este tamaño; las perturbaciones no equivalen al
  shift de hospital/población del privado; tanques y all-data son `in_sample`; `split_v1` no permite OOD verdadero.
- Métrica con historial: OOF pseudo AUC (regla Δ ≥ +0.005). La robustez, a lo sumo, sirve de desempate; no está probada.
