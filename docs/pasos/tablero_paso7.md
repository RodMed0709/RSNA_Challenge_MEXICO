# Tablero paso 7 — seguimiento RSNA Knee

Actualizado: 2026-10-12T08:05Z. Espejo del dashboard (artifact privado). Números de `metrics.json` / `history_f0.csv` de cada experimento, de `docs/pasos/envios_kaggle.md` y de la CLI de Kaggle. Lo que no tiene número todavía va como corriendo / en cola.

## Lo esencial

- Mejor LB público (base pública reproducida): **0.954** (exp023: velci-kcd25-20 tal cual (base 0.954), envío 57081259).
- Mejor modelo propio en LB: **0.939** (exp031 all-data (consenso) + TTA, envío 57091516).
- Mejor eficiencia en LB: **0.945** con Ch (nartaa 224 half, sin TTA; ~8 min hasta score, envío 57089745).
- Saldo RunPod: **US$35.60** (2026-10-12T08:01:52Z).
- Padre del paso 7: exp031, pseudo L2 0.8836; promete si Δ ≥ +0.003 (ruido de semilla 0.0013).

## Experimentos (desde exp021)

| exp | track | padre | variable | pseudo L2 (mejor ép.) | pseudo L2 SWA | gold | Δ pseudo | Δ gold | GPU-h | LB | estado | pod | nota |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp021 etiquetas_L2 | Propia DINOv2 | exp002 | etiquetas train v1 → L2 | 0.8170 | — | 0.8118 | — | — | 3.17 | 0.836 | worked | — | primer movimiento real del LB (+0.024 vs exp002) |
| exp022 entropia_suave | Propia DINOv2 | exp013 | penalización de entropía λ 0.005 | — | — | 0.7999 | — | — | 2.91 | — | didnt | — | pseudo v1 0.835; la atención no se concentra |
| exp035 etiquetas_combinadas | Propia DINOv2 | exp021 | etiquetas L2 → promedio L2+L3+L4 | 0.8248 | — | 0.8059 | +0.0079 | -0.0059 | 1.32 | — | didnt | rsna-e | verdict 3c9fabc: no concluyente (+0.008 pseudo, gold −0.006, sin envío a LB) |
| exp036 ref_v2_L2 | Propia DINOv2 | exp021 | entrada v2 + L2 + 12 épocas | 0.8524 | — | 0.8416 | +0.0354 | +0.0298 | 3.50 | — | worked | rsna-e | verdict e762ac8. Mejor DINOv2 propio, pero lejos de Raptor (0.88) |
| exp023 repro_publicos | T1 Kaggle | — | reproducir 3 notebooks públicos tal cual | — | — | — | — | — | — | 0.954 | worked | Kaggle | velci y blend-mine 0.954, nartaa mirror 0.949 |
| exp025 perfil_raptor | Eficiencia | — | perfilado por etapa de la inferencia Raptor | — | — | — | — | — | — | — | worked | rsna-d | lectura+decode+norm 41 %; en Kaggle manda el forward T4 |
| exp026 t4_variantes | Eficiencia | exp025 | variantes del forward en T4 (sin TTA, channels_last) | — | — | — | — | — | — | 0.948 | worked | Kaggle | sin TTA −50 % de tiempo con Δgold −0.001 |
| exp027 t4_rapido | Eficiencia | exp026 | half, 224, compile en T4 | — | — | — | — | — | — | — | didnt | Kaggle | 224: −48 % tiempo pero veto gold −0.007 |
| exp041 t4_sin_slots | Eficiencia | exp027 | sin slots Cor-otro/Ax en inferencia (pesos nartaa) | — | — | — | — | — | — | 0.945 | didnt | Kaggle | sin slots cuesta gold −0.009; variante Ch (224+half, 94 ventanas) = 0.945 en LB |
| exp029 raptor_entreno | Raptor | — | reproducir receta Raptor (CoAtNet-2 MIL) fold 0 | 0.8682 | 0.8679 | 0.9115 | — | — | 2.51 | 0.933 | didnt | — | criterio pseudo ≥ 0.90 no cumplido, pero LB 0.933 fold 0 |
| exp030 raptor_etiquetas_L2 | Raptor | exp029 | etiquetas Raptor → L2 blandas | 0.8795 | 0.8793 | 0.9123 | +0.0113 | +0.0008 | 2.50 | 0.930 | worked | rsna-c2 | verdict e7e1b6d. LB all-data 0.930 < exp029 0.933: la mejora en pseudo L2 es en parte sesgo hacia L2 |
| exp031 raptor_etiquetas_consenso | Raptor | exp029 | etiquetas = promedio Raptor+L2+L3+L4 | 0.8836 | 0.8835 | 0.9144 | +0.0155 | +0.0028 | 2.49 | 0.939 | worked | rsna-f | PADRE del paso 7; mejor modelo propio en LB (all-data) |
| exp032 raptor_alineado_LR | Raptor | exp029 | espejar rodillas izquierdas a derechas | 0.8704 | 0.8703 | 0.9143 | +0.0022 | +0.0028 | 2.50 | — | didnt | — | el modelo ya aprende ambas orientaciones |
| exp033 raptor_24epocas | Raptor | exp029 | 16 → 24 épocas | 0.8695 | 0.8715 | 0.9195 | +0.0013 | +0.0080 | 3.79 | — | didnt | — | techo en ép. 14; 16 épocas ya en meseta |
| exp034 raptor_convnext | Raptor | exp030 | backbone ConvNeXt-B 384 (diversidad) | 0.8587 | 0.8584 | 0.8681 | -0.0207 | -0.0442 | 1.21 | — | didnt | rsna-d2 | 2× más rápido pero peor; ensamble de rangos −0.0026 |
| exp040 robustez | Raptor | exp029 | análisis de robustez (solo inferencia) | — | — | 0.9118 | — | — | 0.59 | — | worked | rsna-e | ruido 5 % −0.011, sin sagital −0.028, TTA +0.002, ECE 0.20 |
| exp043 raptor_3slots | Raptor | exp030 | entrenar solo con 3 slots (eficiencia) | 0.8785 | 0.8800 | 0.8856 | -0.0010 | -0.0267 | 2.47 | — | killed | — | veto gold −0.027; podar slots solo en inferencia sí sirve |
| exp044 raptor_ft_nartaa | T1 Kaggle | exp031 | fine-tune pesos nartaa con etiquetas consenso, all-data | — | — | — | — | — | — | — | en cola | rsna-c2 | all-data listo; pseudo/gold contaminados; decide el LB vía exp042. Revisar regla de datos externos (OAI) |
| sub031_1 consenso_jev | Paso 7 · T5 | exp031 | consenso + JEV como 5.ª fuente | 0.8846 | 0.8846 | 0.9107 | +0.0010 | -0.0037 | 2.53 | — | didnt | rsna-g | dentro del ruido |
| sub031_2 semilla | Paso 7 · T5 | exp031 | semilla 42 → 7 (ruido) | 0.8852 | 0.8849 | 0.9178 | +0.0015 | +0.0035 | 2.50 | — | worked | rsna-c2 | verdict 1bfa45f (informativo). Ruido de semilla 0.0013; ensamble 2 semillas 0.8878 |
| sub031_3 lr_backbone | Paso 7 · T5 | exp031 | lr backbone 6e-5 | 0.8832 | 0.8840 | 0.9146 | -0.0005 | +0.0003 | 2.50 | — | didnt | rsna-d2 | verdict 6e5c640. Dentro del ruido |
| sub031_4 drop_path | Paso 7 · T5 | exp031 | drop_path 0.2 | 0.8849 | 0.8847 | 0.9097 | +0.0013 | -0.0047 | 2.55 | — | didnt | rsna-f | dentro del ruido |
| exp045 raptor_aug_espacial | Paso 7 · T2 | exp031 | aug espacial: espejo L/R + afín | 0.8899 | 0.8895 | 0.9202 | +0.0063 | +0.0058 | 2.51 | — | worked | rsna-h | único worked de la tanda; v1 +0.008; MCL +0.024, ACL +0.021, LatMen +0.015; all-data listo (gold 0.9186); 5-fold en curso (f1 F, f2 C2, f3 I, f4 H; fin ~10:20). La curva aún sube en ép. 15 |
| sub045_1 afin_fuerte | Paso 7 · T2 | exp045 | afín más fuerte | — | — | — | — | — | — | — | en cola | — | tras 10:20 |
| sub045_2 20_epocas | Paso 7 · T2 | exp045 | exp045 con 20 épocas (la curva aún sube) | — | — | — | — | — | — | — | en cola | — | tras 10:20 |
| exp046 raptor_aug_intensidad | Paso 7 · T2 | exp031 | aug intensidad: gamma, contraste, ruido, blur, sesgo | 0.8854 | 0.8858 | 0.9039 | +0.0018 | -0.0105 | 2.51 | — | didnt | rsna-i | no llega al umbral (+0.003); gold SWA 0.9077 |
| sub046_1 aug_intensidad_suave | Paso 7 · T2 | exp046 | intensidad de exp046 en versión suave | 0.8850 | 0.8849 | 0.9131 | +0.0014 | -0.0013 | 2.51 | — | didnt | rsna-i | versión suave tampoco llega |
| exp047 raptor_slot_dropout | Paso 7 · T2 | exp031 | dropout de slot p 0.15 + jitter de cortes | 0.8844 | 0.8849 | 0.9139 | +0.0007 | -0.0005 | 2.50 | — | didnt | rsna-d2 | análisis de robustez en curso |
| exp048 raptor_mixup | Paso 7 · T2 | exp031 | mixup a nivel estudio | 0.8845 | 0.8847 | 0.9067 | +0.0009 | -0.0077 | 2.49 | — | didnt | rsna-f |  |
| exp049 raptor_clahe | Paso 7 · T3 | exp031 | CLAHE por corte | 0.8864 | 0.8861 | 0.9083 | +0.0028 | -0.0061 | 2.47 | — | didnt | rsna-e2 | al borde del umbral; rank-ens exp049+exp031 = 0.8888 (≈ semilla); candidato a sub056_k |
| exp050 raptor_res384 | Paso 7 · T3 | exp031 | entrada 384 completa (sin recorte 320) | 0.8856 | 0.8861 | 0.9172 | +0.0019 | +0.0028 | 3.63 | — | didnt | rsna-c2 | coste ×1.46 |
| exp051 raptor_norm | Paso 7 · T3 | exp031 | percentiles 0.5–99.5 por serie | (0.8816 ép. 13/16) | — | — | — | — | — | — | corriendo | rsna-e2 | fin ~08:35 |
| exp052 raptor_etiqueta_estocastica | Paso 7 · T4 | exp031 | fuente de etiqueta al azar por estudio y época | 0.8851 | 0.8847 | 0.9150 | +0.0014 | +0.0006 | 2.53 | — | didnt | rsna-g |  |
| exp053 raptor_etiqueta_temperatura | Paso 7 · T4 | exp031 | consenso afilado T 0.7 | (0.8756 ép. 8/16) | — | — | — | — | — | — | corriendo | rsna-g |  |
| exp054 etiquetas_L5 | Paso 7 · T4 | exp031 | + L5 (GPT, 3 prompts) al consenso | (0.8789 ép. 9/16) | — | — | — | — | — | — | corriendo | rsna-d2 | fin ~09:20 |
| exp056 combo | Paso 7 · T6 | exp031 | combo sobre exp045 (único worked) ± exp049 al borde como sub056_k | — | — | — | — | — | — | — | en cola | — | 5-fold de exp045 en curso (fin ~10:20) → ensamble 5 folds + blend-mine; pesos por OOF o fijos |
| exp038 ens_blendmine_nuestro | T1 Kaggle | exp023 | blend-mine + exp030/exp031 all-data, rangos 0.35/0.30/0.35 | — | — | — | — | — | — | 0.952 | didnt | Kaggle | con exp030 0.950, con exp031 0.952: ambos < 0.954 de blend-mine solo → bloque propio a 0.15 en exp039 |
| exp039 ens_bloque_nuestro | T1 Kaggle | exp038 | blend-mine + bloque nuestro a peso 0.15 | — | — | — | — | — | — | — | en cola | Kaggle | kernel ens-exp039-w15 listo; envío 13-oct |
| exp042 ens_ft_nartaa | T1 Kaggle | exp038 | blend-mine con CoAtNet nartaa → exp044 | — | — | — | — | — | — | — | en cola | Kaggle | exp042b: nartaa → rank(nartaa + exp044); envío 13-oct |
| exp055 eficiencia_3slots | Eficiencia | exp031 | exp031 all-data, 3 slots, sin TTA, half en T4 | — | — | — | — | — | — | — | en cola | Kaggle |  |

Δ = mejor época contra la mejor época del padre (como en los verdicts). Entre paréntesis: valor parcial del run en curso, no es resultado.

## Envíos a Kaggle

| fecha (UTC) | ID | qué | propio | LB público | runtime notebook (s) | tiempo hasta score |
|---|---|---|---|---|---|---|
| 2026-10-08 17:01 | 56963705 | exp002 DINOv2-S, 5 folds | sí | 0.812 | 47.2 | — |
| 2026-10-08 19:35 | 56969845 | exp015 DINOv2-S, fold 0, 16 ép. | sí | 0.810 | 45.0 | — |
| 2026-10-09 00:45 | 56981839 | exp021 DINOv2-S, etiquetas L2 | sí | 0.836 | 62.7 | — |
| 2026-10-11 17:29 | 57081163 | exp023: nartaa 0949 mirror tal cual | no | 0.949 | 66.6 | ~25–29 min |
| 2026-10-11 17:33 | 57081259 | exp023: velci-kcd25-20 tal cual (base 0.954) | no | 0.954 | 194.1 | ~3 h 25 min |
| 2026-10-11 18:30 | 57082625 | exp023: blend-mine tal cual | no | 0.954 | 184.3 | ~4 h 10 min |
| 2026-10-11 19:06 | 57083477 | exp026 V2: nartaa sin TTA + channels_last (eficiencia) | no | 0.948 | — | ~13.7 min T4 medido |
| 2026-10-11 20:18 | 57084774 | exp029 Raptor propio fold 0 + TTA | sí | 0.933 | 63.0 | — |
| 2026-10-12 00:01 | 57088360 | exp030 all-data (etiquetas L2) + TTA | sí | 0.930 | 73.6 | ~27 min |
| 2026-10-12 00:28 | 57089745 | Ch: nartaa 224 half, 94 ventanas, sin TTA (eficiencia) | no | 0.945 | 54.1 | ~8 min |
| 2026-10-12 00:37 | 57089954 | exp038: blend-mine + exp030 all-data | mixto | 0.950 | 198.6 | > 3 h (listo antes de 05:00) |
| 2026-10-12 01:44 | 57091516 | exp031 all-data (consenso) + TTA | sí | 0.939 | 64.6 | ~28 min |
| 2026-10-12 02:13 | 57092253 | exp038: blend-mine + exp031 all-data | mixto | 0.952 | 209.1 | listo antes de 08:04 |
| 2026-10-13 00:05 | prog-1 | exp045 all-data solo (aug espacial) | sí | programado | — | programado |
| 2026-10-13 00:05 | prog-2 | exp039: blend-mine + bloque nuestro a 0.15 | mixto | programado | — | programado |
| 2026-10-13 00:05 | prog-3 | exp042b: blend-mine con rank(nartaa + exp044) | mixto | programado | — | programado |
| 2026-10-13 00:05 | prog-4 | ensamble 5 folds exp045 (+ blend-mine) | mixto | programado | — | programado |

## Recursos

| pod | GPU | corre | fin estimado (UTC) | siguiente |
|---|---|---|---|---|
| rsna-c2 | RTX PRO 4500 32 GB | exp045 fold 2 | ~10:20 | — |
| rsna-d2 | RTX PRO 4500 32 GB | exp054 L5 (9/16 ép.) | ~09:20 | — |
| rsna-e2 | RTX PRO 4500 32 GB | exp051 norm (13/16 ép.) | ~08:35 | — |
| rsna-f | RTX PRO 4500 32 GB | exp045 fold 1 | ~10:20 | sub045_k / exp056 |
| rsna-g | RTX PRO 4500 32 GB | exp053 temperatura (8/16 ép.) | ~09:20 | — |
| rsna-h | RTX PRO 4500 32 GB | exp045 fold 4 | ~10:20 | exp056 fold 0 |
| rsna-i | RTX PRO 4500 32 GB | exp045 fold 3 | ~10:20 | — |

Saldo RunPod 2026-10-12T08:01:52Z: US$35.60. 7 pods ≈ US$5.2/h.

## Linaje

```
exp029 (Raptor propio, LB 0.933)
 ├─ exp030 etiquetas L2 ── worked ─┬─ exp034 ConvNeXt (didnt)
 │                                 └─ exp043 3 slots (killed)
 ├─ exp031 etiquetas consenso ── worked, PADRE paso 7, LB 0.939 all-data
 │    ├─ sub031_1..4 (semilla, lr, drop_path, JEV)
 │    ├─ exp045 aug espacial (ÚNICO worked) ──▶ all-data (listo) + 5 folds (f1–f4 corriendo) ──▶ ensamble 5 folds + blend-mine
 │    │      ├─ sub045_1 afín fuerte, sub045_2 20 épocas (en cola)
 │    │      └─▶ exp056 combo (exp045 ± exp049 al borde como sub056_k)
 │    ├─ exp046..exp054 didnt (augs de intensidad, slot dropout, mixup, CLAHE, res384, etiqueta estocástica); exp051/053/054 corriendo
 │    └─ exp044 fine-tune nartaa ──▶ exp042
 ├─ exp032 alineado L/R (didnt)
 └─ exp033 24 épocas (didnt)
exp023 base pública 0.954 ──▶ exp038 (bloque propio a 0.35: 0.950 / 0.952, resta) ──▶ exp039 (bloque a 0.15) / exp042b (rank nartaa+exp044)
```

## Lectura

**Resumen de la primera tanda del paso 7:** de 11 ideas solo la augmentation geométrica (exp045) cruzó el umbral. La receta Raptor ya está
afinada: épocas, lr, drop_path, fuentes extra de etiqueta, intensidad, mixup, CLAHE, resolución y dropout de slot quedan dentro de ~+0.003.

**Qué funcionó y por qué**

- **Las etiquetas son la palanca principal.** exp021 (DINOv2 con etiquetas L2) subió el LB de 0.812 a 0.836 sin tocar el modelo.
  En la receta Raptor pasó lo mismo: el consenso de 4 fuentes (exp031) da pseudo L2 0.8836 contra 0.8682 de exp029 (+0.0155),
  gold estable, y en LB all-data 0.939 contra 0.930 de exp030 (solo L2). Promediar fuentes LLM reduce el ruido de cada una.
  Ojo: la vara pseudo L2 está sesgada a favor (el consenso contiene L2); el LB lo confirma igual.
- **Reproducir la base pública** (exp023) da 0.954 en nuestra cuenta: es el piso sobre el que se ensambla en T1.
- **Eficiencia:** el CoAtNet nartaa a 224 con half y sin TTA (Ch) da 0.945 en ~8 min hasta score, contra ~3.5–4 h de los blends de 0.954.
  Podar slots solo en inferencia funciona con nuestros modelos (exp030 con 3 slots conserva pseudo y gold), no con los pesos nartaa (exp041).
- **Aug espacial** (exp045, espejo L/R + afín): pseudo L2 0.8899 (+0.0063 vs exp031) y gold 0.9202. Es la primera ablation del paso 7 que pasa
  el umbral: la receta Raptor casi no tenía augmentation y el espejo es anatómicamente válido. Su all-data corre en rsna-h y entra a exp056.
- **Ruido medido:** dos semillas de exp031 difieren 0.0013 en pseudo L2 (sub031_2); el ensamble de las dos semillas sube a 0.8878.
  Ese es el umbral real: lo que mueva menos de ~+0.003 no se distingue del azar.

**Qué no funcionó y por qué**

- **Más épocas** (exp033): techo en la época 14, luego baja; 16 épocas ya están en meseta y cuesta ×1.5 GPU-h.
- **Alinear L/R** (exp032): +0.002, dentro del ruido; CoAtNet+MIL ya aprende ambas orientaciones (el TTA espejo funciona por lo mismo).
- **ConvNeXt-B como diversidad** (exp034): 2× más rápido pero −0.021 pseudo y −0.044 gold; el ensamble de rangos con exp030 resta (−0.0026).
- **Entrenar con 3 slots** (exp043): pseudo igual pero gold −0.027 (veto). El axial aporta en derrame, fractura, sinovitis y Baker.
- **lr del backbone** (sub031_3), **JEV como 5.ª fuente** (sub031_1) y **drop_path 0.2** (sub031_4): +0.001 o menos, dentro del ruido.
- **Aug de intensidad** (exp046): +0.0018, no llega al umbral y gold baja (−0.0105). Se prueba una versión suave (sub046_1).
- **Ensambles con nuestro bloque a 0.35** (exp038): con exp030 0.950, con exp031 0.952, ambos por debajo de blend-mine solo (0.954).
  Un modelo propio más débil con tanto peso resta → exp039 baja el peso del bloque a 0.15.
- **Resto de la tanda** (didnt): slot dropout exp047 +0.0007, mixup exp048 +0.0009, CLAHE exp049 +0.0028 (al borde; candidato a sub056_k),
  res384 exp050 +0.0019 con coste ×1.46, etiqueta estocástica exp052 +0.0014, intensidad suave sub046_1 +0.0014.
- **Brecha fold 0 vs LB:** un fold 0 propio da 0.933; la versión pública con todos los datos y supervisión externa da 0.949. Por eso los envíos propios son all-data.

**Cómo se apila lo que funciona**

1. Cada ablation del paso 7 (augs T2, preprocesado T3, etiquetas T4) se mide contra exp031 en fold 0. Promete con Δ ≥ +0.003.
2. **exp056 combo** = exp045 (único worked) ± exp049 al borde como sub056_k. En paralelo, exp045 corre los folds 1–4 en 4 pods.
3. **Ensamble de 5 folds de exp045 + blend-mine**. Pesos elegidos con OOF, corrigiendo que los
   públicos vieron todo el train; si no se puede corregir, pesos de rango fijos decididos antes de mirar el LB.
4. Mientras tanto, T1 sigue con los all-data actuales (exp038 → exp039/exp042). El LB solo elige entre ≤ 2 variantes por día.

**A vigilar**

- exp044 parte de pesos nartaa que traen supervisión externa (OAI): revisar las reglas de datos externos antes del envío final / premio.
- exp030 worked en pseudo L2 pero su all-data (0.930) queda bajo exp029 (0.933) en LB: la vara pseudo L2 premia en parte el sesgo hacia L2. exp035 quedó no concluyente.
- Corriendo: exp045 folds 1–4 (F, C2, I, H; fin ~10:20), exp051 (E2), exp053 (G), exp054 (D2). Envíos del 13-oct programados: exp045 solo,
  ens-exp039-w15, ens-exp042b y ensamble 5-fold de exp045.

