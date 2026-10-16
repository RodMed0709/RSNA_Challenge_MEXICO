# Envíos a Kaggle

Registro de cada envío: qué modelo, LB público, puesto y runtime. Los kernels y logs viven fuera del repo (scratchpad de la
sesión); aquí solo los números. Kernels privados, sin internet, GPU T4.

| Fecha | ID envío | Kernel | Modelo | LB público | Puesto aprox. | Runtime notebook (test visible, 3 estudios) |
|---|---|---|---|---|---|---|
| 2026-10-08 | 56963705 | rsna-knee-exp002-inference v2 | exp002, ensamble 5 folds (8 épocas) | 0.812 | ~4368 / 5530 | 47.2 s |
| 2026-10-08 | 56969845 | rsna-knee-exp015-inference v1 | exp015, solo fold 0 (16 épocas) | 0.810 | ~4374 / 5530 | 45.0 s |
| 2026-10-09 | 56981839 | rsna-knee-exp021-inference v1 | exp021, solo fold 0 (8 épocas, etiquetas L2) | **0.836** | ~4266 / 5543 | 62.7 s |
| 2026-10-11 | 57081163 | mirror-exp023 v1 | exp023: copia tal cual de `nartaa/rsna-knee-0949-anatomical-mirror` (CoAtNet Raptor SWA + espejo L/R) | 0.949 | ~1175–1305 / 5760 (empate) | 66.6 s (score en ~25–29 min) |
| 2026-10-11 | 57081259 | velci-exp023 v1 | exp023: copia tal cual de `bliverrigpdoge/velci-kcd25-20`; `rsna-knee-kcd25leg` privado → queda la base 0.954 (Raptor 0.7 + compact S6 0.3) | **0.954** | ~135–535 / 5760 (empate) | 194.1 s (score en ~3 h 25 min) |
| 2026-10-11 | 57082625 | blendmine-exp023 v2 | exp023: copia tal cual de `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters` (Raptor 0.5 + compact S6 0.5); `rsna-knee-compact-r6cnxfsss` v2 re-alojado privado | **0.954** | ~135–535 / 5760 (empate) | 184.3 s (score en ~4 h 10 min) |
| 2026-10-11 | 57084774 | exp029-raptor-f0 v1 | exp029, solo fold 0 (receta Raptor/nartaa CoAtNet-2 MIL, etiquetas blandas Raptor, best_f0 ép. 15) + TTA espejo, inferencia nartaa 0949 | **0.933** | — | 63 s (script 25 s) |
| 2026-10-12 | 57088360 | exp030-raptor-alldata v1 | exp030 all-data (receta exp029 con etiquetas L2, 4349 no-gold, SWA épocas 12/14/15) + TTA espejo, inferencia nartaa 0949 | 0.930 | — | 73.6 s (script 27 s; score en ~27 min) |
| 2026-10-12 | 57089745 | exp041-sub-e1-94 v1 | exp027/041 Ch: pesos nartaa 224 (`0945-efficient-224crop`), `model.half()`, 94 ventanas en un lote, sin TTA (eficiencia) | **0.945** | — | 54.1 s (script 12 s; score en ~8 min) |
| 2026-10-12 | 57089954 | ens-blendmine-exp030 v1 | exp038: blend-mine (Raptor nartaa + compact S6) + exp030 all-data, rank por hallazgo 0.35/0.30/0.35 (fijos) | 0.950 | — | 198.6 s (Raptor 42 s, exp030 28 s, compact 95 s; score en ~3 h 58 min) |
| 2026-10-12 | 57091516 | exp031-raptor-alldata v1 | exp031 all-data (etiquetas consenso, SWA épocas 13/14/15) + TTA espejo, inferencia nartaa 0949 | 0.939 | — | 64.6 s (script 25 s; score en ~28 min) |
| 2026-10-12 | 57092253 | ens-blendmine-exp031 v1 | exp038: blend-mine + exp031 all-data, rank por hallazgo 0.35/0.30/0.35 (fijos) | 0.952 | — | 209.1 s (Raptor 49 s, exp031 27 s, compact 100 s; score en ~3 h 50 min) |
| 2026-10-15 (cola) | pendiente | r7-raptor-repro v1 | Paso 8 a): copia privada tal cual de `heliosli/0-957-latest-dataset-score-in-one-click` (compact R7 `heliosli/rsna-knee-compact-r7cnxfsss` + Raptor nartaa 384, rank 0.5/0.5). Evidencia: LB público 0.957 del notebook. Nueva base | pendiente | — | 432 s (Raptor 54 s); salida = pública en los 3 estudios visibles |
| 2026-10-15 (cola) | pendiente | r7-raptor-r224flip v1 | Paso 8 b): copia privada tal cual de `varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up` (R7 0.5 + Raptor = 0.65 nartaa 384 + 0.35 nartaa 224 crop plano+espejo). Evidencia: LB público 0.956 del notebook. Pregunta: ¿el 224 espejado suma sobre a)? | pendiente | — | 492 s (Raptor 55 s, 224 28 s); salida = pública en los 3 estudios visibles |

## Lectura

- **exp023 (reproducción de públicos):** la base pública se reproduce en nuestra cuenta: velci y blend-mine dan 0.954,
  el CoAtNet solo de nartaa 0.949. El "runtime" de la columna es el del run sobre el test visible (3 estudios); el tiempo
  hasta tener score (incluye cola de Kaggle y test oculto) es la cota que tenemos del runtime real: ~0.5 h el CoAtNet solo,
  ~3.5–4 h los blends con el compact S6. Para el premio de eficiencia, el compact cuesta ~3 h por +0.005 AUC.

- **exp021 confirma la hipótesis de las etiquetas:** misma arquitectura y receta que exp002, solo etiquetas L2 → LB +0.024.
  Es el primer movimiento real del LB. La pseudo contra v1 bajó (−0.038) mientras el LB subió: v1 ya no sirve de vara interna.

- 1 fold de 16 épocas empata con el ensamble de 5 folds de 8 épocas, con 1/5 del cómputo del modelo (forward 0.52 s/estudio
  contra 0.94 s). Para el track de eficiencia, el ensamble no paga.
- La brecha con el LB es grande: mediana del LB 0.936, nosotros 0.81, mientras la pseudo AUC interna sube a 0.85. Lo que
  más mueve la validación interna (entrada, épocas) apenas toca el LB. Hipótesis principal: las etiquetas de Kaggle siguen
  definiciones estrictas (ACL y MCL solo alto grado o completa; derrame y Baker solo moderado o grande; OA con pérdida de
  cartílago >50 %) y nuestras etiquetas v1 no. La validación interna mide contra v1, no contra la definición del test.
  Esto sube la prioridad de exp011 (etiquetas v2 por LLM con las definiciones oficiales) por encima de más arquitectura.
- En el notebook, leer DICOM domina (22 s de workers en 3 estudios); el forward es menor. Coincide con
  `costo_inferencia.md`: el preprocesado manda el runtime.

- **exp038 (12-oct):** sumar nuestro CoAtNet all-data a blend-mine con peso 0.35 resta: exp030 → 0.950, exp031 → 0.952,
  contra 0.954 de blend-mine solo (los dos tardaron ~3 h 50 min–4 h hasta el score). Solos: exp030 0.930, exp031 0.939.
  Siguientes ensambles con el bloque nuestro a peso bajo (0.15).
- **Datos externos (exp044 / exp042):** los pesos de nartaa que exp044 afina traen supervisión externa (OAI). Las reglas
  dicen "Freely & publicly available external data is allowed, including pre-trained models"
  (`docs/papers/00_competencia/README.md`); los pesos son un dataset público de Kaggle y blend-mine, público, ya los usa.
  OAI pide registro y acuerdo de uso (gratuito): si la organización lo considerara "no libre", afectaría por igual a
  blend-mine y a todo lo que use esos pesos.

## 15-oct (cola del 15, disparo 00:05 UTC) — paso 8, sobre R7

| ID | Kernel | Qué es | LB público |
|---|---|---|---|
| 57198921 | r7-raptor-repro v1 | R7 compact + Raptor nartaa 384, 0.5/0.5 tal cual (público dice 0.957) | 0.956 |
| 57199075 | r7-raptor-r224flip v1 | + nartaa 224 plain+mirror 0.35 en el slot Raptor | 0.956 |
| 57199209 | r7-raptor-exp064 v1 | R7 0.5 + slot Raptor = rank mean(nartaa384, exp064) | 0.956 |
| 57199291 | r7-raptor-exp044 v1 | R7 0.5 + slot Raptor = rank mean(nartaa384, exp044) | 0.956 |
| 57199375 | exp071-solo v1 | exp071 solo (decide su verdict: worked si ≥ 0.946) | 0.944 → didnt |

- La reproducción de R7+Raptor da 0.956 (el público reporta 0.957; diferencia de redondeo/ruido, ~390 estudios).
- Meter nuestras piezas de init nartaa (exp044, exp064) o nartaa 224 en el slot Raptor no mueve el LB: las cuatro 0.956.
  Con Δ < 0.004 como ruido, no hay evidencia para seguir sumando piezas Raptor propias sobre R7.
- Los R7 tardaron ~3 h 40 min hasta el score.

## Regla de finales (fijada el 15-oct ~15:30 UTC, antes de ver los envíos del 16 al 18)

- **A, mejor AUC:** núcleo R7+Raptor + la pieza (o unión de piezas ya medidas) con evidencia interna limpia y LB ≥ 0.956
  que más diversidad aporte. Preferencia por fuerza de evidencia, no por el 4.º decimal: exp067 > S6fall > hijo L6 de
  exp069. Si ninguna llega a 0.956, A = `r7-raptor-r224flip` (0.956). S6fall usa pesos CC BY-NC 4.0: antes de elegirlo
  como final hay que confirmar que no choca con la licencia exigida a los ganadores; si choca, se salta al siguiente.
- **B, eficiente:** antes del 18-oct leer las reglas del track de eficiencia. Por defecto, Raptor 384 solo con espejo
  (0.949, ~27 min), o sin espejo si la regla prima el tiempo (exp026: −0.001, −50 % de tiempo). Si el track no necesita
  una final propia, B = `r7-raptor-repro` como cobertura de A.
- Cada pieza nueva cuenta como mejora solo si LB ≥ 0.958 (0.956 + 0.002); entre 0.956 y 0.957 = empate (ruido).

## Cola 16-oct (disparo 00:05 UTC) y 17-oct

| Fecha | Kernel | Pregunta | Criterio |
|---|---|---|---|
| 16 | r7-raptor-exp067 v1 | compact = rank R7 0.75 / exp067 0.25 | worked si ≥ 0.958 |
| 16 | r7-raptor-s6fall v1 | padre r224flip; compact = rank R7 0.6 / S6fall 0.4 (leoprovorov v2) | ≥ 0.958; killed si S6 no puntúa 100 % (receipt) |
| 16 (adelantado del 17) | r7-raptor-exp069 v1 | compact = rank R7 0.75 / exp069 0.25 (control del hijo L6) | worked si ≥ 0.958 |

exp069 se adelantó al 16-oct (ref 57250257, enviado 01:21 UTC a mano desde rsna-cron); la cola del 17 queda vacía.

## Regla para los 2 lugares libres del 16-oct (fijada 01:30 UTC, antes de ver exp067 / S6fall / exp069)

Cuenta igual que arriba: ≥ 0.958 = worked; 0.957 = empate; ≤ 0.956 = nada.

- **Alguna pieza ≥ 0.958:**
  1. Si ≥ 2 piezas worked → unión: compact = rank R7 + las piezas worked, repartiendo a partes iguales el peso que
     tenían (exp067/exp069 0.25, S6fall 0.4). Si solo una → dosis: subir su peso (exp067/exp069 a 0.4, S6fall a 0.5).
  2. Pieza worked + la mejor de las otras si sacó 0.957 (unión, mismo reparto). Si ninguna otra sacó 0.957, sin 2.º envío.
- **Mejor = 0.957 (empate) en ≥ 2 piezas:** 1 solo envío, unión de las de 0.957 (pregunta: ¿se suman?).
- **Todo ≤ 0.956 o solo una en 0.957:** no se usa ningún lugar (mejor menos que relleno).
- Antes de enviar, el kernel debe caber en el tiempo de Kaggle (S6fall ya estima ~5.2 h): si la unión pasa de ~8 h, se
  quita la pieza más lenta.

## 16-oct — resultados (cola del 16 + exp069 adelantado)

| ref | Kernel | LB | Lectura |
|---|---|---|---|
| 57247862 | r7-raptor-exp067 v1 (R7 0.75 / exp067 0.25) | 0.957 | empate (+0.001, ruido) |
| 57248076 | r7-raptor-s6fall v1 (R7 0.6 / S6fall 0.4, padre r224flip) | 0.957 | empate |
| 57250257 | r7-raptor-exp069 v1 (R7 0.75 / exp069 0.25) | 0.957 | empate |

Ninguna pieza llega a 0.958 → ninguna `worked`. Tres en 0.957 → por la regla de arriba, 1 solo envío con la unión.
Pesos de la unión (fijados aquí, antes de enviarla): compact = rank R7 0.7 + exp067 0.1 + exp069 0.1 + S6fall 0.1
(el peso medio que tenían las piezas, 0.3, repartido a partes iguales). Base = kernel s6fall (Raptor r224flip). Una pieza que
falle o no quepa en el reloj se quita por estudio y se renormalizan los pesos. Kernel `r7-raptor-union3` v1.
Criterio: worked si ≥ 0.958; 0.957 = las piezas no se suman. El 5.º lugar del día queda sin usar.

**union3 (ref 57257627, enviado 06:10 UTC): 0.957.** Las tres piezas no se suman. Verdicts: exp067 didnt, exp069 didnt (en sus ramas).
Por la regla de finales, A = `r7-raptor-exp067` (0.957; preferencia exp067 > S6fall, que además lleva pesos CC BY-NC).
5.º lugar del 16 sin usar.

## Candidato cola 17-oct: r7-raptor-exp067flip v1 (fijado 16-oct 14:40 UTC, antes de enviarlo)
Pregunta: ¿la TTA espejo del trainer v9.4 (`--flip-tta`) en R7 mejora la final A? Un solo cambio sobre `r7-raptor-exp067`
(0.957): R7 corre con `--flip-tta`; exp067, Raptor y pesos iguales. worked si LB ≥ 0.959; entra a la cola solo si la
validación del kernel termina COMPLETE sin error. Evidencia pública de flip TTA: sobre R6, no R7.
