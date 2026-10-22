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

## Track de eficiencia: lectura de reglas y envíos del 17-oct (fijado 16-oct ~14:30 UTC, antes de enviar)

- Fórmula oficial: `Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400` (se minimiza; benchmark = sample
  submission). Con maxAUC ≈ 0.965, **0.001 AUC ≈ 70 s**. Cuenta solo una submission seleccionada como final (o
  autoseleccionada) que supere al benchmark en privado.
- El LB de eficiencia público (ryanholbrook, 16-oct) usa la autoseleccionada = mejor LB público: Mexico-TEAM puesto
  **3417 / 5955** con la 0.954 del 11-oct (~3.5 h). Los 0.945 rápidos (copias de nartaa 224) están en los puestos 22–35.
  ⇒ **La final B tiene que ser la eficiente, elegida a mano.** Por defecto B = `exp041-sub-e1-94` (0.945, ~8 min).
- nartaa `0945-efficient-224crop`: CoAtNet 224 (recorte 0.8 del campo 384), K94, fp16, sin TTA, lector DICOM `fastread
  study_lut`, pool de procesos + un consumidor por T4. exp041 ya tiene todo eso (verificado): reenviarlo no pregunta nada.

| Kernel | Pregunta | Esperado | Criterio |
|---|---|---|---|
| exp041-mirror224 v1 | ¿El espejo L/R (misma lectura, solo GPU) paga su runtime en el 224? Media en probabilidad plano + espejo (`_mirror_tri` de nartaa) | 0.945–0.947; +60–120 s | worked si LB ≥ 0.947 y envío→score ≤ exp041 + 3 min; si no, B = exp041 |
| exp074-student224 v1 (condicional) | ¿El 224 de 3 slots destilado del 0.957 (exp074) sube el LB a runtime de exp057? | 0.945–0.950; runtime ≤ exp041 | solo si exp074 worked y hay all-data antes de 00:05 UTC; worked si LB ≥ 0.948 |

## 17-oct — R6 + flip TTA: copias fieles de los públicos a 0.958 (fijado 15:05 UTC, antes de ver LB)

Origen: barrido público del 17-oct. Dos notebooks públicos llegan a 0.958 con **R6 + flip TTA** (bundle fe055406) en vez de
R7, Raptor nartaa 0.65·384 + 0.35·224 con espejo y pesos de Raptor por hallazgo. Nuestro mejor: 0.957. Pesos copiados tal
cual, sin ajustar nada. Usamos los 2 lugares que quedaban del 17 (ya había 3 envíos).

| ref | Kernel | Copia de | Pregunta | Criterio | Validación |
|---|---|---|---|---|---|
| 57315883 | r6flip-repro v1 | heliosli/blend-mine-in-private-lb-luck-to-all-upvoters | ¿Reproducimos 0.958 con R6+flip+pesos por hallazgo (Raptor Syn 0.8, LatOA 0.2, PFOA 0.6, Baker 0.35, resto 0.5)? | worked si LB ≥ 0.958 | COMPLETE, 459 s, flip_tta True, sin fallback |
| 57315940 | r6flip-v2 v1 | heliosli/this-is-best-public-dataset-new-version-0-9586 | ¿Pesos V2 (LatOA 0.15, PFOA 0.65, Syn 0.85, Baker 0.30) + lector 2.5D de goodpjw a 1/10 superan a r6flip-repro? | worked si LB ≥ 0.959 | COMPLETE, 535 s, flip_tta True, "own leg: blended 3 checkpoints" |

- r6flip-repro: el código es idéntico byte a byte al público (solo se quitaron outputs); mismos datasets montados.
- r6flip-v2: **una única desviación declarada**: `C_SLUG` apunta a `heliosli/rsna-knee-r6-frozen-fe055406-w1` en vez de
  `heliosli/rsna-knee-r6-frozen-fe055406`, porque la última versión de este último contiene R7 (bundle 8c7b666e) y el
  `assert bundle_id == fe055406` fallaría. -w1 es copia byte a byte de la v1 (R6); el notebook verifica el sha256 de cada
  archivo del manifiesto. Resto igual (incluido kernel_source goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958).
- Revisión del código: sin internet (`enable_internet: false`, sin urllib/requests/socket), solo lee `/kaggle/input` y
  el test de la competición; pip offline (`--no-index`) desde wheels de los datasets; sin UIDs codificados ni nada que lea
  etiquetas. `PROBE_CLASS = None`. Los `torch.load(weights_only=False)` son los de nartaa/goodpjw (ya usados antes).
- Licencias de los datasets montados: nartaa/rsna-knee-publication-swa-weights-20261007 = "other" (uso investigación /
  competición; adaptar y redistribuir solo no comercial con crédito; un ganador puede licenciar CC BY-NC 4.0; aviso OAI);
  heliosli/rsna-knee-r6-frozen-fe055406(-w1) = "other" (sin texto de licencia); heliosli/rsna-orthofoundation-fold0-epoch7-20261004
  = "other" ("no new open-source license is granted"; upstream con su licencia); goodpjw2008/rsna-knee-2-5d-convnext-reader
  = Apache-2.0. ⇒ Mismas condiciones de licencia que los assets heliosli que ya usa la final A (R7 + OrthoFoundation); el
  único asset nuevo es el lector goodpjw (Apache-2.0).

## 17 y 18-oct — resultados (LB público)

| Fecha | ref | Kernel | LB | Criterio fijado antes | Verdict |
|---|---|---|---|---|---|
| 17-oct | 57287682 | exp041-mirror224 v1 | 0.945 | worked si ≥ 0.947 | didnt |
| 17-oct | 57287873 | exp074-student224 v1 | 0.948 | worked si ≥ 0.948 | worked |
| 17-oct | 57287971 | r7-raptor-exp067flip v1 | 0.957 | worked si ≥ 0.959 | didnt |
| 17-oct | 57315883 | r6flip-repro v1 | **0.958** | worked si ≥ 0.958 | worked |
| 17-oct | 57315940 | r6flip-v2 v1 | 0.958 | worked si ≥ 0.959 | didnt |
| 18-oct | 57330468 | sub078-1-bigru224 v1 | 0.948 | worked si ≥ 0.950 | didnt |
| 18-oct | 57330704 | exp080-ctrl224 v1 | 0.949 | worked si ≥ 0.950 | didnt (mejor eficiente) |
| 18-oct | 57330828 | exp074-student224-fast v1 | 0.948 | worked si ≥ 0.947 | worked (mismas predicciones) |
| 18-oct | 57330942 | sub075-1-solo v1 | 0.942 | worked si ≥ 0.946; didnt 0.941–0.945 | didnt |

Runtime de los eficientes (Kaggle no lo publica; medido con kernels bench privados `exp0xx-bench-*` que corren el pipeline
puntuado sin cambios sobre 1300 estudios de train en T4×2; "about 1300 studies" en el test oculto). Bucle principal:
exp041-sub-e1-94 506 s; exp074-student224 312 s; exp074-student224-fast 225–253 s (4 corridas, probabilidades idénticas
a la base, max|Δ| = 0); exp080-ctrl224 260 s (otra máquina: la varianza entre VMs es de ±15 %). Sumar ~43 s de arranque y
cierre del notebook. Es estimación con estudios de train, no el RuntimeSeconds oficial.

## 18 a 21-oct — resultados añadidos (LB público, leídos de Kaggle el 21-oct 17:01 UTC)

| Fecha | ref | Kernel | LB | Criterio fijado antes | Verdict |
|---|---|---|---|---|---|
| 18-oct | 57334779 | exp080-ctrl224-fast (final B) | 0.949 | — | final B |
| 19-oct | 57373574 | tank-seeds (exp085) | 0.957 | worked si ≥ 0.959 | didnt |
| 19-oct | 57400180 | tank-v6 (exp088) | 0.957 | worked si > 0.957 y ≥ 0.959 | didnt |
| 19-oct | 57400221 | tank-v6soft (exp089) | 0.957 | worked si > 0.957 y ≥ 0.959 | didnt |
| 20-oct | 57455236 | exp090 exploratorio all-data | 0.947 | — (no registrado) | — |
| 21-oct | 57459212 | exp082 exploratorio (exp080 + nartaa rank-mean 50-50) | 0.952 | umbral estimado 0.975 por runtime | didnt |

Nota: los envíos del 13 y 14-oct (exp045, exp049, exp042a/b, exp054, exp056, exp057) existen en Kaggle y faltan en este archivo.
