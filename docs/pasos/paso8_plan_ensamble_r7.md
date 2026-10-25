# Paso 8 — Plan maestro de ensamble sobre R7 (15 → 22-oct-2026)

Fecha: 2026-10-14. Estado: **PLAN** (no hay SPEC confirmada de ningún experimento de aquí; cada entrenamiento nuevo
necesita su `nuevo_experimento(...)` y su SPEC antes de tocar código). Fuentes: `docs/conocimiento/04_robustez_ensambles_lb.md`,
`08_backlog_priorizado.md`, `05_eficiencia.md`, `local/RSNA_papers/analisis/A4_interno_vs_LB.md` §2–4, `A6_revision_colas.md` §4,
`docs/pasos/raptor_receta.md`, contexto de la revisión del 14-oct y búsqueda web (ver §8).

## 0. Punto de partida (números, no hipótesis)

- Núcleo público: **R7 compact ConvNeXt-S + Raptor CoAtNet nartaa 384, 0.5/0.5 = 0.957**; pesos 40/60…60/40 = 0.956 → la superficie de
  pesos es plana: **no se gana nada moviendo pesos del núcleo**.
- blend-mine (R6 + Raptor) 0.954; todo lo que le añadimos se quedó en 0.950–0.954.
- Nuestro mejor propio: exp056 0.943, exp045 5f 0.941. Desde ImageNet nos quedamos ~0.010 por debajo de nartaa; exp044 (fine-tune
  sobre pesos nartaa, init OAI) en blend 0.953 → **la brecha es el init OAI, no la receta**.
- exp066 compact ConvNeXt-S propio: pseudo 0.8755; rank 0.5/0.5 con exp045 = 0.8955 (+0.006 sobre exp045) → **compact y CoAtNet
  son diversos entre sí** (en OOF limpio). No tiene kernel ni all-data.
- Ruido: LB público ≈ 390 estudios, SD de diferencia pareada 0.002–0.0046; **Δ < 0.003 = empate**. Elegir el máximo de ~35 envíos
  infla ≈ +0.003 (maldición del ganador).
- Toy binormal (A4 §4): un bloque de 0.945 con ρ 0.90 añade +0.001 a peso 0.25; con ρ 0.80 y 0.939, +0.003. **La diversidad pesa más
  que la AUC propia.**

**Meta realista:** 0.959–0.961 público con un bloque propio OAI-init diverso; techo optimista 0.962. Los 0.963–0.966 del top no tienen
notebook público (la web no muestra nada > 0.957), así que hay que suponer que entrenan sobre OAI/pesos propios y no son copiables.

## 1. Fuentes de diversidad, ranqueadas

Ganancia esperada = Δ LB del ensamble final contra el núcleo 0.957 (hipótesis razonada con el toy de A4, **no** resultado).
GPU-h con pod RTX PRO 4500 (~$0.74/h); CoAtNet 384 all-data ≈ 3 GPU-h, compact ≈ 1.5 (ConvNeXt ~2× más rápido, exp034).

| # | Fuente | Qué es concretamente | Por qué es diversa | Δ esperada | GPU-h | Riesgo |
|---|---|---|---|---|---|---|
| 1 | **Raptor OAI-init con NUESTRAS etiquetas + aug espacial** (`ft-nartaa-aug`) | exp044 (fine-tune pesos nartaa, consenso) + aug espacial de exp045, all-data, 2 semillas | mismo init fuerte (≈0.949 solo) pero otra señal de etiqueta (consenso L2/L3/L4/Raptor ± L5) y otra aug → ρ esperado 0.85–0.90 con Raptor | +0.001 a +0.003 | 3/semilla | sin OOF limpio (pesos vieron el train); solo decide el LB |
| 2 | **R7 compact con nuestras etiquetas + aug espacial** (`ft-r7-aug`) | fine-tune de los pesos R7 publicados, all-data, 2 semillas | segunda familia (ConvNeXt) con la misma perturbación de etiqueta; el par 1+2 es un "núcleo gemelo" | +0.001 a +0.003 | 1.5/semilla | depende de que el dataset R7 traiga pesos + código de entrenamiento compatibles; verificar el 15 |
| 3 | **Etiquetas distintas** (mandato: LLM auditado, P(positivo experto), consenso + L5) | aplicar la mejor etiqueta nueva a #1/#2 (no un miembro aparte) | la definición de etiqueta es la única divergencia real interno↔LB (A4: v1 ρ −0.40) | +0.000 a +0.004 (amplificador de #1/#2) | 0 extra si se fusiona con la 2.ª semilla | etiquetador no puede usar gold; ajustar solo con dev-200 humano |
| 4 | **exp066 compact ImageNet-init (propio)**, 5f + all-data | rank-mean de folds | ρ bajo con CoAtNet (medido en OOF), pero débil (~0.93 LB estimado) | +0.000 a +0.001 a w ≤ 0.10 | ~7.5 (4 folds + all-data) | sirve sobre todo al **plan B sin OAI** y al paper |
| 5 | Bloque propio ImageNet-init CoAtNet (exp045 5f + exp054 5f + exp056) | ya entrenado | ρ 0.90–0.95 con Raptor (inferido) | ≈ 0 a w 0.15 | 0 | ya medido: no mueve; solo plan B |
| 6 | Resolución 224 (pesos nartaa 224 crop, 0.945) | miembro barato | misma familia/init que Raptor 384 → ρ alto | ≈ 0 | 0 | coste de runtime casi nulo; no prioritario |
| 7 | 2.5D ConvNeXt reader público (goodpjw2008) | 3 folds públicos | otra entrada | ≈ 0 (otros equipos: +0 en LB a 0.30) | 0 | runtime |
| 8 | 3D nativo / DINO / foundation | — | — | negativo o 0 (DINOv2 0.81–0.84; exp006 killed) | — | **descartado antes del 22** |

Orden de ejecución: **1 y 2 en paralelo el 15 → 3 se monta en la 2.ª semilla de 1 y 2 → 4 solo si sobra GPU**.
No se lanza ninguna otra arquitectura nueva: no hay tiempo para validarla y el LB no puede ver Δ < 0.003.

## 2. Pesado sin sobreajustar a 390 estudios

Reglas fijadas **hoy**, antes de ver un solo resultado (regla 8 del repo):

1. **Media de rangos por hallazgo, pesos fijos jerárquicos.**
   `final = (1 − w)·rank(núcleo R7 0.5 / Raptor 0.5) + w·rank(bloque propio)`, con
   `bloque propio = media de rangos con pesos iguales de sus miembros (familias × semillas)`.
   w ∈ **{0.15, 0.25, 0.35}** solamente; se elige con la tabla de A4 §4 usando el **LB del bloque solo** (un envío), no barriendo w:
   bloque solo < 0.945 → w 0.15; 0.945–0.951 → w 0.25; ≥ 0.952 → w 0.35 (≈ "tercer miembro par").
2. **Pesos por OOF de un conjunto común: solo dentro de lo propio ImageNet-init.** Los públicos (R7, Raptor) y los fine-tunes sobre sus
   pesos vieron todo el train y el gold → su "OOF" es in-sample. Optimizar pesos contra ellos sería un artefacto (A4 §4). Para exp045/054/066
   sí hay OOF limpio de 5 folds sobre los mismos estudios: ahí se permite **solo** decidir inclusión (entra o no), con pesos iguales.
3. **Stacking por hallazgo con OOF: NO antes del 22.** 12 hallazgos × miembros sobre ~3 600 OOF y sin OOF de los miembros fuertes = sobreajuste;
   Müller 2022 (pooling simple ≥ aprendido) y el equipo Misa-Amane lo midió: pesos por hallazgo peores que uno global en mitades de gold.
   Única excepción permitida, fijada a priori: si en OOF limpio un miembro propio supera al resto en un hallazgo por ≥ 0.02 AUC en los 5 folds,
   se puede usar su columna con w 0.5 en ese hallazgo (caso "Lateral OA" de Misa-Amane). Se registra como experimento propio.
4. **Medir la correlación de errores con OOF del volumen** (CPU, script sobre `results/*/oof*.csv` del volumen):
   - Para cada par de miembros propios y cada hallazgo: **Spearman de los scores condicionado a la clase** (por separado en positivos y en
     negativos; es el ρ que usa el toy binormal) y **Q de Yule / tasa de desacuerdo** de los errores al umbral de Youden. Reportar media
     ponderada por hallazgo y bootstrap por estudio (1 000 réplicas).
   - Para miembros públicos u OAI-init (sin OOF limpio): ρ **inferido del LB** con 2 envíos fijos (solo + blend a w fijo) y el toy de A4.
     Es la única medida honesta; declararla como inferida.
   - Decisión: un miembro entra al bloque si ρ_clase < 0.90 con el resto o si su AUC solo está a ≤ 0.010 del mejor.
5. **Gold (58)** solo veta (cae > 0.02 vs padre → fuera); no elige w ni miembros. Pseudo OOF decide dentro de lo propio.
6. **Decisión entre familias en LB:** Δ ≥ +0.003 contra el núcleo **y** ≥ 2 envíos coherentes (p. ej., dos semillas o dos w) antes de que
   una familia sea Final A. Δ < 0.003 = empate → se elige por prior (más diversidad, menos runtime, sin OAI extra).

## 3. Calendario 15 → 22-oct

Presupuesto RunPod: saldo ~$53 → **≤ 45 GPU-h planificadas** (~$33) + reserva; parar si saldo < $4. Pods ≤ 5 a la vez.
**Congelación de entrenamiento: 19-oct 12:00 UTC.** Desde ahí solo kernels y reenvíos.
Los envíos los dispara `rsna-cron` a las 00:05 UTC; el score llega ~4–5 h después. Cada día la cola se escribe antes de las 23:30 UTC.

### 15-oct (miércoles)
- **Pods (5):** P1 `ft-nartaa-aug` s42 all-data (3 h) · P2 `ft-r7-aug` s42 all-data (1.5–2 h, antes verificar formato de pesos R7) ·
  P3 exp066 folds 1–2 · P4 exp066 folds 3–4 · P5 auditoría de etiquetas LLM (CPU/API; si hay etiqueta mejor que consenso en dev-200, se
  entrena f0 de exp045 con ella, 2.5 h). ≈ 12 GPU-h.
- **Kernels:** kernel maestro `ens-r7-core` con slots de miembros por configuración (lista de datasets + w), caché de DICOM compartido,
  2 T4 en paralelo, fallback al núcleo si un miembro falla (patrón de gokdenizby "tiers").
- **Envíos 00:05:** la cola del 15 la escriben otros agentes hoy (repro R7+Raptor y variantes). Este plan no la toca; solo lee los scores.
- CPU: script de correlación de errores (§2.4) sobre exp045/054/056/066-f0 OOF.

### 16-oct
- **Pods:** P1 `ft-nartaa-aug` s7 **con la mejor etiqueta del 15** (une #1 y #3; si no hay etiqueta mejor, consenso+L5) · P2 `ft-r7-aug` s7
  idem · P3 exp066 all-data · P4–P5 libres (o 2.ª semilla de etiqueta nueva en f0 si el f0 del 15 promete Δ ≥ +0.003). ≈ 9 GPU-h.
- **Envíos (una pregunta cada uno):**
  1. `ft-nartaa-aug` s42 **solo** → ¿el miembro OAI-init con nuestras etiquetas llega a ≥ 0.950? (fija w por regla §2.1)
  2. `ft-r7-aug` s42 **solo** → ¿igual para la familia compact?
  3. núcleo 0.85 + rank(`ft-nartaa-aug` s42) 0.15 → ¿aporta la familia 1 sobre 0.957?
  4. núcleo 0.85 + rank(`ft-r7-aug` s42) 0.15 → ¿aporta la familia 2?
  5. núcleo **solo, reenvío** (si el 15 ya lo tiene, R7 solo) → ancla de ruido del día / LB de R7 solo para el toy.

### 17-oct
- **Pods:** solo lo que haya quedado de semillas; si familia 1 o 2 dio Δ ≥ +0.003 en (3)/(4), 3.ª semilla de esa familia. ≤ 6 GPU-h.
- **Envíos:**
  1. núcleo (1 − w) + bloque{ft-nartaa-aug, ft-r7-aug} s42 a w por regla → ¿el bloque combinado suma más que cada familia?
  2. mismo con semillas s42+s7 → ¿pagan las semillas / la etiqueta nueva? (coherencia: debe ≥ envío 1 − 0.002)
  3. núcleo 0.90 + exp066 (all-data o 5f) 0.10 → ¿la diversidad ImageNet compact aporta algo sobre el núcleo?
  4. **Plan B sin OAI** v1: exp056 + exp066 + (R7 solo si su dataset no usa OAI) → ¿cuánto vale nuestro mundo limpio?
  5. **Eficiencia** v1: candidato B (Ch nartaa 224 half o exp057 propio) → runtime real repetido.

### 18-oct
- **Pods:** última semilla si hace falta; si nada aportó, P1 `ft-nartaa-aug` 448 o etiqueta nueva sobre núcleo gemelo (una variable). ≤ 6 GPU-h.
- **Envíos:**
  1. Candidato A v2 (el mejor de 17-1/17-2) con el w siguiente de la regla → ¿w más alto empeora? (no hay otro w)
  2. Candidato A con TTA espejo solo en miembros propios → ¿la TTA paga en el bloque? (Shanmugam: puede restar)
  3. Plan B sin OAI v2 (w alternativo fijado a priori 0.5/0.5 vs 0.7/0.3 del 17)
  4. Eficiencia v1 **reenvío** (varianza de runtime)
  5. Eficiencia v2 (si exp057 vs Ch está en empate de score, el propio)

### 19-oct (congelación 12:00 UTC)
- **Pods:** apagar todo tras la última semilla. Reserva de saldo.
- **Envíos:** 1–2 Candidato A final (miembros congelados) dos veces seguidas · 3 Candidato A-alt (núcleo + solo la familia más fuerte)
  · 4 Plan B sin OAI final · 5 Eficiencia final.

### 20–21-oct
- Sin entrenamiento. Envíos = **reenvíos** de A, A-alt, B-eficiencia y sin-OAI (detectar roturas del entorno, medir runtime) + 1 reserva/día.
- 21-oct 18:00 UTC: decisión escrita de las finales con los números de `scores.csv`.

### 22-oct (cierre 23:59 UTC)
- Sin envíos nuevos salvo emergencia. Marcar las 2 finales antes de las 20:00 UTC.

### Elección de las 2 finales
- **Final A (AUC máxima robusta):** el candidato A si cumple Δ ≥ +0.002 contra el núcleo en ≥ 2 envíos (media de sus reenvíos) **y** runtime
  ≤ 7 h medido. Si es empate (< 0.002): **igual preferir A** sobre el núcleo puro (rompe el empate con cientos de cuentas que enviarán
  R7+Raptor tal cual; riesgo simétrico y pequeño, sesgo esperado ≥ 0 si ρ ≤ 0.9, A4 §3.4). Si A < núcleo − 0.002 dos veces → núcleo puro.
- **Final B (eficiencia):** mínimo de −AUC/0.46 + t/32400 con t = tiempo hasta score medido ≥ 2 veces (0.001 AUC ≈ 70 s). Preferir el
  propio (exp057/exp055) sobre pesos nartaa si está en empate de score: evita OAI y deja el paper limpio.

## 4. Riesgo OAI / datos externos y plan B

- **Regla (rules §6):** externos permitidos si son públicos, accesibles a todos y sin coste, o "razonablemente accesibles / coste mínimo".
  OAI es gratis con registro + acuerdo de uso. Los pesos nartaa son dataset público de Kaggle (licencia "other": uso de investigación,
  **incluida la competición**, sin derechos sobre datos OAI). R7: **verificar el 15** si su entrenamiento usa OAI o pesos nartaa como init.
- Exposición ya existente: blend-mine, velci, R7+Raptor y ~400+ cuentas dependen del Raptor nartaa. Enviar no descalifica; el riesgo cae sobre
  los **ganadores** (obligación de documentar y reproducir). Para premio de eficiencia el riesgo es mayor (somos candidatos a documentar).
- **Vigilancia:** revisar el foro de la competición (host/RSNA) cada día por anuncios sobre OAI; congelar decisión el 21.
- **Plan B si OAI no se permite:** Final A = mundo limpio (exp056 + exp045 5f + exp054 5f + exp066 + R7 si R7 no depende de OAI), pesos
  iguales por familia; Final B = exp057/exp055 propio. Por eso el kernel "sin OAI" se envía ≥ 2 veces antes del 20 (17-4, 18-3, 19-4):
  si el anuncio llega tarde, ya sabemos su LB y su runtime.
- Escritura del paper: cualquier miembro OAI-init se declara como "fine-tune de pesos públicos supervisados con OAI"; los claims de método
  salen de los modelos ImageNet-init.

## 5. Cuota GPU Kaggle y runtime ≤ 9 h

- **Runtime del rerun oculto:** blend-mine/R7+Raptor ≈ 3.5–4 h hasta score (compact ≈ 3 h de eso; Raptor 384 + TTA ≈ 0.5 h).
  Cada miembro CoAtNet 384 extra ≈ +0.5 h con TTA (≈ +0.25 h sin), cada compact extra ≈ +2.5–3 h → **un solo compact extra pone el
  kernel en ~7 h**. Límite interno: **≤ 6.5 h estimadas** (margen de 2.5 h a las 9 h por cola/arranque).
- Mitigaciones, por orden: (1) **DICOM decodificado una vez** y compartido por todos los miembros (I/O = 41–70 % del runtime);
  (2) **2 T4 en paralelo**: núcleo en GPU0, bloque propio en GPU1; (3) fp16 y sin TTA en los miembros propios salvo que 18-2 diga lo contrario;
  (4) compact extra (ft-r7-aug) **compartiendo el preprocesado del R7** (misma entrada) — su coste marginal es el forward, no la lectura;
  (5) presupuesto por miembro medido en el run de 3 estudios × factor 1300/3 antes de encolar; (6) fallback: si el tiempo acumulado
  supera 7.5 h, escribir el submission del núcleo y salir (nunca timeout = 0 puntos).
- **Cuota semanal GPU de Kaggle (~30 h):** **supuesto no verificado:** el rerun de scoring no consume cuota del usuario (confirmar el 15 mirando la cuota antes/después de un envío); sí la consumen los "Save & Run" de cada versión
  de kernel (3 estudios visibles, ~3–5 min) y cualquier entrenamiento en Kaggle. Regla: **cero entrenamiento en Kaggle** (todo en RunPod),
  versiones de kernel con `quick-save` cuando solo cambia la config de pesos, y un kernel maestro parametrizado en vez de un kernel por envío.
  Revisar la cuota restante cada mañana; si < 5 h, solo reenvíos de versiones ya guardadas.

## 6. Qué no se hace

Barrido de pesos en LB; stacking por hallazgo; pesos optimizados contra miembros públicos; DINO/3D/backbones nuevos; TENT/BN-adapt;
pseudo-labeling del test; más augs de intensidad. Cualquier idea fuera de §1 se anota en `08_backlog_priorizado.md` como P2.

## 7. Gates SDD pendientes

Cada entrenamiento (`ft-nartaa-aug`, `ft-r7-aug`, exp066 5f/all-data, etiqueta nueva) necesita número global con
`nuevo_experimento(tag, padre=...)` y SPEC §1–2 confirmada por Rodrigo **antes** de lanzar el pod. Padres propuestos: `ft-nartaa-aug` ← exp044;
`ft-r7-aug` ← núcleo R7 (nuevo T1); etiqueta nueva ← exp045. Criterio a priori para los fine-tunes OAI-init (sin OOF limpio): LB solo
≥ 0.950 y envío en bloque a 0.15 ≥ núcleo − 0.001.

## 8. Fuentes web consultadas (14-oct)

- heliosli, "≈0.957 | Latest dataset | score in one click" (solo título accesible): https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click
- gokdenizby/rsna-knee-abnormality (tiers con fallback, rank por etiqueta, pesos por gold-58 sobreajustan: 0.949): https://github.com/gokdenizby/rsna-knee-abnormality
- nathirad/rsna-knee-solo (Misa-Amane: reader 2.5D a 0.30 no mueve el LB; pesos por hallazgo peores; gold-58 no sirve para elegir blends): https://github.com/nathirad/rsna-knee-solo
- Kaggle efficiency LB: https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb
- No se encontró ningún notebook ni discusión pública > 0.957.
