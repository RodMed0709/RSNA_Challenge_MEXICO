# 08 — Backlog priorizado (tabla única)

Fecha: 2026-10-12. Consolida las ideas de 01–07 y de A2 §3, A5 Parte B, RESUMEN 01–05. Ganancias esperadas = **hipótesis razonadas**, no resultados.
GPU-h: screen f0 receta Raptor ≈ 2.5 (A2); pod ≈ $0.74/h (A5). Prioridad: **P0** = antes del 22-oct y barato; **P1** = antes del 22-oct si hay GPU; **P2** = después del 22-oct / paper; **P3** = no recomendado ahora.
Sirve a: **LB** / **EF** (eficiencia) / **PA** (paper). ★ = top-10.

| id | Idea | Cita (clave bib o fuente) | Tema | Ganancia esperada (razonada) | GPU-h | Riesgo | Sirve | Prio | Estado |
|---|---|---|---|---|---|---|---|---|---|
| ★ENS-3 | `update_bn` sobre SWA de exp045 + TTA espejo sí/no (ENS-8) | izmailov2018swa; shanmugam2021tta | ensamble | corrige BN de pesos enviados; dato de TTA decide exp057 | 0.4 | bajo | LB EF | P0 | idea |
| ★ENS-2 | Ensamble de rangos exp045 5f × exp054 5f, 0.5/0.5 fijo | fort2019deepensembles; rame2022diwa | ensamble | "no suman en un modelo" → deberían como ensamble; teacher de EFF-1 | 0.5 | bajo | LB EF | P0 | idea |
| ★ENS-1 | Fold soup de exp045 (uniforme → greedy) | wortsman2022soups; sanjeev2024fissionfusion | ensamble/eficiencia | AUC entre 1 modelo (gold 0.920) y ensamble (0.924) a 1/5 de coste | 0.5 | medio | EF PA | P0 | idea |
| ★EFF-1 | Destilar ENS-2 → CoAtNet 224/3 slots/sin TTA/fp16 (padre exp057) | hinton2015distilling; beyer2022kd; noothout2022kd | eficiencia | +0.002 a +0.006 sobre exp057 al mismo runtime | 4–5 | medio | EF PA LB | P0 | idea |
| ★LAB-2 | Dawid-Skene por hallazgo, fase 1 CPU vs dev-200 | dawid1979skene; raykar2010crowds; ratner2017snorkel | etiquetas | tabla de C1 gratis; fase 2 +0.000 a +0.004 | 0 (+2.5) | bajo | PA LB | P0 | DS ya ajustado (A3 §3.4) |
| ★LAB-1 | Consenso dedup L2/L4 (SPEC lista, A3 §4) | ratner2017snorkel | etiquetas | +0.000 a +0.003; 703 celdas cambian | 2.5 | bajo | LB PA | P1 | SPEC redactada |
| ★ENS-10 | Harness exp040 sobre exp046/sub046_1/exp047 | varoquaux2022failures | robustez | decide la prior de toda la línea de augs de intensidad | 0.6 | nulo | PA | P0 | idea |
| ★AUG-F1 | Afín sobre corte 384 y recorte 320 después | A2 §2.1; mrnet (bien2018mrnet) | aug | quita bordes negros de la única aug que funciona; +0.000 a +0.003 [inferencia] | 2.6 | bajo | LB | P1 | idea |
| ★AUG-ANI | Anisotropía/grosor de corte in-plane p0.25 | perezgarcia2021torchio; billot2023synthseg | aug RM | +0.000 a +0.003 pseudo; valor en sitio croata | 2.5 | medio | PA LB | P1 | ficha en 02 §3.1 |
| ★PAP-R1R2 | Bootstrap pareado + DeLong/Holm sobre OOF existentes | maierhein2018rankings; efron1979bootstrap | paper | requisito de revisión | CPU | nulo | PA | P0 | idea |
| EFF-3 | Curva presupuesto de cortes/secuencias (solo inferencia) | ma2024flare22 | eficiencia | elige punto de la frontera | ~1 | bajo | EF PA | P0 | idea |
| EFF-4 | Atacar I/O DICOM (paralelo, leer solo cortes usados) | ma2024flare22 | eficiencia | I/O = 41 % (exp025) → mayor palanca de runtime | 0 | bajo | EF | P0 | idea |
| EFF-5 | Enviar 2 veces cada candidato eficiente | A4 §3.1 | LB | mide varianza del runtime | 0 | nulo | EF | P0 | idea |
| PAP-R6 | Runtime oficial de cada punto de la frontera en LB | A5 R6 | paper | imposible después del 22-oct | envíos | nulo | PA EF | P0 | idea |
| ENS-12 | Medir ρ OOF exp045 vs Raptor nartaa | A4 §4 | ensamble | decide peso 0.15 vs 0.25 | CPU | nulo | LB | P0 | idea |
| ENS-7 | 2–3 semillas exp045 all-data | fort2019deepensembles | ensamble | +0.002 a +0.004 (medido +0.0042 pseudo con 2 semillas exp031) | 3/semilla | bajo | LB PA | P1 | idea |
| LAB-4 | Reponderación por OOF / CROWDLAB | goh2022crowdlab; karimi2020noisylabels | etiquetas | +0.000 a +0.004 (Lumbar 2.º +1 % LB) | 2.5 | medio | LB PA | P1 | idea |
| LAB-3 | Cabezas por fuente LLM | tanno2019annotator; raykar2010crowds | etiquetas | sustituto con fundamento de exp052 | 2.5 | bajo | PA | P1 | idea |
| ENS-4 | Cola de LR constante + SWA real (padre largo) | izmailov2018swa | ensamble | habilita re-prueba justa de regularizadores | 3.2 | bajo | LB | P1 | idea |
| AUG-LONG | exp045 a 20–24 épocas (sub045_2 rehabilitado) | A2 §2.11; theoviel Kaggle | entrenamiento | padre para regularizadores; exp033 SWA +0.0037 a 24 ép | 3.2–3.8 | bajo | LB | P1 | cancelado → reabrir |
| AUG-MOT | Motion k-space p0.15 | shaw2019motion; perezgarcia2021torchio | aug RM | ~0 in-dist; robustez | 2.5 | medio-alto | PA | P1 | ficha 02 §3.2 |
| AUG-BIAS | Bias field por slot p0.3 | chen2020advbias; billot2023synthseg | aug RM | ~0 in-dist; robustez | 2.5 | bajo-medio | PA | P1 | ficha 02 §3.5 |
| AUG-GAM | Gamma/contraste por slot p0.3 (re-prueba exp046) | A2 §2.3; zhang2020bigaug | aug | ~0 in-dist | 2.5 | bajo | PA | P1 | ficha 02 §3.8 |
| AUG-F2 | Afín fuerte p0.3 sobre 384 (tras F1) | AnyMC3D; Lumbar 1.º | aug | +0.000 a +0.003 [inferencia] | 2.6 | medio | LB | P1 | idea |
| AUG-F3 | Resize 384→320 en vez de recorte | A2 §2.7 | aug/prepro | separa FOV de resolución | 2.5 | bajo | LB PA | P2 | idea |
| AUG-F4 | `padding_mode="border"` en grid_sample | A2 §2.2 | aug | alternativa barata a F1 | 2.5 | bajo | LB | P2 | idea |
| AUG-GHO | Ghosting por slot p0.15 | perezgarcia2021torchio | aug RM | robustez | 2.5 | medio | PA | P2 | ficha 02 §3.3 |
| AUG-BIG | BigAug suave apilado (padre largo) | zhang2020bigaug | aug RM | dominio no visto; in-dist puede bajar | 3.5 | alto | PA | P2 | ficha 02 §3.6 |
| AUG-GIN | GIN (+IPA) p0.3 | ouyang2022gin | aug DG | robustez | 2.5 | medio | PA | P2 | ficha 02 §3.7 |
| AUG-SPK | Spike (solo dentro de BigAug) | perezgarcia2021torchio | aug RM | bajo | — | bajo | PA | P3 | ficha 02 §3.4 |
| AUG-MIX | Mixup en espacio de rasgos / alineado por slot (padre largo) | zhang2018mixup; A2 §2.5 | aug | ~0 a +0.002 | 3.5 | medio | LB | P2 | idea |
| AUG-DP | drop_path 0.2 sobre padre largo | huang2016stochasticdepth | regularización | solo con sobreajuste | 3.5 | bajo | LB | P2 | idea |
| ARC-1 | Caché de embeddings congelados por fold | lu2021clam; Kaggle C-spine 3.º | arquitectura | habilita ARC-2..7 en minutos | 5 | bajo | PA | P1 | idea |
| ARC-2 | Embedding de slot aprendible | inferencia | arquitectura | prerequisito de slot dropout / mixup alineado | 2.5 | bajo | LB PA | P1 | idea |
| ARC-3 | Atención gated (Ilse) | ilse2018attentionmil | arquitectura | +0.000 a +0.002 [inferencia] | 2.5 | bajo | LB | P2 | idea |
| ARC-4 | Agregación jerárquica corte→secuencia→estudio | qiu2024copas; AnyMC3D | arquitectura | +0.000 a +0.004; interpretabilidad | 2.5 | bajo | PA | P2 | idea |
| ARC-5 | Cross-attention CoPAS (sagital query) | qiu2024copas | arquitectura | baseline R9 | 2.5 | medio | PA | P2 | idea |
| ARC-6 | biGRU por slot + aux atención | Kaggle Lumbar 1.º | arquitectura | 0.01–0.02 logloss en Kaggle; aquí ~0 a +0.004 | 2.5 | medio | LB PA | P2 | idea |
| ARC-7 | Fusión HeMIS | havaei2016hemis | arquitectura | secuencias faltantes | 2.5 | medio | PA | P2 | idea |
| ARC-8 | Bloques 3D inicializados a cero | vuskov2025multicondition | arquitectura | contexto 3D barato | 3 | medio | LB PA | P2 | idea |
| ARC-9 | FPN / pooling multiescala | MRPyrNet (Dunnhofer 2021) | arquitectura | lesiones pequeñas | 2.5 | bajo | LB | P2 | idea |
| ARC-10 | Recorte ROI + jitter antes de 224 | Xie 2025; Kaggle Lumbar 1.º | eficiencia/arquitectura | +0.002 a +0.006 en modelo eficiente | 3–4 + cache | alto | EF PA | P2 | idea |
| ARC-11 | FM congelado (DINOv3 / MRI-CORE) + agregador | simeoni2025dinov3; Dong 2025 | arquitectura | prior baja LB (exp006 killed); baseline | 1–6 | bajo | PA | P2 | idea |
| ARC-12 | Edad/sexo al embedding | MeniOmni; OrthoDiffusion | arquitectura | OA/degenerativos | ~0 | verificar | LB | P2 | verificar metadato |
| ARC-13 | SSL en dominio (MAE) | KneePreM; SB-SSL | arquitectura | clases raras | días | alto | PA | P3 | idea |
| LAB-5 | 6.ª fuente LLM de otra familia | plazadelarco2023llmcrowds | etiquetas | +0.000 a +0.003; punto en curva C1 | API + 2.5 | medio | PA | P1 | idea |
| LAB-6 | Estado "incierto" explícito | irvin2019chexpert | etiquetas | — | API + 2.5 | medio | PA | P2 | idea |
| LAB-7 | Afinar solo con celdas unánimes | smit2020chexbert | etiquetas | — | 1 | medio | LB | P2 | idea |
| LAB-8 | Pérdida no lineal en y (peso por acuerdo) | peterson2019human | etiquetas | cambia gradiente esperado | 2.5 | medio | PA | P2 | idea |
| ENS-5 | SWAD | cha2021swad | ensamble | +0.000 a +0.002 | 2.5 | bajo | LB | P2 | idea |
| ENS-6 | WiSE-FT con checkpoint temprano | wortsman2022wiseft | ensamble | robustez a shift | ~0 | medio | LB | P2 | idea |
| ENS-11 | Calibración ECE/Brier + temperature scaling | guo2017calibration; ovadia2019uncertainty | robustez | ECE 0.20 → menor | CPU | bajo | PA | P2 | idea |
| ENS-9 | BN-adapt / TENT en test | wang2021tent; karani2021tta | robustez | incierta | inferencia | alto | — | P3 | descartar antes del 22-oct |
| EFF-6 | INT8 TensorRT/ORT | MedPTQ (Qu 2026) | eficiencia | menor vs fp16 | medio | alto | EF | P3 | después |
| EFF-7 | Student ligero (RepViT/FastViT) | Wang 2024; Vasu 2023 | eficiencia | si domina I/O, nulo | medio | medio | EF | P2 | perfilar antes |
| EFF-8 | Cascada / early exit | Agarwal 2023 | eficiencia | pequeño con 12 etiquetas | medio | alto | EF | P3 | idea |
| PAP-R3 | ≥3 semillas exp031/045/054 | bouthillier2021variance | paper | valida no aditividad | ~10 | bajo | PA | P2 | idea |
| PAP-R5 | Validación externa MRNet/KneeMRI/fastMRI+ | bien2018mrnet | paper | requisito de revisión | ≤1 | medio (acceso) | PA | P2 | **pedir acceso ya** |
| PAP-R8 | Sensibilidad por sitio/idioma | varoquaux2022failures | paper | croata 0.817 vs inglés 0.880 | CPU | bajo | PA | P2 | idea |
| PAP-MR | Mini-Roelofs: ρ interno↔LB sobre registro | roelofs2019metaanalysis | paper | resultado publicable | CPU | bajo | PA | P2 | requiere merge del registro |

**Conteo:** 60 ideas (10 ★).

## Top-10 (★) en orden de ejecución

1. ENS-3 (`update_bn` + TTA sí/no) — 0.4 GPU-h, puede corregir pesos ya enviados.
2. ENS-10 (harness exp040 sobre ablaciones de intensidad) — 0.6 GPU-h, fija la prior de las augs de RM.
3. PAP-R1R2 (bootstrap + DeLong) — CPU.
4. LAB-2 fase 1 (DS vs dev-200) — CPU.
5. ENS-2 (ensamble exp045 × exp054) — teacher.
6. ENS-1 (fold soup) — eficiencia.
7. EFF-1 (destilación a 224/3 slots) — la apuesta principal de eficiencia.
8. AUG-F1 (afín sobre 384) — arregla lo que funciona.
9. AUG-ANI (anisotropía) — primera aug de RM; abre la ablación del paper.
10. LAB-1 (consenso dedup) — SPEC ya lista.

## No hacer antes del 22-oct (A5)
Más intensidad/CLAHE/normalización sin cambio de diseño; backbones más grandes o ViT fine-tuned; INT8; pseudo-labeling del test; TENT.
