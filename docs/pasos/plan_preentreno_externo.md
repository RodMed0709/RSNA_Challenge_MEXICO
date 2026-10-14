# Plan: preentreno / supervisión externa (14-oct-2026)

Estado SDD: **idea → falta SPEC**. Nada de esto se programa sin SPEC confirmada (padre propuesto: exp045).
Gold (58, fold -1) solo veta; decide pseudo fold 0 (Δ ≥ +0.005) y después el LB.

## 1. Cómo lo hicieron nartaa y R7 (fuentes primarias)

**nartaa (Raptor 0.949/0.945)**, descripción del dataset `nartaa/rsna-knee-publication-swa-weights-20261007`:
- "Both models used the same **2,399 baseline OAI knees** for **masked external supervision**".
- Etiquetas: **lecturas semicuantitativas MOAKS** de OAI, emparejadas con series de RM con ayuda de
  **3DReasonKnee** (Sambara et al. 2025, arXiv 2510.20967, CC BY 4.0; HF `rajpurkarlab/3DReasonKnee`, solo texto:
  BML, cartílago, osteofitos, Hoffa/efusión-sinovitis, extrusión/morfología de menisco, LCA/LCP, quiste poplíteo, bursas).
  No KL, no WORMS. "Masked" = pérdida solo en los hallazgos que tienen equivalente MOAKS (mapeo exacto no publicado).
- Secuencias: no se dice. 3DReasonKnee usa DESS sagital; nartaa entra con sus 5 slots (Sag/Cor/Ax), así que
  probablemente usó IW-TSE FS sag + DESS + cor (inferencia, no verificado).
- Selección con gold58 (lo declaran). Tabla, caché y notebook de entrenamiento: privados.
- **Términos (añadidos el 13-oct):** se permite fine-tunear/adaptar y redistribuir con fines no comerciales, incluida la
  competición, y licenciar una solución ganadora bajo CC BY-NC 4.0 con atribución + aviso OAI.

**heliosli R7** (`heliosli/rsna-knee-compact-r7cnxfsss`, kernel 0.957): ConvNeXt-S 4 folds, crop 352, **sin OAI directo**.
Usa **fastMRI+ como tarea auxiliar** (trainer v9.8 `--fm-aux`): en cada paso, un lote extra de 96 ventanas coronales de
3 cortes (store fm384, RSS) por el mismo encoder; cabeza densa de 11 clases (meniscus_tear, cart_partial, cart_full,
bone_edema, bone_fx_contusion, effusion, cyst, acl_high, acl_low, collateral, other); BCE densa (pos_weight 8) + presencia;
λ = 0.3. El kernel también monta `heliosli/rsna-orthofoundation-fold0-epoch7` (OrthoFoundation-L SSL, preentrenado con
OAI + fastMRI + datos internos, arXiv 2601.18250); su peso en el score no verificado. El blend 0.957 incluye Raptor nartaa (OAI).

## 2. Inventario

| Dataset | Tamaño / etiquetas | Solapa con los 12 | Licencia / acceso | Descarga | ¿Hoy? |
|---|---|---|---|---|---|
| OAI (NDA) + MOAKS | ~4.8k sujetos; MOAKS ~2.4k rodillas | ~10/12 (no fractura; MCL débil) | Gratis; login.gov/eRA + NDA + términos OAI; plazo no publicado | cientos de GB–TB | **No** (acceso + volumen + matching) |
| fastMRI+ (knee) | ~1k vol. coronales PD/PDFS, 16k cajas, 22 clases | ~8/12 (menisco, OA vía cartílago, contusión, fractura, efusión, quiste≈Baker, LCA, colateral) | fastMRI DUA NYU (investigación, no redistribuir) + etiquetas GitHub MS | singlecoil train+val ~110 GB [de memoria, verificar] (usar `reconstruction_rss`) | **Sí si NYU aprueba hoy-mañana** |
| MRNet (Stanford AIMI) | 1,370 exámenes, 3 etiquetas (anormal, LCA, menisco) | 2–3/12 | RUA AIMI, no comercial, clic | ~6 GB | Sí, pero es nuestra **validación externa del paper** (R5) |
| KneeMRI (Rijeka, Zenodo 14789903) | 917 sagitales, LCA 3 niveles | 1/12 | CC BY-NC-ND | <10 GB | Sí; poco valor |
| SKM-TEA (AIMI) | 155 exámenes, cajas 16 patologías | parcial | RUA AIMI | 1.6 TB (DICOM menor) | No (pocos casos) |
| MeniMV (arXiv 2512.18437) | 3k exámenes, grado menisco | 2/12 | por verificar | ? | No verificado |
| Pesos nartaa (OAI-init) | — | — | permitido en competición (13-oct) | 0.5 GB, ya en volumen | **Sí** |

## 3. Plan recomendado (8 días; congelar entrenamiento 19-oct)

1. **Hoy:** pedir acceso fastMRI (fastmri.med.nyu.edu) y MRNet (AIMI). No pedir OAI para esta competición.
2. **Vía rápida, ya:** `ft-nartaa-aug` (paso8 ítem 1) = init nartaa + receta exp045, all-data, 2 semillas. Es lo que
   mueve el LB (exp044 → blend 0.953 frente a 0.939–0.943 ImageNet). Para el paper se declara como init de terceros.
3. **Vía propia (paper):** expNNN `fm_aux` (padre exp045), una variable: tarea auxiliar fastMRI+ estilo R7 (λ 0.3,
   solo slot coronal). Store fm384 en un pod con disco de contenedor (no el volumen: cuota). Smoke → fold 0 (~3.5 GPU-h)
   el 16-oct; si Δ ≥ +0.005 pseudo y gold sin caída > 0.02, 5 folds en 5 pods el 17–18. Ganancia esperada +0.000 a +0.003
   LB (R6→R7 +0.003 en blend, confundido con stackers).
4. MRNet **no** se usa para entrenar: queda limpio como validación externa.

## 4. Riesgos

- Reglas §6: OAI/fastMRI/MRNet son gratis con registro → "reasonably accessible"; ningún anuncio del host visto (vigilar foro).
- Licencias: fastMRI y MRNet no comerciales; si ganamos, la licencia CC BY-NC de nartaa es compatible; CC0/MIT no.
- Fuga: NYU/Stanford podrían ser sitios del RSNA (mismo paciente en ambos); riesgo bajo, sin etiquetas del concurso.
  Gold nunca entra en la elección de λ ni de época.
- Literatura: RadImageNet (Mei 2022, doi:10.1148/ryai.210315) +4.5–4.8 % AUC en LCA/menisco con datasets pequeños; una
  reevaluación 2025 (arXiv 2508.17567) no lo reproduce → con 4.4k estudios esperamos poco del preentreno genérico.
