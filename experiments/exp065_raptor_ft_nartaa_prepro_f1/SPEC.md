# exp065_raptor_ft_nartaa_prepro_f1

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp044_raptor_ft_nartaa (all-data: SWA [3,4,5] gold 0.91924; init nartaa gold 0.91907; 1.05 GPU-h; corpus crudo d96/384) |
| Fecha | 2026-10-13 |
| Rama | exp065_raptor_ft_nartaa_prepro_f1 |
| Estado | didnt (no corrido) |

Idea: Track B (desde 0.954): exp044 + preprocesado ganador de la Fase 1 (all-data); se omite si no hay ganador.

Track B = entrenar a partir del ganador público 0.954 (blend-mine: nartaa CoAtNet-2 Raptor SWA + compact convnext_small,
rank blend 0.5/0.5). Pedido explícito de Rodrigo (2026-10-13). Hermano: exp064 (misma receta + aug de artefactos light).

## 1. SPEC (antes de tocar el notebook)

**Regla de elección del preprocesado (fijada aquí, 2026-10-13 ~04:05 UTC, antes de cualquier resultado de la Fase 1):**
se usa la salida de `orq/p7c_chain/p7c_decide.py phase1` (regla de la Fase 1 escrita por la cadena p7c antes de resultados;
misma regla que elige el padre de exp062/exp063): `parent` = brazo `promising` con 2 semillas y mayor media de pseudo entre
P1 `exp058_prepro_clahe`, P2 `exp059_prepro_nyul` y `exp061_prepro_kneecrop`.
- `parent = exp058_prepro_clahe` → corpus CLAHE (clip 2.0, tiles 8×8, `build_corpus_clahe.py` de exp058, no ajusta nada).
- `parent = exp059_prepro_nyul` → corpus Nyúl–Udupa v2 con landmarks ajustados en **todo el train no-gold** (all-data;
  `orq/p7c_chain/build_corpus_nyul_fold.py --fit_fold 99`: el fold 99 no existe, así que solo se excluye gold/−2); landmarks
  guardados en `experiments/exp065_raptor_ft_nartaa_prepro_f1/nyul_landmarks_alldata.json` (los necesita el kernel).
- `parent = exp061_prepro_kneecrop` → corpus con recorte centrado en la rodilla reconstruido desde DICOM
  (`build_corpus_kneecrop.py` de exp061 + `check_corpus_kneecrop.py`).
- `parent = exp045_raptor_aug_espacial` (ningún brazo promising) → **este experimento no corre**; se registra como
  `didnt` ("no corrido: la Fase 1 no dejó preprocesado ganador") y se anota en `orq/p7c_prepro/status.log` ("[trackb]").
- P3 N4 no entra (igual que en la regla de la Fase 1).

**Hipótesis:** si el preprocesado de intensidad/recorte que ganó en la Fase 1 (donde mejoró la pseudo AUC desde ImageNet)
se aplica también al fine-tune de los pesos de nartaa, el modelo afinado aporta más al blend-mine (raptor = rank mean de
nartaa + este) que exp044 con corpus crudo, y el LB público sube.

**Única variable cambiada:**

| Parámetro o celda | Padre (exp044) | Este experimento |
|---|---|---|
| Corpus de entrenamiento (`CORPUS`, `PREPRO`) | `/root/corpus96_384` crudo (p2–98 uint8) | corpus del brazo ganador de la Fase 1 (regla de arriba) |

Todo lo demás igual a exp044: init pesos SWA de nartaa (carga estricta), all-data (4349 no-gold), 6 épocas, LR 1e-5 / 1e-4,
SWA épocas fijas [3, 4, 5], recorte central 320, K 12, bs 8, semilla 42, etiquetas de consenso de exp031, sin aug nueva.
Nota: nartaa se entrenó sobre corpus crudo; el cambio de distribución de entrada al afinar es parte de la variable.

**Contaminación (explícita; condiciona qué métrica decide):** los pesos de nartaa vieron **todo** el train (all-data) y
gold-58 (peso 3 / selección). La pseudo AUC de fold 0 y la gold de cualquier modelo inicializado con nartaa están
contaminadas: no miden generalización. Métricas internas = solo sanidad. Pesos con supervisión externa OAI (revisar reglas
de datos externos antes de un envío final).

**Efecto esperado:** Δ LB del blend ≈ 0 a +0.003. Coste ≈ padre (+ construcción del corpus: CLAHE ~1 min, Nyúl ~7 min,
kneecrop ~1 h de CPU).

**Criterios (no cambian después de correr):**
- **Decide el LB público** con el kernel `ens-exp065` (blend-mine estilo exp042b: grupo raptor 0.5 = rank mean de nartaa +
  exp065 `swa_alldata.pt`, con el preprocesado inyectado en la inferencia **solo para el miembro exp065**, idéntico al de
  entrenamiento — `orq/p7c_chain/make_kernel_p7c.py`; grupo compact 0.5) contra `ens-exp042b` (mismo kernel con exp044).
  - `worked` si LB(ens-exp065) ≥ LB(ens-exp042b) + 0.003 **y** ≥ 0.954.
  - `didnt` si no.
- **Sanidad (no decide; contaminada):** gold del SWA all-data ≥ 0.90 y ≥ 0.89924 (gold SWA padre − 0.02). Si falla →
  `killed` y no se construye kernel.
- Coste: GPU-h ≤ 1.5 × presupuesto.
- Sin envío desde aquí: el kernel validado queda como candidato en `orq/ESTADO.md` para las colas del 15/16-oct.

**Presupuesto:** ≈ 1.2 GPU-h (+ ≤ 1 h de CPU del pod si es kneecrop) · RTX PRO 4500 ($0.72/h) · ≈ $1–2.

**Riesgos:** desajuste entre la distribución de entrada de nartaa (crudo) y el corpus nuevo (6 épocas pueden no bastar para
adaptarse); diferencias train/inferencia en el preprocesado (mitigado: misma función, selftest de make_kernel_p7c); kneecrop
añade tiempo de inferencia en Kaggle (segunda reconstrucción del volumen); ruido del LB público.

## 2. PLAN

- Celda 0 (parámetros, celda 2 del .ipynb): `PREPRO` (nuevo, informativo) y `CORPUS` (el worker los pasa con papermill
  `-p CORPUS <dir> -p PREPRO <tipo>`), `NOTES`.
- Celda 6 (metrics, celda 14 del .ipynb): `corpus.prepro` = `build_meta.json["prepro"]` del corpus usado, y `prepro`.
- Ninguna otra.
- Disparo automático: `orq/trackb/daemon_b.sh` (rsna-cron) consulta `p7c_decide.py phase1` cada 10 min; cuando
  `ready=true` despliega pod (≤ 4 pods GPU en total, saldo ≥ $15) y corre `orq/trackb/worker_b.sh ... <clahe|nyul_all|kneecrop>`:
  construcción + checks del corpus → smoke → all-data → sanidad → podStop. Luego dataset privado `rsna-knee-exp065-weights`
  + kernel privado `ens-exp065` con el preprocesado inyectado; COMPLETE + `check_kernel.sh`. Sin envío.

## 3. VERDICT (números de `results/exp065_raptor_ft_nartaa_prepro_f1/alldata/metrics.json` y LB)

**Resultado:** didnt (no corrido) · **Fase alcanzada:** spec (armado)

Orquestador, 13-oct 09:00 UTC: la regla de la SPEC (corre solo si la Fase 1 deja un brazo `promising`) no se cumplió. Decisión de la cadena p7c (07:47 UTC): padre = exp045 (CLAHE Δ −0.0006, Nyúl Δ −0.0012, recorte Δ −0.0041 vs ref 0.8880; todos `didnt`). daemon_b: `B2 SKIP ... not run (verdict: didnt, no corrido)`. Sin GPU-h, sin kernel, sin LB.

| Métrica | Padre (exp044 / ens-exp042b) | Este (exp065 / ens-exp065) | Δ |
|---|---|---|---|
| Preprocesado elegido por la regla | crudo | | |
| LB público blend-mine (decide) | | | |
| gold SWA all-data (sanidad, contaminada) | 0.91924 | | |
| GPU-h | 1.05 | | |

**Qué aprendimos (2-4 líneas):** ningún preprocesado de la Fase 1 superó al corpus crudo; no hay variante de corpus que probar sobre nartaa.

**Siguiente paso sugerido:** ninguno con esta variable; Track B se juzga con el LB de ens-exp064 (B1).
