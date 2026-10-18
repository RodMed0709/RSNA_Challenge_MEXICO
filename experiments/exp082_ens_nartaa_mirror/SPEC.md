# exp082_ens_nartaa_mirror

| Campo | Valor |
|---|---|
| Autor | rodrigo (agente eff3b) |
| Padre | exp080_seqhead_ctrl_mil |
| Fecha | 2026-10-18 |
| Rama | exp082_ens_nartaa_mirror (desde exp080_seqhead_ctrl_mil 278dd0f: exp080 aún no está en main) |
| Estado | spec |

Idea: track de eficiencia — ensamble del kernel final `exp080-ctrl224-fast` (LB 0.949) con el modelo público nartaa 384
"anatomical mirror" (LB 0.949 solo), media de rangos 0.5/0.5 por hallazgo. Sin entrenamiento.

## 1. SPEC (antes de tocar el kernel)

**Hipótesis:** dos modelos de LB 0.949 con arquitecturas/resolución/datos distintos (alumno 224 destilado + cabeza MIL de
exp080 vs CoAtNet-2 384 de nartaa con OAI y TTA espejo anatómico) son lo bastante diversos para que su media de rangos suba
el AUC del LB lo suficiente como para pagar el runtime extra de nartaa (Efficiency ≈ −2.13·AUC + runtime_s/32400 ⇒
0.001 AUC ≈ 70 s).

**Única variable cambiada:** añadir el brazo nartaa al kernel del padre.

| Parámetro | Padre (`exp080-ctrl224-fast`) | Este experimento (`exp082-ens-nartaa-mirror`) |
|---|---|---|
| Brazos | exp080 (224, 64 ventanas, slots 0–2) | exp080 idéntico + nartaa `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (CoAtNet-2 384, INPUT_RES 320, K 94, vista plana + espejo anatómico, autocast fp16), código verbatim del kernel público `nartaa/rsna-knee-0949-anatomical-mirror` |
| Lectura DICOM | 3 slots (66 cortes) | 5 slots (96 cortes) **una sola vez**; exp080 recibe `vol[:66], mask[:66]` (byte-idéntico al lector de 3 slots: el lector rellena slot a slot con LUT por serie) |
| Mezcla | rankpct del único brazo | 0.5·rankpct(exp080) + 0.5·rankpct(nartaa), por hallazgo; estudio sin nartaa (fallo o deadline) → solo rankpct(exp080) |

Todo lo demás igual: pesos exp080 all-data (sha e4ab6994…), 6 procesos de prep, ventanas en GPU, `cudnn.benchmark = False`
(el ajuste del padre; el módulo nartaa pone True al importarse y se restablece a False: única desviación respecto al kernel
público de nartaa, solo cambia la elección de algoritmo cuDNN).

**Peso de la mezcla fijado a priori: 0.5/0.5.** Fuente: los dos tienen el mismo LB público (0.949 exp080-ctrl224, ref 57330704;
0.949 nartaa según su kernel/dataset público), no hay evidencia pública previa de que uno deba pesar más, y está prohibido
ajustar pesos con fold 0 o gold. Media de rangos por hallazgo = la receta de blend de ambos kernels (`rankpct`).

**Trampa de fuga (declarada):** los pesos de nartaa se entrenaron con todo train (la descripción del dataset dice "Gold58 was
reused for selection"), así que fold 0 y gold están dentro de su entrenamiento. Cualquier AUC de gold/fold 0 del ensamble está
inflada y es **solo informativa**. Lo que decide es el LB.

**Medida previa al envío (bench, mismo protocolo que eff2):** kernel BENCH privado en Kaggle T4×2 sobre 1300 estudios de train
(58 gold + 871 fold 0 + 371 al azar, seed 0), pipeline idéntico al de envío. **Δt = bucle main del ensamble − 251.7 s**
(bucle de `exp080-bench-fastng6`). Comprobación: probabilidades del brazo exp080 en el bench idénticas a las de
`exp080-bench-fastng6` (max|Δ| esperado 0).

**Criterio LB (fijado con el Δt medido ANTES de cualquier envío; no cambia después):**
- `worked` si LB pública ≥ 0.949 + ceil_a_milésima(Δt / 70000) + 0.001.
- `didnt` si no. (No hay fold 0 que decida: sin entrenamiento; gold no veta porque está fugado en nartaa.)
- Valor numérico del umbral: se escribe en la sección 1b con el Δt del bench, antes de encolar.

**Presupuesto:** $0 RunPod (sin pod). Kaggle: 1 sesión GPU a la vez (bench ~25–40 min + validación del kernel).

**Riesgos:** el coste GPU de nartaa (94 ventanas × 2 vistas a 320 px vs 64 a 224 px ≈ 6× el FLOP del padre) puede dar
Δt ≈ 1000 s ⇒ umbral ≈ 0.965, casi inalcanzable; la ganancia de un ensamble de dos 0.949 suele ser +0.002–0.006.
Variabilidad entre máquinas Kaggle ±15 %. Runtime estimado con estudios de train.

**Licencia del dataset nartaa** (`nartaa/rsna-knee-publication-swa-weights-20261007`, licencia Kaggle "other"): pesos para
investigación/educación incluida la competición; desde 2026-10-13 permite adaptar y redistribuir con fines no comerciales con
crédito, **incluido licenciar una Submission ganadora bajo CC BY-NC 4.0** para las obligaciones de ganador; hay que conservar el
aviso/agradecimiento OAI. Compatible con premio si la competición acepta CC BY-NC 4.0 (a confirmar contra las reglas de
eficiencia antes de reclamar premio).

## 1b. Umbral numérico (rellenar con el bench antes de enviar)

_pendiente del bench_

## 2. PLAN

- Generador (fuera del repo, `orq/eff3b/make_ens_kernel.py`): toma `exp080-ctrl224-fast.ipynb` y el kernel público de nartaa;
  escribe el módulo nartaa verbatim a `/kaggle/working/nartaa_mirror.py` (`%%writefile`) y lo importa como módulo (su `main()`
  no corre); cambia en el cuerpo de exp080 solo: lector de 5 slots, carga del modelo nartaa por GPU, consumidor (exp080 sobre
  `vol[:66]`, después nartaa plano+espejo con try/except y deadline 8 h), mezcla de rangos con fallback por estudio.
- `exp082-bench-ens` (privado): mismo cuerpo + arnés de eff2 (raíz falsa con 1300 estudios de train) → `bench_report.json`.
- `exp082-ens-nartaa-mirror` (privado): kernel de envío; validación COMPLETE sin Traceback, submission 13 columnas sin NaN.
- `exp082_ens_nartaa_mirror.ipynb` de esta carpeta = el kernel de envío (no hay notebook de entrenamiento).
- CANDIDATE.json en `orq/eff3b/`; el orquestador encola. No se envía nada desde aquí.

## 3. VERDICT (números de `results/exp082_ens_nartaa_mirror/metrics.json`)

_pendiente del LB_
