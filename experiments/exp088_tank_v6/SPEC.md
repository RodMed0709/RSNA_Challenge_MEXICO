# exp088_tank_v6

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp085_tank_seeds (tanque all-data 4407: fine-tune 6 ép. de nartaa SWA 384 + aug exp045 + gold-58 duro en train; consenso exp031 de 4 fuentes; 4 semillas, ~1.06 GPU-h c/u) |
| Fecha | 2026-10-18 |
| Rama | exp088_tank_v6 |
| Estado | run (pendiente LB) |

Idea: exp085 con etiquetas de entrenamiento `consenso_v6.csv` (exp086: L5 + L6 Claude + adjudicación de desacuerdos;
gold-58 = gold dura) en vez del consenso de 4 fuentes (raptor, L2, L3, L4).

## Excepciones heredadas de exp085 (decisión de Rodrigo, 18-oct)

Regla 6 suspendida como en el padre: los 58 gold entran al entrenamiento con etiqueta dura, no hay validación ni pseudo ni
fold 0; **decide solo el LB público**. Las AUC gold que imprime el notebook son sobre estudios de entrenamiento (sanidad).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** las etiquetas de informe de exp086 (Claude L6 + adjudicación) son menos ruidosas que el promedio de las 4
fuentes públicas; con la misma receta tanque, las semillas entrenadas con ellas dan un miembro Raptor 384 mejor y el kernel
tanque sube en el LB.

**Única variable cambiada:**

| Parámetro o celda | Padre (exp085) | Este experimento |
|---|---|---|
| Etiquetas blandas de entrenamiento (celda de datos, `LABELS_V6`) | consenso exp031 = media por celda de raptor, L2, L3, L4 | `/workspace/users/rodrigo/orq/labels_claude/consenso_v6.csv` (exp086, final con READY.json) |

Todo lo demás idéntico a exp085: mismo notebook (copiado con `nuevo_experimento`), init SWA nartaa (sha 7e5315da…), 6
épocas, BB_LR 1e-5 / HEAD_LR 1e-4, SWA [3, 4, 5], aug exp045, 4407 estudios con los 58 gold dentro y etiqueta dura
(`GOLD_HARD` se mantiene; en consenso_v6 los gold ya son la gold dura, así que es idempotente), corpus d96/384, semillas
42, 7 (y 123, 2024 si da el tiempo), misma cadena (`orq/tank_v6/chain.sh` = la de exp085 con rutas de exp088).

**Validación del CSV antes de entrenar (si falla, no se entrena):** columnas `StudyInstanceUID` + 12 `LABELS`; 4407 IDs
únicos = IDs del consenso de exp085 ∪ gold-58 (mismo conjunto de entrenamiento que el padre); valores en [0, 1]; en los 58
gold, igual a la gold dura de `train.csv`. Lo comprueban asserts en la celda de datos y una ejecución en CPU de las celdas
0–6 antes de lanzar los pods.

**Efecto esperado:** LB +0.000 a +0.003 sobre el kernel tanque con las semillas de exp085. Coste ≈ 1.1 GPU-h por semilla.

**Kernel (lo construye el orquestador, pregunta fijada ahora):** variante de `tank-seeds` idéntica salvo que la media de
semillas usa las semillas de exp088 en vez de las de exp085 (mismo número de semillas en ambos si se compara; mismos pesos
de mezcla, mismo fallback).

**Criterios (no cambian después de correr):**
- **Decide el LB público:** `worked` si LB(kernel exp088) ≥ LB(`tank-seeds` con exp085) + 0.001; `didnt` en otro caso.
- Para que cuente: ≥ 2 semillas terminadas antes del congelamiento (19-oct 12:00 UTC) y validación del kernel COMPLETE.
- Sanidad (no decide): pérdida finita y decreciente, carga estricta del init, gold-58 impreso solo como sanidad.
- Una semilla cuyo fin proyectado pase de 11:45 UTC no se empieza; la que acabe después de 12:00 se descarta.

**Presupuesto:** ≤ 1.5 GPU-h por semilla, ≤ 6 GPU-h; 2 pods RTX PRO 4500 ($0.72/h); tope $10.

**Riesgos:** consenso_v6 da gold macro AUC más baja que el consenso actual en la versión v1 de exp086 (0.8367 vs 0.8948,
informativo; gold está en el entrenamiento de todos modos); la etiqueta final llega tarde y puede quedar solo para 2
semillas; sin señal interna (ruido LB ±0.004).

**Licencia / legal:** igual que exp085 (init nartaa, pesos y etiquetas fuera de git; kernel y dataset privados).

## 2. PLAN

- Notebook = el de exp085 copiado por `nuevo_experimento` y parcheado por `orq/tank_v6/make_nb_v6.py` (fuera del repo):
  celda 0 (título), celda de parámetros (`LABELS_V6`, `NOTES`), celda de datos (`soft` = consenso_v6, el de 4 fuentes queda
  como `soft_cons4` informativo; asserts de IDs y gold), celda de métricas (`labels_train`). Ninguna otra.
- Cadena por pod: copia corpus → setup → smoke (pod A) → semillas en serie → export
  `kaggle_export/exp088_tank_v6/swa_alldata_s<SEED>.pt` → quita guard_skip → podStop. Pod A: 42 (+123); pod B: 7 (+2024).

## 3. VERDICT (números de `results/exp088_tank_v6/alldata_s*/metrics.json` y LB público)

**Resultado:** pendiente LB
