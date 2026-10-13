# exp060_prepro_n4

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp045_raptor_aug_espacial |
| Fecha | 2026-10-13 |
| Rama | exp060_prepro_n4 (desde exp045_raptor_aug_espacial 607644c, como los sub045_*) |
| Estado | spec |

Idea: Fase 1 del estudio de preprocesado (aprobado por Rodrigo 2026-10-13): exp045 + corrección de campo de sesgo N4 por pila de slot. Origen: Tustison et al. 2010, IEEE TMI 29(6) (N4ITK).

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** si el corpus de entrada pasa de uint8 percentil 2–98 por serie (padre) a uint8 con corrección de campo de sesgo N4 por pila de slot, la pseudo L2
fold 0 sube porque la inhomogeneidad de bobina (superficie de rodilla) cambia la intensidad del mismo tejido según la posición; N4 la aplana.

**Única variable cambiada:**

| Parámetro o celda | Padre | Este experimento |
|---|---|---|
| Preprocesado del corpus (celda 0 `CORPUS`) | `/root/corpus96_384` (build_corpus96.py) | `/root/corpus_n4` = corpus96_384 + corrección de campo de sesgo N4 por pila de slot (`build_corpus_n4.py`) |

Método: SimpleITK `N4BiasFieldCorrectionImageFilter` sobre float(uint8)+1 de la pila de cortes válidos de cada slot (máscara `OtsuThreshold`, shrink 4 en x, y y z —z solo si la pila tiene ≥ 8 cortes—, iteraciones y puntos de control por defecto); campo de sesgo evaluado a resolución completa y dividido; después percentil 2–98 de la pila → uint8.

Afecta a train, validación y gold (y obligaría a aplicarlo en la inferencia de Kaggle). Todo lo demás igual a exp045: split
`split_v1_seed42`, semilla 42, 16 épocas, etiquetas, aug espacial, SWA top-3, fold 0.

**Efecto esperado:** Δ pseudo fold 0 ≈ +0.003…+0.005; coste de entrenamiento × 1. Coste de preprocesado en inferencia: ≈ 27 s/estudio (medido en rsna-cron, 4 estudios, 1 hilo; shrink anisótropo 4,4,1 costaba ≈ 104 s).

**Criterios (fijos antes de correr; no cambian después de ver resultados):**
- **Referencia** = media de la pseudo L2 fold 0 (mejor época) de `sub045_5_seed7` (exp045 con semilla 7) y de exp045 semilla 42
  (0.8898899); ídem para gold (exp045 s42 = 0.9201799). Se calcula cuando termine sub045_5_seed7.
- **promising** (candidato a Fase 2 y a segunda semilla): Δ pseudo mejor época ≥ **+0.003** vs referencia **y** gold ≥ ref_gold − 0.02.
- **killed**: gold < ref_gold − 0.02 (diga lo que diga pseudo).
- **didnt**: cualquier otro caso.
- **Ronda 2** (automática si hay presupuesto): si Δ ≥ **+0.002** en ronda 1, se corre la semilla 7 de este brazo; la decisión final usa
  la media de las 2 semillas del brazo vs la referencia con el mismo +0.003 (y el mismo veto gold con la media).
- La decisión automática de la celda 5 (regla SDD estándar vs padre +0.005) es solo informativa; mandan estos criterios.
- Gold (58) solo veta; el checkpoint lo elige pseudo (`SELECT_ON = "pseudo"`). Coste de preprocesado en inferencia (ms/estudio,
  1 hilo) se registra en `metrics.json` (`corpus.prepro`) y aquí.

**Presupuesto:** smoke + screen fold 0 ≈ 2.7 GPU-h (+ 2.6 si hay ronda 2) · GPU barata con ≥ 200 GB de disco local (RTX PRO 4500 / 4090 / 5090).
Construcción del corpus en CPU del pod (disco local, no /workspace).

**Riesgos:** corpus mal construido (se revisa con el smoke y `build_meta.json`); contención de CPU si el constructor corre
a la vez que otro entrenamiento (solo afecta al tiempo); coste de inferencia en Kaggle (límite 9 h).

## 2. PLAN

- Constructor `build_corpus_n4.py` en esta carpeta → `/root/corpus_n4` en el disco local del pod, desde `/root/corpus96_384`
  (copia de `/workspace/data/cache/raptor96_384`); escribe `build_meta.json` con `prepro` (método, segundos, ms/estudio 1 hilo).
- Celdas que cambian: **0** (`CORPUS`, `NOTES`) y **6** (`metrics.json`: añade `corpus.prepro` = `build_meta["prepro"]`, solo registro).
  Ninguna otra.
- Smoke (`SMOKE = True`) y después fold 0 (`SMOKE = False`, `FOLDS = [0]`) con papermill, cola `orq/p7c_prepro/worker.sh`.
- Ronda 2 (si Δ ≥ +0.002): mismo notebook con `-p SEED 7` en el worktree `wt_exp060_prepro_n4_seed7`; su metrics.json se copia como
  `results/exp060_prepro_n4/metrics_seed7.json`.

## 3. VERDICT (números de `results/exp060_prepro_n4/metrics.json`)

Pendiente.
