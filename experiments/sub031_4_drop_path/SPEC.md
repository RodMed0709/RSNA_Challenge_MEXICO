# sub031_4_drop_path

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp031_raptor_etiquetas_consenso |
| Fecha | 2026-10-12 |
| Rama | sub031_4_drop_path |
| Estado | run (fold 0) |

Idea: sub-experimento de afinado de exp031 (pseudo L2 0.8836). exp033 (24 épocas) tocó techo en la época 14 y luego
bajó: hay sobreajuste al final del entrenamiento. Regularizar el backbone con stochastic depth puede subir la pseudo L2.

## 1. SPEC (antes de tocar el notebook)

**Hipótesis:** `drop_path_rate = 0.2` en el backbone CoAtNet sube la pseudo AUC L2 en fold 0 en ≥ +0.003 contra exp031
sin que gold caiga más de 0.02.

**Única variable:** `drop_path_rate` del backbone `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`: **0.0 → 0.2**.
Valor por defecto comprobado en el pod (timm 1.0.30): `MaxxVit.__init__(drop_path_rate=0.0)`; el modelo del padre no
tiene ningún módulo `DropPath`. Con 0.2 timm crea 39 `DropPath` con probabilidad lineal 0.009 → 0.200 por bloque. Como
el defecto es 0, se usa el 0.2 pedido (no "defecto + 0.1"). Solo afecta al entrenamiento (en `eval()` DropPath es la
identidad): inferencia, coste y pesos cargables idénticos.

**Todo lo demás idéntico al padre:** etiquetas de consenso (Raptor + L2 + L3 + L4, promedio por celda), corpus 96×384
en `/root/corpus96_384`, recorte central 320, `RaptorClassifier` (dropout 0.2 de la atención igual), 16 épocas,
bs 8 × k 12 (k_eval 24), AdamW / OneCycle / LRs, bf16, semilla 42, `split_v1_seed42` fold 0, selección por pseudo L2
(L2 binarizada > 0.5) en val fold 0, SWA top-3 por pseudo L2. Gold y v1 solo se reportan.

**Métrica principal:** pseudo L2 fold 0 de la mejor época. SWA top-3 se reporta.

**Criterios (fijos, no cambian después de correr):**
- `worked` (promete): pseudo L2 fold 0 ≥ exp031 + 0.003 (= **≥ 0.8866**) **y** gold no cae > 0.02 contra exp031
  (gold ≥ 0.8944). Umbral menor que el +0.005 habitual porque es un sub-experimento de afinado.
- `killed`: pseudo L2 fold 0 ≤ exp031 − 0.005 (= **≤ 0.8786**) **o** gold cae > 0.02 contra exp031.
- `didnt`: cualquier otro caso.
(Padre exp031, de `results/exp031_raptor_etiquetas_consenso/metrics.json`: pseudo L2 0.8836, gold 0.9144.)
La `decision_auto` del notebook usa el umbral estándar ±0.005; el verdict se aplica a mano con estos criterios.

**Presupuesto:** fold 0 ≈ 2.5 GPU-h (como el padre; DropPath no cambia el coste) en rsna-f (RTX PRO 4500 32 GB).

**Riesgos:** con 16 épocas y OneCycle la regularización puede retrasar la convergencia (pseudo más baja a igual nº de
épocas) en vez de quitar sobreajuste; la vara pseudo L2 tiene el sesgo hacia L2 ya declarado en exp031 (igual en padre
y en este, así que no afecta a la comparación).

## 2. PLAN

- Notebook `sub031_4_drop_path.ipynb` = copia del de exp031; cambios: celda 0 (`EXP`, `PADRE`, `DROP_PATH = 0.2`,
  `NOTES`), celda 3 (`build_backbone` pasa `drop_path_rate=DROP_PATH` a `timm.create_model`).
- Smoke: `SMOKE=True` (64 train / 48 val, 1 época) → `metrics_smoke.json`.
- Run real: `FOLDS=[0]`, `SMOKE=False`, en rsna-f.

## 3. VERDICT (números de `results/sub031_4_drop_path/metrics.json`)

**Resultado:** pendiente · **Fase alcanzada:** screen fold 0 en curso

| Métrica | Padre (exp031) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8836 (ep 15) | | |
| pseudo L2 fold 0 (SWA top-3) | 0.8835 | | |
| v1 fold 0 | 0.8368 | | |
| gold (veto) | 0.9144 | | |
| GPU-h | 2.49 | | |

**Qué aprendimos:**

**Siguiente paso sugerido:**
