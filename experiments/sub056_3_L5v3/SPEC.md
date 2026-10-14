# sub056_3_L5v3

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | sub056_2_L5v2 (exp056 combo con L5v2 como 5.ª fuente) |
| Fecha | 2026-10-12 |
| Rama | sub056_3_L5v3 (desde sub056_2_L5v2, d5af1da) |
| Estado | archivado sin correr: sub056_2 = `didnt` (Δ +0.0017 < +0.003), la condición no se cumplió |

Idea: tercera ronda de etiquetas LLM. Si 5 lecturas (L5v2) mejoran a 3 (L5), probar 7 lecturas, poniendo un modelo
más fuerte en los hallazgos más ruidosos.

## 1. SPEC (antes de tocar el notebook)

### Fase 1 — generar L5v3

| Lectura | Modelo | Qué | Estado |
|---|---|---|---|
| A, B, C, D, E | las de L5v2 | caché (`/workspace/users/rodrigo/l5v3/cache/`) | no se pagan |
| **F_focus3** | **gpt-5.4** (razonamiento `none`) | solo Contusion, Synovitis y Baker's: definiciones afinadas a cómo se describen de forma indirecta (sinovitis ≠ derrame solo; Baker ≠ quiste parameniscal; contusión ≠ edema degenerativo). Probabilidad 0–100. Las 3 más ruidosas: menor AUC contra el consenso y contra gold en L5/L5v2. | nueva |
| **G_quote** | gpt-5.4-mini (`none`) | los 12 hallazgos: frase textual más corta que decide + probabilidad 0–100 | nueva |

L5v3 = media por celda de las lecturas presentes (Contusion/Synovitis/Baker's con 7 lecturas, el resto con 6). Una
lectura nueva entra en la media solo si cubre ≥ 99 % de los estudios. Salida: `/workspace/data/labels/ours/L5v3_<fecha>.csv`
(+ copia fija `L5v3.csv`, que lee el notebook, y las lecturas sueltas).

**Costo estimado** (con los tokens por informe del piloto de 20 de D_strict en gpt-5.4-mini: 1316 in / 192 out; F con un
prompt más corto y 3 salidas ≈ 900 in / 80 out; G con citas ≈ 1400 in / 420 out). Precio de gpt-5.4 supuesto
(US$2.50 / 15 por M, no verificado), gpt-5.4-mini US$0.75 / 4.50:
F ≈ US$15.2 · G ≈ US$13.0 · **total ≈ US$28.2** (tope US$30; corte del script a US$29).

**Regla de reducción (fija):** el script corre primero un piloto de 20 informes con las 2 lecturas nuevas y proyecta el
costo con los tokens reales. Si F + G > US$29 y F sola cabe, corre **solo F_focus3**; si ni F cabe, para
(`L5V3_OVER_BUDGET`). Si la API responde sin cuota: `L5V3_NO_QUOTA` y sale (reanuda desde caché al relanzar).

**Validación (sin gold para seleccionar; mismos chequeos fijos que L5 y L5v2):** cobertura ≥ 99 %; macro AUC ≥ 0.90 contra
cada una de L2, L3, L4 y Raptor; ≥ 0.93 contra el consenso de las 4; ningún hallazgo < 0.80. Si no pasa:
`L5V3_UNHEALTHY` y no se entrena. Se reporta: acuerdo con L5v2, κ entre lecturas, gold (58).

### Fase 2 — sub056_3 (fold 0)

**Hipótesis:** con L5v3 en lugar de L5v2 como 5.ª fuente, la pseudo AUC L2 de fold 0 sube ≥ +0.003 contra sub056_2.

**Única variable:** `L5_CSV`: `L5v2_20261012.csv` → `L5v3.csv`. Todo lo demás idéntico a sub056_2 (= exp056).

**Criterios (fijos):** con los números de `results/sub056_2_L5v2/metrics.json` (fold 0):
- `worked`: pseudo L2 fold 0 (mejor época) ≥ sub056_2 + 0.003 **y** gold ≥ gold sub056_2 − 0.02.
- `killed`: pseudo L2 fold 0 ≤ sub056_2 − 0.005 **o** gold < gold sub056_2 − 0.02.
- `didnt`: cualquier otro caso.

**Presupuesto:** API ≤ US$30; ≈ 2.5 GPU-h.

## 2. PLAN

- Todo vive en `/workspace/users/rodrigo/l5v3/` (copia versionada en esta carpeta): `run_l5v3.sh` (orquesta),
  `generar_L5v3.py` (HTTP directo, sin paquete `openai`; clave en `/root/.openai_key` de rsna-cron), `validar_L5v3.py`,
  `train_sub056_3.sh` (en el pod de GPU: smoke → fold 0), `README.md`.
- Se lanza desde rsna-cron: `bash run_l5v3.sh --pod <PUERTO> <HOST>` solo si existe `l5v3/GO`.
- Notebook `sub056_3_L5v3.ipynb` = copia del de sub056_2; cambian la celda 0 (título) y la de parámetros (`EXP`,
  `PADRE`, `L5_CSV`, `NOTES`). Worktree en el volumen `wt_sub056_3_L5v3` con symlink `results/sub056_2_L5v2`.
- Dry-run hecho (2026-10-12 11:13 UTC, sin API ni pods): piloto/run en seco OK, estimación US$28.16, chequeo de pod
  OK (d2 ocupado → exit 7); build + validación con las 5 lecturas en caché reproducen L5v2 (AUC 1.0 contra L5v2).

## 3. VERDICT

No corrido: sub056_2 salió `didnt` (pseudo L2 0.8874 vs exp056 0.8857). El pipeline queda en `/workspace/users/rodrigo/l5v3/` por si se reactiva.
