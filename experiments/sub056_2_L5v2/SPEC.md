# sub056_2_L5v2

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp056_raptor_combo (exp045 aug espacial + consenso de 5 fuentes con L5) |
| Fecha | 2026-10-12 |
| Rama | sub056_2_L5v2 (desde exp056_raptor_combo, 2083da0) |
| Estado | didnt |

Idea (autorizada por Rodrigo vía orquestador): mejorar la 5.ª fuente del consenso con **más lecturas por informe**.
L5 = 3 lecturas (gpt-4.1-mini). **L5v2 = las mismas 3 + 2 lecturas nuevas** con prompts distintos y un modelo más
fuerte, promediadas por celda (k = 5).

## 1. SPEC (antes de tocar el notebook)

### Fase 1 — generar L5v2

| Punto | Decisión |
|---|---|
| Lecturas A, B, C | las de L5 tal cual (caché de exp054, gpt-4.1-mini); no se pagan de nuevo. |
| **D_strict** (nueva) | estilo radiólogo con las definiciones oficiales de Kaggle **estrictas** escritas explícitamente (ACL/MCL solo rotura de alto grado o completa; derrame y Baker solo moderado o grande; OA solo con pérdida de cartílago > 50 % / grado 3-4; menisco solo rotura que llega a superficie). Por hallazgo: categoría de severidad + probabilidad 0–100. Modelo **gpt-5.4-mini**, razonamiento `none`. |
| **E_checklist** (nueva) | lista de verificación por hallazgo: `mentioned`, `negated`, `hedged`, `meets_definition` (yes/no/unclear) → probabilidad con mapeo fijo (no mencionado 0.05; negado 0.01; yes 0.95 / con duda 0.55; unclear 0.60 / con duda 0.40; no 0.15 / con duda 0.10). Modelo **gpt-5.4-nano**, razonamiento `low`. |
| L5v2 | media por celda de las 5 lecturas; 12 columnas, mismo orden y nombres. |
| Dónde vive | `/workspace/data/labels/ours/L5v2_20261012.csv` + las 5 lecturas sueltas `L5v2_20261012__<prompt>.csv`. No va a git. |
| Presupuesto | ≤ US$15 de API para D + E. |

**Elección de modelo (por costo, antes de ver la calidad completa).** Piloto de 20 informes (semilla 54), costo de D + E:
gpt-5-mini no disponible (organización no verificada); gpt-4.1 ≈ 5× gpt-4.1-mini (~US$40 proyectado, fuera);
gpt-5.4-mini `none` US$0.086 / 20 → ~US$18.9 los dos prompts (fuera); gpt-5.4-mini `low` ~US$24 (fuera);
gpt-5.4-nano `low` ~US$6.6 (precio de lista supuesto US$0.20 / 1.25 por M). Sin caché de prompt (prefijo < 1024
tokens). Decisión: D en gpt-5.4-mini `none` (~US$8), E en gpt-5.4-nano `low` (~US$3.3), dentro de US$15. Los prompts
no se tocaron después del piloto.

**Validación de L5v2, sin gold** (mismos chequeos que L5 en exp054, fijos): cobertura ≥ 99 %; macro AUC ≥ 0.90 contra
cada una de L2, L3, L4 y Raptor; macro AUC ≥ 0.93 contra el consenso de las 4; ningún hallazgo < 0.80 contra el
consenso de las 4; costo ≤ US$15. Se reporta, no decide: κ entre las 5 lecturas, macro AUC de cada lectura contra el
consenso de las 4, acuerdo L5v2–L5, gold (58). Si L5v2 no está sana, la fase 2 no corre.

### Fase 2 — sub056_2 (entrenamiento, solo fold 0)

**Hipótesis:** con L5v2 (5 lecturas) en lugar de L5 (3 lecturas) como 5.ª fuente del consenso, la pseudo AUC L2 de
fold 0 sube ≥ +0.003 contra exp056 sin que gold caiga más de 0.02.

**Única variable:** el CSV de la 5.ª fuente (`L5_CSV`: `L5_20261011.csv` → `L5v2_20261012.csv`). Todo lo demás
idéntico a exp056 (aug espacial de exp045, consenso de 5 fuentes, CoAtNet-2 384, 16 épocas, semilla 42, fold 0,
selección pseudo L2, SWA top-3).

**Criterios (fijos; los números del padre se copian de `results/exp056_raptor_combo/metrics.json` del fold 0 cuando
exista, la regla no cambia):**
- `worked`: pseudo L2 fold 0 (mejor época) ≥ exp056 + 0.003 **y** gold ≥ gold exp056 − 0.02.
- `killed`: pseudo L2 fold 0 ≤ exp056 − 0.005 **o** gold < gold exp056 − 0.02.
- `didnt`: cualquier otro caso. Ruido de semilla de referencia: 0.0013 (sub031_2).

**Sesgos declarados:** como en exp054, la vara pseudo L2 binariza L2; L5v2 no cambia el peso de L2 (1/5), solo la 5.ª
fuente. D es deliberadamente más estricta (puede alejar el consenso de L2 y bajar la pseudo sin ser peor). Gold no
tocó L5v2.

**Presupuesto:** fase 1 ≤ US$15; fase 2 ≈ 2.5 GPU-h en el pod que asigne el orquestador.

## 2. PLAN

- Fase 1: `generar_L5v2.py` (= `generar_L5.py` de exp054 + prompts D/E, modelo por prompt, `--only`); `validar_L5.py`
  con 5 lecturas; números en `L5v2_validacion.json` y en `BITACORA.md`.
- Fase 2: notebook `sub056_2_L5v2.ipynb` = copia del de exp056; cambios: celda 0 (título) y celda de parámetros
  (`EXP`, `PADRE`, `L5_CSV`, `NOTES`). Smoke `SMOKE=True` y luego `FOLDS=[0]`.

## 3. VERDICT

**Resultado:** `didnt` · **Fase alcanzada:** screen fold 0

### Fase 1 — L5v2 (números de `L5v2_validacion.json`)

L5v2 **sana** (pasa los 4 chequeos y el costo). Costo de las 2 lecturas nuevas: D_strict US$8.27 (gpt-5.4-mini, 5.95 M
tokens in / 0.85 M out) + E_checklist US$3.46 (gpt-5.4-nano, 6.38 M in / 1.75 M out) + pilotos de selección de modelo
≈ US$0.17 → **≈ US$11.9** (≤ US$15). El precio de gpt-5.4-nano es el de lista supuesto (US$0.20 / 1.25 por M).
0 errores; cobertura 4407/4407.

| Referencia (binarizada > 0.5) | macro AUC L5v2 | macro κ | (L5, exp054) |
|---|---|---|---|
| L2 | 0.9944 | 0.634 | 0.9925 |
| L3 | 0.9926 | 0.562 | 0.9895 |
| L4 | 0.9938 | 0.645 | 0.9917 |
| Raptor | 0.9606 | 0.703 | 0.9567 |
| consenso de las 4 | 0.9939 (mín. Contusion 0.984) | 0.638 | 0.9920 |
| L5 (3 lecturas) | 0.9984 | 0.889 | — |

- Por lectura, macro AUC vs consenso de las 4: A 0.969 · B 0.986 · C 0.983 · **D 0.985** · **E 0.971**. κ de las nuevas
  con las viejas 0.70–0.77; D–E 0.734.
- L5v2 ordena un poco mejor que L5 contra todas las referencias (+0.002–0.004) y es algo más estricta (media 0.176 vs
  0.182). Muy correlada con L5 (AUC 0.998): 3 de 5 lecturas son las mismas, así que el efecto en el entrenamiento puede
  ser pequeño.
- Gold (solo reporte): L5v2 0.896 · L5 0.889 · consenso de las 4 0.895 · L2 0.893.

### Fase 2 — sub056_2 (fold 0, rsna-d2, 11:43–14:15 UTC)

Padre: `results/exp056_raptor_combo/metrics.json` (fold 0, 2026-10-12 11:41 UTC).

| Métrica | Padre (exp056) | Este | Δ |
|---|---|---|---|
| pseudo L2 fold 0 (mejor época) | 0.8857 (ep 15) | 0.8874 (ep 15) | **+0.0017** |
| pseudo L2 fold 0 (SWA top-3) | 0.8854 | 0.8869 | +0.0015 |
| v1 fold 0 | 0.8407 | 0.8437 | +0.0030 |
| gold (veto) | 0.9198 (SWA 0.9212) | 0.9201 (SWA 0.9203) | +0.0002 |
| GPU-h | 2.51 | 2.51 | 0 |

Criterio: +0.0017 < +0.003 y > −0.005; gold no cae → **didnt**. Δ ≈ 1.3 × el ruido de semilla (0.0013): no se
distingue de ruido.

**Qué aprendimos:** pasar de 3 a 5 lecturas por informe (L5 → L5v2, +US$11.9) mejora un poco el acuerdo de la fuente con
las referencias (macro AUC +0.002–0.004) pero no mueve el modelo de forma medible: L5v2 y L5 se parecen demasiado
(AUC 0.998, κ 0.889). Por hallazgo (pseudo): Contusion +0.014, Synovitis +0.013, ACL +0.010; MCL −0.011, Baker's −0.008,
Medial Meniscus −0.006. La ganancia de exp054 venía de añadir una fuente distinta, no de afinarla.

**Siguiente paso:** L5 (3 lecturas) sigue en el combo. **L5v3 / sub056_3 no se corre** (la regla de ESTADO.md exige
sub056_2 = `worked`); queda preparado en `/workspace/users/rodrigo/l5v3/` por si se quiere probar otra fuente
realmente distinta. Sin all-data.
