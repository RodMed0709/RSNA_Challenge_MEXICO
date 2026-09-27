# Guía para integrantes nuevos

Esta guía es el recorrido completo para quien llega al equipo: qué buscamos, dónde está cada cosa y cómo se hace un
experimento de principio a fin. No repite los detalles de los demás documentos; enlaza a ellos.

## 1. El reto y nuestra meta

- **Competencia:** [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
  (Kaggle). Por cada resonancia de rodilla se predicen 12 hallazgos; la métrica es el promedio de las 12 AUC ROC.
  Detalle en [01_EL_CHALLENGE.md](01_EL_CHALLENGE.md).
- **Fechas:** unión final del equipo **15 de octubre de 2026**; entrega final **22 de octubre de 2026, 23:59 UTC**.
- **Nuestra meta no es el leaderboard de AUC.** Vamos por el **premio de eficiencia** (modelo barato, buen AUC y algo
  de novedad) y por un **paper de journal**. Por eso todo experimento se mide con dos números: **AUC** y **costo**
  (GMACs por estudio y tiempo). La fórmula de eficiencia y su "tipo de cambio" (0.001 de AUC ≈ 70 s de runtime) están
  en `docs/papers/RANKING.md` §1.
- La hoja de ruta por pasos está en `docs/Hoja_de_Ruta_Eficiencia.docx` (versión 2, 27/09/2026, ya con
  los resultados del paso 1). Cada **paso** es una pregunta que se responde con varios experimentos.

## 2. Glosario corto

| Término | Qué significa aquí |
|---|---|
| **pseudo** | Etiquetas sacadas de los informes radiológicos con un LLM (hoy Qwen2.5-7B): 1 presente, 0 ausente, 0.5 incierto. Cubren ~4,350 estudios. `pseudo AUC` = AUC contra ellas en el fold de validación. **Es la métrica que decide.** |
| **gold** | Los 58 estudios con etiqueta de experto (`fold = -1`). Nunca se entrena con ellos y **nunca eligen nada**: solo vetan. |
| **silver** | Set etiquetado a mano por el equipo (propuesto para el siguiente paso) para comparar extractores de etiquetas sin gastar gold. |
| **fold** | Una de las 5 particiones (0–4) de `split_v1_seed42`. Se entrena con 4 y se valida con la otra. |
| **OOF** | *Out-of-fold*: las predicciones de validación de los 5 folds juntas; su AUC es la medida confirmatoria. |
| **screen / confirm** | Screen = correr solo `FOLDS = [0]` y comparar contra el fold 0 del padre. Confirm = los 5 folds, solo si el screen promete. |
| **GMACs/estudio** | Miles de millones de multiplicaciones-suma para procesar un estudio. Nuestra medida de costo, junto con ms/estudio. |
| **SDD** | *Spec-driven development*: nada se programa sin SPEC escrita y confirmada. Ciclo en [06_SDD.md](06_SDD.md). |
| **SPEC** | `experiments/expNNN_tag/SPEC.md`: hipótesis, **una** variable contra el padre, criterios numéricos, presupuesto y plan. |
| **verdict** | Sección 3 de la SPEC + fila en [EXPERIMENTOS.md](EXPERIMENTOS.md): `worked`, `didnt` o `killed`, con números copiados de `metrics.json`. |
| **padre** | El experimento contra el que te comparas. Normalmente la **referencia actual** de [EXPERIMENTOS.md](EXPERIMENTOS.md). |

## 3. Mapa del repo (tu clon)

Repo: `https://github.com/RodMed0709/RSNA_Challenge_MEXICO.git`. Cada persona lo clona en `/workspace/users/<tu_nombre>/`.

| Ruta | Qué es | ¿La tocas? |
|---|---|---|
| `CLAUDE.md` / `AGENTS.md` | Reglas del equipo (léelas aunque no uses IA) | No |
| `docs/` | Proceso, reglas, registro [EXPERIMENTOS.md](EXPERIMENTOS.md), diseño por pasos en `docs/pasos/` | Solo tu fila de `EXPERIMENTOS.md` |
| `experiments/expNNN_tag/` | `SPEC.md` + notebook de cada experimento | Solo los tuyos |
| `experiments/_plantilla/SPEC.md` | Plantilla de SPEC | No |
| `results/expNNN_tag/` | Pesos, predicciones, logs y `metrics.json` | Los tuyos; solo `metrics.json` va a git |
| `src/knee/` | Código compartido (cache, dataset, modelo, métricas, costo, experimentos) | No; cambia por PR aparte |
| `SPLITS/` | Split canónico `split_v1_seed42` ([07_SPLIT.md](07_SPLIT.md)) | Nunca |
| `setup_pod.sh`, `run_notebook.sh`, `_env.sh` | Preparar el pod y correr notebooks en segundo plano | No |
| `inferencia_kaggle.ipynb` | Notebook que se sube a Kaggle | No |

## 4. Mapa del volumen compartido (`/workspace`)

Volumen de RunPod `RSNA-KNEE`, región `EU-RO-1`, montado en `/workspace`. Detalle en [04_DATOS.md](04_DATOS.md).

| Ruta | Contenido |
|---|---|
| `/workspace/data/raw/` | DICOM y CSV de Kaggle tal cual (~819 mil archivos). **No lo recorras**: entrena desde el cache. |
| `/workspace/data/cache/v1_256/` | Cache v1: una `.npy` uint8 `[cortes, 256, 256]` por serie + `index.csv` (~59 GB). Lo usa exp002. |
| `/workspace/data/cache/meta_v2/` | Metadatos DICOM v2: plano, ponderación, lateralidad, slots (exp009). |
| `/workspace/data/cache/v2_05mm/` | Cache v2 a 0.5 mm/px, ventana de 160 mm (~72 GB). Diseño en [pasos/paso3_entrada_v2.md](pasos/paso3_entrada_v2.md). |
| `/workspace/data/labels/report_labels_v1.csv` | Pseudo-etiquetas v1 (Qwen2.5-7B). |
| `/workspace/data/_logs/` | Logs de descarga, unzip, autoapagado y migración. |
| `/workspace/models/` | Pesos locales: `dinov2-small/`, `Qwen2.5-7B-Instruct/`, `hf_cache/` (caché de Hugging Face/timm). |
| `/workspace/venv/` · `/workspace/venv312/` | Ambientes de Python 3.11 y 3.12. `setup_pod.sh` elige el correcto. |
| `/workspace/users/<nombre>/` | Clon personal de cada integrante. **Solo escribes en el tuyo.** |
| `/workspace/users/.claims/` | Reservas de números de experimento que hace `nuevo_experimento()`. No se edita a mano. |

## 5. Flujo completo de un experimento

Los pasos 1–3 son de cada sesión; del 4 en adelante, de cada experimento.

**1. Prende el pod.** En RunPod: `Storage` → volumen `RSNA-KNEE` (`EU-RO-1`) → `Deploy`, template `RSNA-KNEE`,
GPU de 48 GB o más si hay (ver [03_RUNPOD.md](03_RUNPOD.md)). Abre Jupyter → `Terminal`.

**2. Prepara el ambiente** (cada vez que abras un pod; el kernel no sobrevive entre pods):

```bash
cd /workspace/users/<tu_nombre>
bash setup_pod.sh
```

En Jupyter usa siempre el kernel **Python (rsna)**.

**3. Actualiza `main`:**

```bash
git switch main && git pull
```

**4. Reserva número (CLAIM).** En una celda de cualquier notebook:

```python
import sys; sys.path.insert(0, "/workspace/users/<tu_nombre>")
from src.knee.experiments import nuevo_experimento, tabla_experimentos
nuevo_experimento("mi_tag", padre="exp002", idea="una línea con la idea")
```

o en terminal: `python -m src.knee.experiments nuevo mi_tag --padre exp002 --idea "..."`
(con el `python` del venv activo; si no, `/workspace/venv/bin/python -m ...`).
Crea `experiments/expNNN_mi_tag/` (notebook copiado del padre con `SMOKE = True`, `SPEC.md` en blanco),
`results/expNNN_mi_tag/` y tu fila `spec` en `docs/EXPERIMENTOS.md`. **Nunca elijas el número a mano.**

**5. SPEC y PLAN.** Llena las secciones 1 y 2 de `SPEC.md`: una sola variable, criterios, presupuesto en GPU-h y la
lista cerrada de celdas que cambian. Para pedir ayuda a Claude sin acceso al repo usa el prompt de [06_SDD.md](06_SDD.md).
Pide a Rodrigo que la confirme.

**6. Rama y commit de la SPEC, antes de tocar el notebook:**

```bash
git switch -c expNNN_mi_tag
git add experiments/expNNN_mi_tag docs/EXPERIMENTOS.md
git commit -m "expNNN: spec"
git push -u origin HEAD
```

**7. IMPLEMENT + SMOKE.** Cambia solo las celdas del PLAN. Corre el notebook completo con `SMOKE = True`
(1 época, pocos estudios). Si termina sin errores, pon `SMOKE = False` y `FOLDS = [0]`.

**8. RUN (screen, fold 0)** en segundo plano, para poder cerrar el navegador:

```bash
bash run_notebook.sh experiments/expNNN_mi_tag/expNNN_mi_tag.ipynb
tail -f results/expNNN_mi_tag/run_*.log      # Ctrl+C sale del tail, no detiene el entrenamiento
nvidia-smi
```

Referencia de tiempo: exp002 tarda ~1.5 h por fold en RTX PRO 6000.

**9. Lee `results/expNNN_mi_tag/metrics.json`.** La celda 5 ya calcula `decision_auto` contra el padre:

- Δ pseudo fold 0 ≥ +0.005 → `promising` → corre los 5 folds:
  `bash run_notebook.sh experiments/expNNN_mi_tag/expNNN_mi_tag.ipynb -y "FOLDS: [0, 1, 2, 3, 4]"`
- −0.005 < Δ < +0.005 → `neutral` → cuenta como `didnt`, salvo que ahorre ≥ 30 % de cómputo (GMACs/estudio).
- Δ ≤ −0.005 → `didnt`. Gold cae más de 0.02 contra el padre → `killed`, diga lo que diga pseudo.

**10. VERDICT.** Llena la sección 3 de la SPEC y completa tu fila en `docs/EXPERIMENTOS.md` copiando los números de
`metrics.json` (nunca de memoria). Funcione o no.

**11. Commit, push y PR:**

```bash
git add experiments/expNNN_mi_tag docs/EXPERIMENTOS.md results/expNNN_mi_tag/metrics.json
git commit -m "expNNN: verdict"
git push
```

Abre el PR a `main` en GitHub (el enlace sale al hacer push). Rodrigo lo revisa y lo junta.

**12. Apaga el pod** desde la consola de RunPod, después de confirmar que `metrics.json` existe. El volumen conserva todo.

Para ver lo de todo el equipo: `tabla_experimentos()` en un notebook o `python -m src.knee.experiments tabla`.
Para cargar pesos de otro sin copiarlos: `pesos("expNNN")`.

## 6. Reglas no negociables (resumen)

La versión completa está en `CLAUDE.md` y [02_REGLAS_DEL_EQUIPO.md](02_REGLAS_DEL_EQUIPO.md).

1. SPEC antes de código, confirmada.
2. Una variable por experimento contra el padre. Dos cambios = dos experimentos.
3. Número global único vía `nuevo_experimento()`.
4. Se entrena en el notebook del experimento, por celdas. Nada de `.py` para experimentos; `src/knee/` cambia solo por PR aparte.
5. El split `split_v1_seed42` no se toca.
6. Gold nunca selecciona época, checkpoint ni hiperparámetro; solo veta. `SELECT_ON = "pseudo"` siempre.
7. Decide la pseudo AUC: screen en fold 0, promete con Δ ≥ +0.005, se confirma con 5 folds.
8. Los criterios de la SPEC no cambian después de ver resultados.
9. Verdict obligatorio, aunque haya fallado.
10. Solo escribes en `/workspace/users/<tu_nombre>/`.
11. A git nunca van pesos, predicciones, caches ni datos. De `results/` solo `metrics.json`.
12. Una rama por experimento desde `main` actualizado, PR a `main`. Nada directo a `main`, nunca `push --force`.
13. Los datos de la competencia (imágenes, informes, CSV) no salen del equipo: no se pegan en ChatGPT, Claude web ni
    ninguna API externa. Para redactar SPEC o verdict con un chat, pega solo la SPEC y `metrics.json`.

## 7. Dónde vamos

- **Referencia actual: exp002_dinov2s_ref** (DINOv2-S): OOF pseudo 0.802, gold 0.788 (5 folds); fold 0: pseudo 0.827,
  gold 0.777; ~490 GMACs/estudio.
- **Paso 1 (barrido de backbones, exp003–exp008): cerrado.** Ninguno superó a DINOv2-S en pseudo fold 0.
  ConvNeXtV2-Nano (exp005) quedó en pseudo 0.802 con ~196 GMACs/estudio, **~60 % menos cómputo**. CoAtNet-2 (exp008)
  fue el más cercano en AUC (0.814) pero con 2.5× el cómputo. Números en `metrics.json` de cada experimento
  (campo `cost_a100`) y en [EXPERIMENTOS.md](EXPERIMENTOS.md).
- **exp009_preparacion_v2:** metadatos DICOM v2 y cache `v2_05mm` listos para el paso 4, entrada v2
  ([pasos/paso3_entrada_v2.md](pasos/paso3_entrada_v2.md); el archivo se escribió cuando era el paso 3).
- **Paso 2 propuesto: "¿Quién etiqueta mejor?"** Comparar LLMs abiertos locales (Qwen, Llama, MedGemma y un
  consenso) contra un set **silver** etiquetado a mano, para no gastar gold. Detalle en la hoja de ruta;
  las SPECs se definen con Rodrigo.

## 8. Errores comunes

- **Olvidar `bash setup_pod.sh` al abrir un pod nuevo:** el kernel `Python (rsna)` no aparece o faltan paquetes.
- **Trabajar sobre una rama vieja:** siempre `git switch main && git pull` antes de `nuevo_experimento()`.
- **Inventar el número o copiar una carpeta de experimento a mano:** rompe la numeración global. Usa `nuevo_experimento()`.
- **Tocar el notebook antes de commitear la SPEC**, o cambiar celdas fuera del PLAN.
- **Mirar gold para elegir la época "buena"** o ajustar un hiperparámetro: invalida el experimento.
- **Correr 5 folds sin screen:** 5 folds de exp002 son ~7.5 h de GPU. Primero `FOLDS = [0]`.
- **Recorrer `/workspace/data/raw`** (`ls -R`, `find`, abrir DICOM en masa): vuelve lento el disco para todos.
- **Guardar todos los checkpoints:** solo el mejor por fold. Revisa espacio con `du -sh /workspace/users/*`.
- **`git add .` o `git add results/`:** puede arrastrar pesos. Agrega rutas explícitas.
- **Dejar un `metrics.json` de smoke o de prueba** con el nombre de un experimento ajeno en tu `results/`:
  `tabla_experimentos()` y `pesos()` lo leen como si fuera real. Bórralo al terminar la prueba.
- **Conflicto en `git pull` que no sabes resolver:** `git merge --abort` y avisa.
- **Dejar el pod prendido:** la GPU cobra mientras esté encendido.
- **Tokens o contraseñas dentro del repo:** nunca.
