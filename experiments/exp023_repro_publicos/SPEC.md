# exp023_repro_publicos

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | — (base externa: notebooks públicos) |
| Fecha | 2026-10-11 |
| Rama | exp010_dev_informes (registro); kernels solo en Kaggle |
| Estado | worked |

Idea: reproducir en Kaggle, con nuestra cuenta y sin cambiar nada, los notebooks públicos líderes, para tener una base
medida (LB público y runtime) sobre la que construir y contra la que medir el track de eficiencia.

Contexto en `docs/pasos/paso6_plan_lb.md`.

## 1. SPEC

**Hipótesis:** reproducir la base pública en nuestra cuenta da LB público ≥ 0.945 (los notebooks declaran 0.949–0.955).

**Qué se corre (copias exactas, kernels privados `medellinrodrigo/<corto>-exp023`, GPU T4 ×2, sin internet):**

| Corto | Original | LB declarado | Fuentes |
|---|---|---|---|
| velci | `bliverrigpdoge/velci-kcd25-20` | 0.954 (citado como 0.955) | heliosli/rsna-knee-compact-v1, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-orthofoundation-fold0-epoch7-20261004 |
| mirror | `nartaa/rsna-knee-0949-anatomical-mirror` | 0.949 | nartaa/rsna-knee-publication-swa-weights-20261007 |
| blendmine | `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters` | — (no declara) | heliosli/rsna-knee-compact-r6cnxfsss, nartaa/rsna-knee-publication-swa-weights-20261007, heliosli/rsna-orthofoundation-fold0-epoch7-20261004 |

Única modificación: el metadata (id, título, privado). El código del notebook no se toca. Una fuente de velci
(`bliverrigpdoge/rsna-knee-kcd25leg`, aparece vacía en el metadata público) es privada (403): el notebook está escrito para
seguir sin ella (deja la salida de la base 0.954 sin el blend KCD25), así que nuestra copia mide la base sin KCD25.

Blend-mine: el envío de la copia v1 fue rechazado ("non-versioned dataset" `heliosli/rsna-knee-compact-r6cnxfsss`). Se
re-alojó la versión 2 (mismos archivos, `bundle_id 209a32c8b50a6b1c`, que el notebook verifica) como dataset privado
`medellinrodrigo/rsna-knee-compact-r6cnxfsss` y se corrió la v2 del kernel; el código sigue sin cambios (busca la carpeta
por slug con cualquier dueño).

No se buscó ningún notebook con LB declarado mayor que 0.955: los títulos con "0.965" son objetivos, no resultados.

**Criterios (no cambian después de correr):**
- `worked`: al menos un kernel da LB público ≥ 0.945.
- `didnt`: todos los envíos < 0.945 (la base no se reproduce en nuestra cuenta: investigar fuentes o versión de datos).
- Kernel que falla o pasa de 9 h: se reporta, no cuenta.
- Gold y pseudo no aplican: no hay entrenamiento ni etiquetas nuestras. Decide el LB público (es una reproducción, no una
  selección de modelo).

**Presupuesto:** 0 GPU-h de pods · ≤ 3 envíos (límite 5/día; quedan ≥ 2 para el orquestador) · cuota GPU Kaggle ≤ ~4 h.

**Riesgos:** fuentes públicas que cambian de versión entre su run y el nuestro; runtime en test oculto distinto del visible.

## 2. PLAN

1. `kernels pull -m` de cada original a `scratchpad/kaggle_exp023/<corto>/kernel/`; cambiar solo el metadata.
2. `kernels push`; Kaggle admite 2 sesiones GPU simultáneas, así que blendmine entra cuando se libera una.
3. Al terminar: verificar `submission.csv` (filas = test, 13 columnas, sin nulos) y enviar con
   `competitions submit -k medellinrodrigo/<corto>-exp023 -v 1`, en orden de LB declarado.
4. Registrar LB, runtime del run (log del kernel) y puesto aproximado.

## 3. VERDICT

Números del LB público (`competitions submissions`), runtime del run sobre el test visible (log del kernel) y tiempo
desde el envío hasta tener score (cota del runtime en el test oculto; incluye la cola de Kaggle).

| Corto | Kernel | ID envío | LB público | Puesto aprox. (de 5760) | Run test visible (3 estudios) | Envío → score |
|---|---|---|---|---|---|---|
| velci | medellinrodrigo/velci-exp023 v1 | 57081259 | **0.954** | 135–535 (empate en 0.954) | 194.1 s | ~3 h 25 min |
| blendmine | medellinrodrigo/blendmine-exp023 v2 | 57082625 | **0.954** | 135–535 (empate en 0.954) | 184.3 s | ~4 h 10 min |
| mirror | medellinrodrigo/mirror-exp023 v1 | 57081163 | 0.949 | 1175–1305 (empate en 0.949) | 66.6 s | ~25–29 min |

**Verdict: `worked`.** La hipótesis (LB ≥ 0.945) se cumple en los tres. Los LB declarados se reproducen exactos (0.954 y 0.949).
Velci sin KCD25 da lo mismo que su base pública 0.954; blend-mine (pesos 0.5/0.5) empata con velci (0.7/0.3) en LB público.

**Lectura:**
- Base para construir encima: el blend Raptor CoAtNet + compact ConvNeXt-small S6 (velci o blend-mine), 0.954.
- Base para eficiencia: el CoAtNet solo de nartaa. Da −0.005 AUC pero tarda ~0.5 h hasta score contra ~3.5–4 h de los blends.
  El compact S6 cuesta ~3 h de runtime por +0.005.
- El puesto con 0.954 está en un empate de ~400 equipos (forks públicos): subir requiere algo propio por encima de la base.
- Gold/pseudo no aplican (no hay entrenamiento).

