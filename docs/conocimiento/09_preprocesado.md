# 09 — Preprocesado de RM: qué probar además de CLAHE

Fecha: 2026-10-12. Fuentes: `docs/papers/08_preprocesamiento_mri/` (README + PDFs, extraídos con `pdftotext`),
`local/RSNA_papers/01_knee_mri_sota` (MRNet, CoPAS, Vuskov), `local/RSNA_papers/03_mil_multiseq_arch/kaggle_*.md`,
`local/RSNA_papers/analisis/A2_implementacion_vs_papers.md` §2.6–2.8, el script de corpus
`experiments/exp029_raptor_entreno/build_corpus96.py` (rama `exp056_raptor_combo`), y búsqueda web con cada cita
verificada por DOI (Crossref / Semantic Scholar) o arXiv. Tiempos de CPU: **medidos por nosotros** en un PC local
(SimpleITK 2.5.6, OpenCV, 1 hilo salvo que se diga), volumen sintético; en Kaggle hay que volver a medir.
Convenciones de `README.md`: **[paper]**, **[Kaggle]**, **[inferencia]**, **[medido]**.

---

## 0. Resumen en 10 líneas

1. Ningún paper muestra que una normalización de intensidad mejore **clasificación DL** de RM de forma grande. Lo más
   cercano a nosotros (rodilla, Mills 2026) da diferencias de **décimas de punto de Dice** y explica el 0.5 % de la varianza.
2. El único paso que la literatura sí da por esencial es **unificar el espaciado (mm/px)** (Kondrateva 2024). Lo tenemos
   en el plano (recorte de 140 mm) **con una fuga**: si el FOV mide menos de 140 mm, el recorte se queda en el FOV y se
   **amplía** → escala física distinta según la serie (ver P-2).
3. El recorte de 140 mm es **centrado en la imagen, no en la rodilla**. Todos los sistemas multi-hallazgo de rodilla
   publicados (CoPAS, Xie 2025) y los ganadores Kaggle de 2 etapas recortan **sobre la anatomía**. Es el candidato con más
   techo y además es palanca de **eficiencia** (mismos mm de rodilla en menos píxeles).
4. **N4 está mal justificado para nosotros**: evidencia nula o negativa en DL (Kondrateva 2024) y coste prohibitivo:
   **4–22 s por serie en 1 hilo [medido]** → del orden de 1–2 h extra sobre el test oculto ≈ −0.05 a −0.10 de AUC
   equivalente en la fórmula de eficiencia. Si interesa el bias field, va como **augmentación** (AUG-BIAS, `02` §3.5).
5. **Nyúl está justificado pero con prior baja**: lo usan MRNet y Vuskov; en rodilla es el mejor en externo (+0.14 DSC
   vs z-score, Mills 2026) y nuestro método actual (min-max robusto por percentiles) quedó en el grupo malo externo
   (−0.48 DSC). In-distribution: ≈0. Coste en inferencia ≈ gratis si se hace con LUT.
6. **CLAHE: prior baja**. Mills 2026: peor que z-score/min-max en interno; en nuestros datos +0.0028 sobre padre viejo y
   −0.0018 sobre combo. Sirve como miembro de ensamble, no como cambio del pipeline.
7. Denoising (NLM/BM3D/DnCNN), super-resolución, remuestreo isotrópico, ComBat, WhiteStripe: **no** (evidencia de calidad
   de imagen o de segmentación, coste alto, o requieren tejido de referencia / sitio conocido).
8. Antes de entrenar nada: **3 diagnósticos de CPU** sobre el corpus/DICOM de train (minutos): % de series con FOV < 140
   mm, desplazamiento del centroide de la rodilla respecto del centro, y orientaciones no estándar. Deciden si P-1/P-2/P-5
   valen una corrida.
9. Todo cambio de preprocesado = **reconstruir corpus** (~20 min, 24 procesos) + screen f0 (~2.5 GPU-h) + aplicarlo idéntico
   en `infer.py`. Una variable por experimento (regla 2).
10. Tipo de cambio (RANKING §1): **0.001 AUC ≈ 70 s de runtime total**. Con ~1300 estudios y ~4 vCPU, 200 ms/estudio de CPU
    ≈ 65 s ≈ 0.001 AUC. Ese es el umbral para cualquier paso nuevo.

---

## 1. Pipeline actual (verificado en `build_corpus96.py`)

| Paso | Qué hace hoy | Comentario |
|---|---|---|
| Lectura | `apply_modality_lut` → float32; MONOCHROME1 invertido | correcto |
| Orden de cortes | `ImagePositionPatient · (row × col)`; fallback `InstanceNumber` | correcto |
| Selección | 5 slots (Sag FS 26, Sag 22, Cor FS 18, Cor 12, Ax 18) por `Anatomical_Plane` + `Fluid_Sensitive`; cortes **equiespaciados en índice** entre 2 % y 98 % de la serie | el espaciado físico entre cortes elegidos varía con la cobertura y el grosor de la serie (P-4) |
| Intensidad | percentiles 2–98 sobre **todos** los píxeles de los cortes elegidos (incluido el aire) → clip lineal [0,1] → uint8 | = "min-max robusto"; el ancla baja la fija el fondo |
| Geometría | `mm_crop_resize`: recorte **central de la matriz** de 140 mm (`PixelSpacing[0]`), `cv2.INTER_AREA` → 384; train recorta 320 | si FOV < 140 mm: `cpx = min(cpx, min(h, w))` → se amplía; `PixelSpacing[1]` ignorado |
| Entrada | 3 cortes vecinos = RGB (2.5D), normalización ImageNet | — |
| Lateralidad | espejo anatómico en train (exp045) | ya cubierto |

Cosas ya resueltas que **no** hace falta probar: unificación de mm/px en el plano (salvo la fuga), orden por posición,
selección de series por metadatos, lateralidad, ventana por serie.

---

## 2. Tabla de candidatos (rankeada)

Coste de inferencia = por estudio, CPU de Kaggle salvo que se diga; "total" = ~1300 estudios / 4 vCPU.
Ganancia esperada = **hipótesis** sobre pseudo f0 (ruido entre semillas ≈ 0.0015).

| Rank | id | Candidato | Evidencia (fuerza) | ¿Ya lo cubrimos? | Coste inferencia | Esfuerzo | Riesgo | Ganancia esperada |
|---|---|---|---|---|---|---|---|---|
| 1 | P-1 | **Recorte centrado en la rodilla** (centroide de foreground por serie, sin ML) y, si se confirma, recorte más ajustado (120 mm) | media-baja para clasificación (nadie publica ablación con/sin recorte en rodilla); fuerte como práctica (CoPAS, Xie, Kaggle 2 etapas) | **no**: recorte en el centro de la matriz | ~5–10 ms/serie (umbral + momentos en baja resolución) [inferencia] | bajo (corpus + infer) | medio: centroide sesgado por muslo/pantorrilla en sagital/coronal | +0.000 a +0.004 si hay rodillas descentradas; además habilita 224 px con menos pérdida (EF) |
| 2 | P-2 | **Escala física consistente**: *pad* en lugar de ampliar cuando FOV < 140 mm; usar `PixelSpacing` [fila, col] | indirecta pero sólida: el espaciado es el único paso esencial (Kondrateva 2024); nnU-Net remuestrea siempre | **parcial** (fuga) | 0 | muy bajo | bajo | 0 a +0.002; depende del % de series afectadas (diagnóstico D-1) |
| 3 | P-3 | **Nyúl por slot** (landmarks p1…p99 en foreground, estándar aprendido en train por slot) | baja-media: MRNet y Vuskov lo usan; Mills 2026 mejor en externo (+0.14 DSC vs z-score; min-max robusto −0.48) | no (tenemos min-max robusto de 2 landmarks) | ≤20 ms/serie con LUT; 80 ms/serie con `np.interp` a 384 [medido] | bajo | bajo | ≈0 in-dist; algo de robustez a sitio |
| 4 | P-4 | **z-score / percentiles sobre foreground** (máscara de aire) en vez de sobre todos los píxeles | baja: Mills (z-score ≈ Nyúl, ambos > min-max robusto en externo); nnU-Net usa z-score en RM | no | ~20 ms/serie [medido] | muy bajo | bajo (uint8: hay que fijar una ventana tras el z-score, p.ej. [−2, 4]σ) | ≈0 in-dist; es el **control natural** de P-3 |
| 5 | P-5 | **Muestreo through-plane en mm** (cortes a paso físico fijo centrado en la rodilla, no equiespaciados en índice) | baja (Kondrateva para segmentación; [inferencia]) | no | 0 (solo cambia qué cortes se leen) | medio | medio: cambia la "vista" de las ventanas 2.5D | 0 a +0.002 |
| 6 | P-6 | **Orientación canónica en el plano** desde `ImageOrientationPatient` (filas/columnas giradas o invertidas) | sin literatura [inferencia]; práctica estándar de reorientación (nnU-Net/TorchIO trabajan en espacio físico) | no comprobado | ~0 (flip/transpose) | bajo | bajo | 0 salvo que D-3 encuentre series raras |
| 7 | P-7 | **CLAHE** (re-test en el mejor padre; si se hace, clip 1.0) | baja: Mills interno peor que z-score; nuestras 2 corridas: +0.0028 / −0.0018 | probado 2× | 16 ms/serie a 384 [medido]; A2: 66 ms/estudio | ya hecho | bajo | ≈0; útil como diversidad de ensamble |
| 8 | P-8 | Recorte guiado por atención/CAM en 2 pasadas (estilo AOLM de Xie 2025) | media en su sistema (quitar AOLM baja AUC) pero mezclado con arquitectura | no | **×1.3–2 de GPU** (segunda pasada) | alto | alto | incierta; mala para EF |
| 9 | P-9 | Multi-ventana como canales (p.ej. 2 ventanas de intensidad) | CT (Lee 2018, WSO); nada en RM | no; choca con 2.5D (los 3 canales ya son cortes) | ~0 | medio | medio | ≈0 |
| 10 | P-10 | Cuantizar a uint8 **después** del recorte/resize o guardar uint16 | anecdótica (C-spine 3.º: "not sure if this helped") | no | 0 / ×2 de RAM | bajo | bajo | ≈0 |
| — | N4 | N4ITK bias correction | **negativa** en DL (Kondrateva 2024); de Raad 2021: depende de la tarea | no | **4.4–22 s/serie** en 1 hilo [medido] → ~2 h total en el ajuste más rápido | medio | **alto (EF)** | ≈0; coste ≈ −0.05 a −0.10 AUC equivalente |
| — | DEN | Denoising NLM / BM3D / DnCNN | solo calidad de imagen; ninguna mejora de clasificación sólida | no | NLM: 174 ms/**corte** 384 [medido] → ~17 s/estudio | medio | alto (borra textura sutil: contusión, menisco) | ≈0 o negativo |
| — | SR | Super-resolución / remuestreo isotrópico | calidad de imagen (Chaudhari 2018); CoPAS: interpolar "no aporta información" | no | GPU extra | alto | medio | ≈0 |
| — | CB | ComBat / harmonización por sitio | negativa para DL (Kushol 2023) | no | requiere sitio en test | — | — | — |
| — | WS | WhiteStripe / GMM | cerebro; en rodilla (GMM) peor que z-score (Mills) | no | — | — | — | — |

Alternativa a "estandarizar" que no cuesta nada en inferencia: **aleatorizar** en train (bias field por slot, gamma por
slot, GIN, BigAug) → fichas en `02_augmentations.md` §3.

---

## 3. Fichas (origen y detalle)

### P-1 · Recorte centrado en la rodilla — prioridad 1

- **Qué es.** Calcular por serie un centro anatómico y recortar los 140 mm (o menos) alrededor de él, con el **mismo centro
  para todos los cortes** de la serie (las ventanas 2.5D siguen alineadas).
- **Origen.**
  - CoPAS (Qiu 2024, Nat Commun, doi:10.1038/s41467-024-51888-4) [paper]: *"we segment the meniscus by U-Net as an upstream
    task. Note that a coarse segmentation is adequate because we only need a rough region of interest (ROI) to identify the
    center of the meniscus. We crop the original images according to the segmentation mask"*. Sin ablación del recorte.
  - Xie 2025 (eClinicalMedicine, doi:10.1016/j.eclinm.2025.103534) [paper, PMC12508577]: recorte 256×256 centrado en la
    máscara de menisco sagital, propagado a todos los cortes. Su ablación del módulo de localización por atención (AOLM)
    baja el AUC por hallazgo (de 0.730–0.966 a 0.687–0.953), pero es un módulo de red, no el recorte de preprocesado.
  - Astuto 2021 (Radiol AI, doi:10.1148/ryai.2021200165): pipeline localización → clasificación por tejido.
  - Kaggle [Kaggle]: Abdominal 2023 2.º (Theo Viel): *"key is to feed to the model crops where the information is located
    … It improves performances on kidney, liver and spleen by a good margin"*; C-spine 2022 1.º/3.º y Lumbar 2024 1.º
    recortan sobre segmentación/coordenadas (allí había etiquetas de localización; **aquí no**).
  - de Raad 2021 (ISBI, doi:10.1109/ISBI48211.2021.9433952): 24 configuraciones (normalización, **selección de ROI**, bias
    field, remuestreo) en cerebro, hígado y **rodilla**: hasta 64 puntos de diferencia entre configuraciones; la mejor
    cambia por dataset. Segmentación.
- **Por qué nos importa.** exp050 mostró que el borde que quita el recorte 320 no aporta (A2 §2.7) y exp027/exp057 que bajar
  a 224 px cuesta −0.007 gold. Si la rodilla está centrada, se puede recortar más (120 mm) y ganar mm/px al mismo coste, o
  bajar px sin perder mm de rodilla → **directamente útil para el premio de eficiencia**.
- **Sin etiquetas de localización.** Opción barata (recomendada primero): centroide de la máscara de foreground (Otsu en una
  versión 96 px de 3 cortes centrales), restringido a una ventana central (para que muslo/pantorrilla no lo arrastren) y con
  límite de desplazamiento (±25 mm). Opción cara: CAM del modelo actual como pseudo-etiqueta para un localizador mínimo
  (P-8) — no antes del 22-oct.
- **Diagnóstico previo D-2 (CPU, sin GPU).** Sobre 300 estudios de train: distribución del desplazamiento centroide–centro
  por slot, en mm. Si la mediana es < 5 mm y el p90 < 12 mm, P-1 no merece corrida; pasar directamente a "recorte 120 mm"
  como experimento de resolución.
- **SDD.** Una variable: "centro del recorte = centroide de foreground" (140 mm igual). Recorte más ajustado = otro
  experimento. Padre: el vigente (combo). Criterio: regla 7 del repo; secundaria: harness exp040.

```python
# Reemplaza mm_crop_resize en build_corpus96.py y en infer.py. Un centro por SERIE, no por corte.
import cv2, numpy as np

def knee_center(slices_f32, ps, max_shift_mm=25.0, win_frac=0.7):
    """slices_f32: lista de cortes ya en [0,1] (tras la ventana de intensidad). Devuelve (cy, cx) en px nativos."""
    mid = slices_f32[len(slices_f32) // 2 - 1: len(slices_f32) // 2 + 2]
    a = np.mean(mid, 0); h, w = a.shape; s = 96.0 / max(h, w)
    small = cv2.resize(a, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    u8 = (small * 255).astype(np.uint8)
    _, m = cv2.threshold(u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sh, sw = m.shape; wy, wx = int(sh * win_frac / 2), int(sw * win_frac / 2)
    sub = np.zeros_like(m); sub[sh//2-wy:sh//2+wy, sw//2-wx:sw//2+wx] = m[sh//2-wy:sh//2+wy, sw//2-wx:sw//2+wx]
    M = cv2.moments(sub, binaryImage=True)
    if M["m00"] < 50: return h / 2, w / 2
    cy, cx = M["m01"] / M["m00"] / s, M["m10"] / M["m00"] / s
    lim = max_shift_mm / max(ps, 1e-3)
    return float(np.clip(cy, h/2 - lim, h/2 + lim)), float(np.clip(cx, w/2 - lim, w/2 + lim))
```

### P-2 · Escala física consistente (pad en vez de ampliar) — prioridad 2

- **Qué es.** Hoy `cpx = min(cpx, min(h, w))`: con FOV de 120 mm la serie entra a 120/384 mm/px y con 160 mm a 140/384.
  La misma estructura sale de distinto tamaño según el protocolo. Arreglo: rellenar con ceros hasta 140 mm (como ya pasa con
  cortes vacíos) y respetar `PixelSpacing` por eje.
- **Origen.** Kondrateva 2024 (BSPC 96:106599, doi:10.1016/j.bspc.2024.106599; arXiv 2204.05278) [paper, abstract]:
  *"the only essential transformation for accurate deep learning analysis is the unification of voxel spacing across the
  dataset"* (segmentación de tumor cerebral). nnU-Net (Isensee 2021, doi:10.1038/s41592-020-01008-z) remuestrea siempre
  al espaciado mediano. Notebooks públicos: recorte físico 130 mm (`08_LO_QUE_HACEN_OTROS.md` #1).
- **Diagnóstico D-1.** % de series por slot con `min(rows, cols) · PixelSpacing < 140 mm` y % con píxel no cuadrado. Si < 3 %,
  meterlo como corrección silenciosa junto a otro cambio de corpus **no** vale (regla 2): registrarlo como experimento
  propio igualmente, o aceptar que el efecto es inmedible y aplicarlo solo si D-1 > 10 %.

```python
def mm_crop_resize(a, ps_rc, center=None, crop_mm=140.0, img=384):
    """ps_rc = (PixelSpacing[0] fila, PixelSpacing[1] columna). Pad con 0 si el FOV no llega a crop_mm."""
    h, w = a.shape; ph, pw = int(round(crop_mm / ps_rc[0])), int(round(crop_mm / ps_rc[1]))
    cy, cx = center if center is not None else (h / 2, w / 2)
    y0, x0 = int(round(cy - ph / 2)), int(round(cx - pw / 2))
    out = np.zeros((ph, pw), a.dtype)
    ys, xs = max(0, y0), max(0, x0); ye, xe = min(h, y0 + ph), min(w, x0 + pw)
    out[ys - y0:ye - y0, xs - x0:xe - x0] = a[ys:ye, xs:xe]
    return cv2.resize(out, (img, img), interpolation=cv2.INTER_AREA)
```

### P-3 · Nyúl por slot (con P-4 como control) — prioridad 3

- **Qué es.** Estandarización de histograma por tramos (Nyúl & Udupa 1999, MRM, doi:10.1002/(SICI)1522-2594(199912)42:6<1072::AID-MRM11>3.0.CO;2-M;
  variantes en Nyúl, Udupa & Zhang 2000, IEEE TMI, doi:10.1109/42.836373): por serie, percentiles p1, p10…p90, p99 del
  foreground se mapean linealmente a tramos sobre landmarks estándar aprendidos en train. Nuestro p2–98 actual es el caso de
  2 landmarks con estándar identidad: Nyúl añade la **forma** del histograma.
- **Origen.**
  - MRNet (Bien 2018) [paper]: *"For each series, a representative intensity distribution was learned from the training set
    exams. Then, the parameters of this distribution were used to adjust the pixel intensities of exams in all datasets"*.
  - Vuskov 2026 [paper]: *"Histogram normalization was applied to each image sequence separately. The normalization
    parameters were calculated using the internal dataset … clipped at their respective 99.9th percentiles and linearly
    rescaled to [0,1]. Then, normalization using the ImageNet mean and variance was applied"* → es casi nuestro pipeline
    con Nyúl en lugar de p2–98.
  - Mills 2026 (MIUA, arXiv 2607.20028) [paper], segmentación de menisco nnU-Net, IWOAI → SKM-TEA. Interno DSC: z-score
    89.04, min-max robusto (p1–p99) 88.90, CLAHE 88.83, Nyúl 88.93 (sin diferencias relevantes). Externo (vs z-score):
    **Nyúl +0.14** [0.05, 0.24], CLAHE −0.09, **min-max robusto −0.48**, HE −0.65. El método explica el **0.5 %** de la
    varianza (R² marginal 0.005). Una sola secuencia (DESS) y segmentación: transferencia a clasificación multi-secuencia = [inferencia].
  - Carré 2020 (Sci Rep, doi:10.1038/s41598-020-69298-z): radiómica + ML, no DL: balanced accuracy 0.67 sin normalizar →
    0.82 / 0.79 / 0.82 (Nyúl / WhiteStripe / z-score). Recomiendan z-score.
  - En contra: Kondrateva 2024 — histogram matching no mejora la segmentación DL.
- **Lectura honesta.** In-distribution esperamos ≈0 (exp051 ya mostró que mover el clip no mueve nada). El argumento es
  robustez multi-sitio (test oculto) y que el método actual es de la familia que peor transfiere en Mills. Exigir la
  métrica secundaria del harness exp040 (sitio croata).
- **Detalles de implementación.** Landmarks por **slot** (5 estándares), aprendidos con los estudios de train del fold
  (no usan etiquetas, pero mejor no tocar gold/val); foreground = píxeles > media de la serie (umbral clásico de Nyúl);
  los valores > p99 mapeados con la pendiente del último tramo y luego clip [0, 1.0]; aplicar **antes** de cuantizar.
  En inferencia, LUT uint16 para que cueste ~ms.
- **Control P-4** (otro experimento): z-score sobre foreground y ventana fija [−2, 4]σ → [0,1]. Si P-3 ≈ P-4 (como en Mills),
  quedarse con el más simple.

```python
NYUL_Q = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99], float)

def nyul_landmarks(vol_f32):
    fg = vol_f32[vol_f32 > vol_f32.mean()]
    p = np.percentile(fg[:: max(1, fg.size // 200_000)], NYUL_Q)        # submuestreo: ~ms
    return (p - p[0]) / (p[-1] - p[0] + 1e-6)                           # escala [0,1] entre p1 y p99

def fit_standard(series_iter):            # train del fold, un estándar por slot
    return np.mean([nyul_landmarks(v) for v in series_iter], 0)        # (11,), monótono

def nyul_apply(vol_f32, std):
    fg = vol_f32[vol_f32 > vol_f32.mean()]
    p = np.percentile(fg[:: max(1, fg.size // 200_000)], NYUL_Q)
    xs = np.concatenate([[p[0] - (p[1]-p[0]) * 10], p, [p[-1] + (p[-1]-p[-2]) * 10]])
    ys = np.concatenate([[std[0] - (std[1]-std[0]) * 10], std, [std[-1] + (std[-1]-std[-2]) * 10]])
    return np.clip(np.interp(vol_f32, xs, ys), 0.0, 1.0)              # sustituye al clip p2–98
```

### P-5 · Muestreo through-plane en mm

- Hoy: `np.linspace(lo, hi, k)` sobre índices entre 2 % y 98 % de la serie. Con series de 20 a 40 cortes y 3–4 mm de grosor,
  la separación entre cortes elegidos y la cobertura física cambian por estudio; con series cortas (< k) se duplican cortes
  y las ventanas 2.5D tienen canales idénticos.
- Propuesta: posiciones físicas (IPP·normal) a paso fijo (p.ej. 3.5 mm sagital) centradas en la mediana de la cobertura
  (o en el centro de P-1 proyectado), corte más cercano a cada posición, ceros fuera de cobertura.
- Evidencia: solo indirecta (Kondrateva: espaciado; nnU-Net). [inferencia]. Hacerlo **después** de P-1/P-2.

### P-6 · Orientación canónica (diagnóstico D-3 primero)

- Comprobar con `ImageOrientationPatient` qué fracción de series de cada plano tiene las filas/columnas en dirección no
  estándar (p.ej. coronal con la dirección de fila invertida, sagital con anterior a la derecha, axial girado 90°). Si hay
  > 2 %, normalizar con flip/transpose (coste ~0). El espejo L/R de exp045 no corrige esto. [inferencia, sin literatura].

### P-7 · CLAHE (brazo planeado)

- Origen: Zuiderveld 1994 (Graphics Gems IV; `zuiderveld1994clahe`), Pizer 1987. Ningún paper de clasificación de RM de
  rodilla ni ganador Kaggle lo usa (A2 §2.6). Mills 2026: en interno, junto con HE, peor que los tres mejores; en externo
  empata con z-score (−0.09).
- Nuestros datos: +0.0028 sobre exp031 (≈ +0.0020 contra media de 2 semillas; gold −0.006) y −0.0018 sobre el combo.
- Si se re-prueba en el mejor padre: es una réplica, no una idea nueva; el resultado más probable es ±0.002 (ruido).
  Valor real: diversidad de ensamble (correlación 0.949 ≈ una semilla, A2).

### N4 (brazo planeado) — no recomendado

- Origen: Tustison 2010 (IEEE TMI, doi:10.1109/TMI.2010.2046908).
- Evidencia en DL: Kondrateva 2024: *"intensity equalization steps (denoising, bias-field correction and histogram matching)
  do not improve models' performance"*; de Raad 2021: el efecto de bias field depende del dataset. Las CNN con normalización
  local y aug de intensidad absorben campos de baja frecuencia; Madaan 2023 (MIDL, arXiv 2306.13276) muestra que en rodilla
  (fastMRI+) la robustez al bias field depende sobre todo de la capa de normalización (GN/LN +8 % AUROC vs BN), no del preprocesado.
- **Coste [medido]**, SimpleITK, volumen 26×384×384, 1 hilo: shrink 4 / 4 niveles × 50 it **22 s**; 4 hilos 7.4 s;
  shrink 8 / 3 niveles × 10 it **4.4 s** (fit 3.7 s + campo 0.6 s). A 30×512×512: 19–51 s. Con 5 series/estudio y el ajuste
  más rápido: ~22 s/estudio → 1300 × 22 / 4 ≈ **2 h** ≈ −0.10 AUC equivalente. Ni con N4 por corte 2D se acerca al umbral
  de 200 ms/estudio.
- Si se quiere atacar el bias field: AUG-BIAS por slot (`02` §3.5, coste de inferencia 0).

### DEN · Denoising — no

- NLM adaptativo (Manjón 2010, JMRI, doi:10.1002/jmri.22003), BM3D (Dabov 2007, IEEE TIP, doi:10.1109/TIP.2007.901238),
  DnCNN (Zhang 2017, IEEE TIP, doi:10.1109/TIP.2017.2662206). Evidencia de mejora de **clasificación** DL en RM: solo
  trabajos pequeños de tumores cerebrales con pipelines de varias etapas, sin control; nada en rodilla. Kondrateva: no ayuda.
- Coste: `cv2.fastNlMeansDenoising` 174 ms por corte 384 [medido] → ~17 s/estudio. Riesgo de borrar señal (edema óseo,
  rotura meniscal fina). Descartado.

### SR · Super-resolución / isotrópico — no

- DeepResolve (Chaudhari 2018, MRM, doi:10.1002/mrm.27178) mejora calidad en grosor de corte de rodilla; no hay evidencia de
  mejora de clasificación. CoPAS [paper]: *"using interpolation will reduce this imbalance, but the interpolation with one
  image itself does not provide any information gain and may introduce additional errors"*. Nuestra respuesta a la
  anisotropía es 2.5D + planos ortogonales + AUG-ANI.

### Otros descartados

- **ComBat**: Kushol 2023 (Sci Rep, doi:10.1038/s41598-023-43715-5): cross-vendor hunde el DL y ComBat no mejora; además en
  test no conocemos el sitio.
- **WhiteStripe** (Shinohara 2014, NeuroImage Clin, doi:10.1016/j.nicl.2014.08.008): necesita un tejido de referencia
  (sustancia blanca); su análogo GMM en rodilla queda en el grupo peor de Mills.
- **Multi-ventana**: Lee 2018 (ML4H, arXiv 1812.00572) — ventana aprendible en **CT** (UH). En RM no hay escala absoluta y
  nuestros 3 canales ya son cortes vecinos.
- **Resolución**: Sabottke 2020 (Radiol AI, doi:10.1148/ryai.2019190015): el óptimo depende del hallazgo (radiografía). Es
  un experimento de resolución (exp057/P-1), no de preprocesado.

---

## 4. Veredicto sobre los 3 brazos planeados

| Brazo | Justificación | Coste EF | Recomendación |
|---|---|---|---|
| CLAHE re-test (mejor padre) | baja: 2 corridas propias ±0.002, Mills interno peor | ~0.0004 AUC eq. (16 ms/serie) | baja prioridad; si se corre, declarar que es réplica y usarlo para ensamble |
| Nyúl | baja-media: estándar en MRNet/Vuskov, mejor en externo en rodilla; ≈0 in-dist | ≈0 con LUT | **sí**, con landmarks por slot y foreground, más P-4 (z-score) como control; métrica secundaria harness exp040 |
| N4 | **mala**: evidencia nula/negativa en DL y ~2 h de CPU en el test | ≈ −0.05 a −0.10 AUC eq. | **no**; sustituir por AUG-BIAS |

**Faltaba** en la cola: P-1 (recorte centrado en la rodilla) y P-2 (escala física sin ampliar). Ambos cuestan ~0 en
inferencia, atacan el único paso que la literatura considera esencial (geometría/espaciado) y P-1 abre la puerta a 224 px
con menos pérdida. Orden sugerido: D-1/D-2/D-3 (CPU, 1 h) → P-2 o P-1 según diagnóstico → Nyúl → (P-4) → P-5.

---

## 5. Diagnósticos de CPU (antes de gastar GPU)

| id | Qué medir | Sobre | Decide |
|---|---|---|---|
| D-1 | % series por slot con FOV (`min(rows, cols) · PixelSpacing`) < 140 mm; % píxel no cuadrado | cabeceras DICOM de train (sin píxeles) | P-2 |
| D-2 | desplazamiento centroide de foreground – centro de matriz (mm) por slot; p50/p90 | 300 estudios del corpus o DICOM | P-1 y el tamaño de recorte |
| D-3 | tabla de `ImageOrientationPatient` redondeado por plano | cabeceras | P-6 |
| D-4 | cobertura física (mm) y nº de cortes por slot; % con nº de cortes < k | cabeceras | P-5 |
| D-5 | tiempo de cada paso nuevo en el notebook de inferencia de Kaggle, por estudio | 20 estudios de train en Kaggle | umbral 200 ms/estudio |

## 6. Claves bib nuevas

`nyul1999standardizing`, `nyul2000variants`, `tustison2010n4`, `kondrateva2024negligible`, `deraad2021preprocessing`,
`mills2026kneenorm`, `madaan2023normschemes`, `kushol2023scanner`, `carre2020standardization`, `isensee2021nnunet`,
`shinohara2014whitestripe`, `astuto2021knee`, `xie2025ninekneeabn`, `manjon2010nlm`, `dabov2007bm3d`, `zhang2017dncnn`,
`chaudhari2018deepresolve`, `lee2018wso`, `sabottke2020resolution`. Ya existentes que se citan aquí: `bien2018mrnet`,
`vuskov2025multicondition`, `qiu2024copas`, `zuiderveld1994clahe`, `pizer1987ahe`, `perezgarcia2021torchio`,
`reinhold2019normalization`, `ouyang2022gin`, `zhang2020bigaug`.
