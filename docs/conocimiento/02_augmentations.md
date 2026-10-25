# 02 — Augmentations: qué probamos, qué falló y por qué, y las de RM que faltan

Fecha: 2026-10-12. Fuentes: `local/RSNA_papers/04_robustness_aug_ensemble/RESUMEN.md` (+PDFs, extraídos con `pdftotext`),
`local/RSNA_papers/analisis/A2_implementacion_vs_papers.md` (§1, §2.1–2.3), writeups Kaggle de `03_mil_multiseq_arch/`,
y la celda 8 del notebook de exp045 (`git show exp045_raptor_aug_espacial:experiments/exp045_raptor_aug_espacial/exp045_raptor_aug_espacial.ipynb`).
Convención: **[paper]** = textual o paráfrasis del PDF; **[inferencia]** = razonamiento nuestro, no verificado;
**[no en corpus]** = conocimiento general sin PDF en la carpeta. Números propios: de los verdicts (vía A2).

---

## 0. Resumen en 10 líneas

1. Lo único que funcionó es **geométrico**: exp045 (espejo L/R anatómico p=0.5 + afín leve, misma matriz por estudio) **+0.0063** pseudo, OOF 5f 0.8844.
2. Todo lo de **intensidad** (exp046 fuerte +0.0018, sub046_1 suave +0.0014), CLAHE, mixup, slot dropout: `didnt`. Contra la media de 2 semillas del padre (0.8844) son **nulos** (A2 §1.3).
3. Pero esos "didnt" tienen **sospechas de diseño**: p=1 para gamma/contraste, mismos parámetros para las 5 secuencias, 16 épocas sin sobreajuste, y una vara in-distribution que no mide robustez (A2 §1.1–1.4).
4. **Nunca probamos artefactos de adquisición de RM** (motion, ghosting, spike, anisotropía/grosor de corte, bias field por secuencia) ni domain randomisation (BigAug apilado, GIN, SynthSeg-style).
5. El argumento para probarlos: el test oculto es multi-centro; en exp040 el sitio croata da 0.817 vs inglés 0.880 (A5 R8). Las augs de artefacto atacan el shift de **adquisición**, no el de contraste global.
6. Prior honesto: baja para la AUC in-distribution (pseudo f0). Valor esperado mayor en **robustez por sitio** y en el **paper** (ablación "artefactos vs intensidad", hueco no publicado en rodilla — RESUMEN 04, "Huecos" #3).
7. Toda aug nueva va **en GPU, por estudio, después de `spatial_aug`** en la celda 4/8 (entrenamiento), sin tocar el DataLoader (que es el cuello: I/O).
8. Padre por defecto: **exp045**. Para efectos esperados < +0.003: padre con 2 semillas (A2 §1.3).
9. Evaluar **siempre** también con el harness de robustez de exp040 (ruido, sitio croata) — ~0.2 GPU-h por checkpoint (A2 §1.4).
10. Orden sugerido: (a) arreglos baratos de lo que ya funciona (afín sobre 384), (b) anisotropía/grosor, (c) motion+ghosting, (d) bias field por secuencia p=0.3, (e) BigAug apilado, (f) GIN.

---

## 1. Dónde y cómo se aplica hoy la aug (exp045)

Celda 2 (parámetros):
```python
AUG_FLIP_P = 0.5          # espejo L/R anatómico por estudio
AUG_ROT_DEG = 10.0        # rotación U(-10°, 10°)
AUG_SCALE = [0.9, 1.1]    # escala
AUG_TRANS = 0.06          # traslación ±6 % del lado
```
Celda 8 (`StudyWindows` + `spatial_aug`), resumen:
- `StudyWindows.__getitem__` devuelve `tri` **uint8 (K, 3, RES, RES)**: K=12 ventanas 2.5D (3 cortes consecutivos → RGB), recorte **central 320 de un corte de 384**, más `gain` ×U(0.9, 1.1) y `sag` (bool por ventana: el corte central es de un slot sagital 0/1).
- `SLOT_SIZES = [26, 22, 18, 12, 18]` (Sag fluid, Sag, Cor fluid, Cor, Ax) → 96 cortes; `SLOT_OF[c]` da el slot de cada corte. **Las ventanas de un batch mezclan slots** (no hay identidad de slot en el modelo).
- En el loop de entrenamiento: `x` (B, K, 3, H, W) en GPU → `x = spatial_aug(x, sag)` (**solo train**) → ganancia → normalización ImageNet (`_MEAN`, `_STD`).
- `spatial_aug`: espejo por estudio (`x.flip(2)` si sagital = invertir orden de cortes; si no `x.flip(-1)`), luego **una theta por estudio** repetida a las K ventanas (`theta.repeat_interleave(K_)`), `F.affine_grid` + `F.grid_sample(padding_mode="zeros")`, con p=1.

**Punto de inserción para augs nuevas** [inferencia]: inmediatamente después de `spatial_aug` y antes de ganancia/normalización ImageNet, sobre `x` en float [0,1] (dividir el uint8 entre 255 si aún no se hizo). Toda aug debe:
1. muestrear parámetros **por estudio** (B) — o **por estudio × slot** cuando simule un efecto de secuencia (ver §3);
2. aplicarse con **p por transformación** (0.1–0.3), no p=1;
3. respetar la estructura 2.5D: el canal C=3 son **cortes vecinos**, no colores. Un artefacto "3D" (grosor de corte) se simula sobre ese eje.

Helper común que usan los snippets (por estudio → por ventana, y por slot):
```python
def per_study(v, K):            # v: (B, ...) -> (B*K, ...)
    return v.repeat_interleave(K, 0)

def slot_of_windows(cs_slots):  # cs_slots: (B, K) int con SLOT_OF[cs]; hay que devolverlo desde __getitem__
    return cs_slots              # igual que hoy se devuelve `sag`
```
Para augs por slot hay que devolver `SLOT_OF[cs]` desde `__getitem__` (análogo a `sag`). Es un cambio de plumbing, no una segunda variable.

---

## 2. Diagnóstico de lo ya probado (resumen de A2)

| Exp | Qué | Δ pseudo vs padre | Problema probable | Re-prueba (1 variable, padre exp045) |
|---|---|---|---|---|
| exp045 | espejo + afín leve | **+0.0063 worked** | relleno con **ceros** tras recorte 320 (se tiran 32 px reales por lado) | **afín sobre 384 y recorte 320 después** (~2.6 GPU-h) |
| sub045_1 | afín fuerte (±20°, 0.8–1.2, ±10 %) | −0.0023 didnt | **p=1** + ceros. La literatura usa magnitudes fuertes con p 0.2–0.5: AnyMC3D (nnU-Net) rot ±30° p0.2, zoom 0.7–1.4 p0.2 [paper, arXiv en A1]; Lumbar 1.º `ShiftScaleRotate(0.1, 0.1, 20, p=0.5)` [Kaggle] | afín fuerte **p=0.3** sobre 384 (después de la anterior) |
| exp046 / sub046_1 | gamma U(0.7,1.5) + contraste p=1; bias poly grado 2 ≤0.15 p0.3; blur σ0.5–1 p0.25; ruido σ≤0.05 p0.5 | +0.0018 / +0.0014 didnt | gamma/contraste **p=1**; **mismos parámetros para las 5 secuencias**; aplicado sobre datos ya saturados a p2–98; no se evaluó con harness exp040 | (a) harness exp040 sobre `best_f0.pt` (0.3 GPU-h, sin entrenar); (b) intensidad **por slot, p=0.3** |
| exp048 | mixup píxel (Beta 0.4, 50 % batches) | +0.0009 | mezcla ventana k de A con ventana k de B que puede ser **otro plano** | mixup en espacio de rasgos pooled o alineado por slot |
| exp047 | slot dropout p0.15 (+ jitter ±1) | +0.0007 | redundante con muestreo K=12; el MIL no sabe de qué slot viene; **2 variables** | solo tras embedding de slot |
| exp049 | CLAHE clip 2 | +0.0028 (−0.0018 sobre combo) | sin respaldo en literatura de RM | no prioritaria |

Lección transversal (A2 §1.1): mejor época = última o penúltima en 16 épocas → los regularizadores pagan el costo (curva más lenta: exp046 ep10 0.8787 vs 0.8813) sin cobrar. Cualquier aug fuerte debería probarse **también** con un padre largo (exp045 a 20–24 épocas, sub045_2 rehabilitado) — pero eso es otro experimento.

---

## 3. Augmentations de RM no probadas — fichas implementables

Formato de cada ficha: qué es · cita · parámetros en la literatura · implementación en nuestro pipeline · snippet · costo · riesgo · experimento SDD · criterio sugerido.

**Criterio estándar** (06_SDD, A5): padre exp045 fold 0, `SELECT_ON="pseudo"`. `worked` si Δ pseudo L2 f0 ≥ +0.003 (A5) — o ≥ +0.005 para "promete" según la regla 7 del repo; **usar la regla del repo** y declarar en la SPEC — y gold no cae > 0.02. **Métrica secundaria obligatoria** para todas estas: AUC en el harness exp040 (sitio croata, ruido) frente al padre; reportarla aunque no decida. Para el paper, una aug de robustez que sea neutra en pseudo pero mejore el sitio croata ≥ +0.005 se registra como hallazgo (no como `worked`) [inferencia].

### 3.1 Anisotropía / grosor de corte (`RandomAnisotropy`, SynthSeg-style resolution randomisation) — **prioridad 1**

- **Qué es.** Simula una adquisición con menor resolución en un eje (cortes más gruesos o más espaciados): se submuestrea a lo largo de un eje y se reinterpola de vuelta. SynthSeg modela dirección de adquisición, espaciado y grosor: primero desenfoca con un kernel gaussiano que aproxima el perfil de excitación del corte, luego submuestrea a `r_spac` con interpolación trilineal y re-sube a la resolución alta [paper: Billot 2023, MedIA, doi:10.1016/j.media.2023.102789, arXiv 2107.09559].
- **Por qué para nosotros.** SynthSeg muestra que aleatorizar resolución/grosor importa tanto como la intensidad (RESUMEN 04 #6). El protocolo de rodilla varía mucho entre centros en grosor de corte (2–4 mm) y matriz [inferencia/no en corpus]. Es una perturbación de **geometría/resolución**, la familia que sí funcionó (A5 #6).
- **Parámetros en la literatura.** TorchIO `RandomAnisotropy(axes=(0,1,2), downsampling=(1.5, 5))` por defecto [no en corpus: defaults de la doc de TorchIO; verificar en `torchio.readthedocs.io`]. SynthSeg: espaciado y grosor muestreados de uniformes; ver tabla de hiperparámetros del PDF para las cotas exactas (no extraídas hoy).
- **Implementación en nuestro pipeline (2 variantes, elegir UNA por experimento):**
  - **(a) In-plane** (más simple, más segura): por ventana, bajar H o W por factor f∈U(1.5, 3) con `F.interpolate(mode="area")` y volver a subir con `bilinear`. Simula matriz de adquisición menor / pixel más grande.
  - **(b) Through-plane** (fiel al grosor): el eje de cortes está en C=3 (vecinos) y en la secuencia de ventanas. Simular grosor = **promediar cada corte con sus vecinos** con pesos gaussianos σ∈U(0.5, 1.0) cortes. Sobre la ventana 2.5D: `x_c' = w0*x_c + w1*(x_{c-1}+x_{c+1})`. Requiere leer 5 cortes en vez de 3 para no perder el borde — o aceptar el borde [inferencia].
- **Snippet (variante a, por estudio, p=0.25):**
```python
def rand_anisotropy_inplane(x, p=0.25, f_rng=(1.5, 3.0)):
    # x: (B, K, 3, H, W) float en [0,1], GPU
    B, K, C, H, W = x.shape
    on = torch.rand(B, device=x.device) < p
    if not on.any():
        return x
    out = x.clone()
    for b in torch.nonzero(on).flatten().tolist():      # B=8: el bucle es despreciable
        f = float(torch.empty(1).uniform_(*f_rng))
        axis_h = bool(torch.rand(1) < 0.5)              # eje submuestreado, mismo para el estudio
        size = (max(8, int(H / f)), W) if axis_h else (H, max(8, int(W / f)))
        lo = F.interpolate(x[b], size=size, mode="area")
        out[b] = F.interpolate(lo, size=(H, W), mode="bilinear", align_corners=False)
    return out
```
- **Costo.** GPU despreciable (2 interpolaciones de 12×3×320² por estudio afectado); entrenamiento = 1 screen f0 ≈ **2.5 GPU-h** (A2 costos).
- **Riesgo.** Medio: puede borrar lesiones finas (menisco, fisuras condrales) si f es grande. Mantener f ≤ 3 y p ≤ 0.25. Prior in-distribution baja (A5 #6: +0.000 a +0.003).
- **Experimento SDD.** `nuevo_experimento("aug_anisotropia", padre="exp045")`. Única variable: añadir `rand_anisotropy_inplane(p=0.25, f∈[1.5,3])` tras `spatial_aug`. Todo lo demás idéntico (16 épocas, seed 42, fold 0).
- **Criterio.** Estándar + secundaria harness exp040. Sugerido: `killed` si pseudo cae ≤ −0.005.

### 3.2 Motion en k-space (`RandomMotion`, Shaw 2019) — prioridad 2

- **Qué es.** Simula movimiento del paciente durante la adquisición: se generan varias transformaciones rígidas, se "de-median", se remuestrea la imagen bajo cada una, se pasa cada versión a k-space y se compone el k-space final tomando de cada versión las líneas adquiridas en ese intervalo; vuelta a imagen [paper: Shaw et al., MIDL 2019, PMLR v102, proceedings.mlr.press/v102/shaw19a]. Movimiento en el centro de k-space → **ghosting**; en los bordes → **ringing** [paper, Shaw cita a Zaitsev 2015].
- **Parámetros en la literatura.** Shaw: rotación U(−30°, 30°), traslación U(−10 mm, 10 mm) en los tres ejes [paper, texto extraído]. Mejora robustez en imágenes artefactadas con caída **no significativa** en limpias [paper]. TorchIO `RandomMotion(degrees=10, translation=10, num_transforms=2)` por defecto [no en corpus: verificar doc].
- **Implementación 2D (por ventana, en GPU)** [inferencia]: usar `torch.fft.fft2` sobre (H, W). Para n=2 movimientos: generar una versión con afín rígido pequeño (reusar `F.affine_grid` con ángulo U(±3°) y traslación ±2 % — mucho menor que Shaw porque es 2D y la rodilla está inmovilizada en bobina), FFT de ambas, tomar las filas de k-space a partir de un índice de corte t∈U(0.3, 0.7)·H de la versión movida, IFFT y magnitud. **Misma decisión de movimiento para los 3 cortes de la ventana** y para todas las ventanas del mismo slot (el movimiento ocurre durante la adquisición de una serie, no entre series).
- **Snippet:**
```python
def rand_motion_kspace(x, p=0.15, max_deg=3.0, max_tr=0.02):
    B, K, C, H, W = x.shape
    on = torch.rand(B, device=x.device) < p
    if not on.any():
        return x
    xb = x[on].reshape(-1, C, H, W)                       # (Bon*K, 3, H, W)
    n = xb.shape[0] // K
    ang = (torch.rand(n, device=x.device) * 2 - 1) * math.radians(max_deg)
    tr = (torch.rand(n, 2, device=x.device) * 2 - 1) * max_tr * 2
    cos, sin = torch.cos(ang), torch.sin(ang)
    th = torch.stack([torch.stack([cos, -sin, tr[:, 0]], 1), torch.stack([sin, cos, tr[:, 1]], 1)], 1)
    th = th.repeat_interleave(K, 0)
    moved = F.grid_sample(xb, F.affine_grid(th, xb.shape, align_corners=False),
                          padding_mode="border", align_corners=False)
    k0, k1 = torch.fft.fftshift(torch.fft.fft2(xb), dim=(-2, -1)), torch.fft.fftshift(torch.fft.fft2(moved), dim=(-2, -1))
    t = int(H * float(torch.empty(1).uniform_(0.3, 0.7)))  # línea de k-space donde "se movió"
    k = k0.clone(); k[..., t:, :] = k1[..., t:, :]
    out = torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1))).abs().clamp(0, 1)
    x = x.clone(); x[on] = out.reshape(-1, K, C, H, W)
    return x
```
  Nota: en este snippet la decisión es por estudio (no por slot). La versión por slot requiere `slot` por ventana (§1).
- **Costo.** FFT de 12×3×320² por estudio afectado: ms en GPU. 1 screen ≈ 2.5 GPU-h.
- **Riesgo.** Medio-alto: puede borrar lesiones finas (A5 #6). Con p bajo y movimientos pequeños el riesgo es menor. La dirección de fase real de cada serie es desconocida en el corpus (se perdió al pasar a npy) → usar eje aleatorio por estudio [inferencia].
- **Experimento SDD.** `aug_motion`, padre exp045, única variable `rand_motion_kspace(p=0.15)`.
- **Criterio.** Estándar + harness exp040.

### 3.3 Ghosting (`RandomGhosting`) — combinable con 3.2 solo como experimento aparte

- **Qué es.** Réplicas desplazadas de la imagen por movimiento periódico (pulsación, respiración): se atenúan líneas periódicas de k-space a lo largo de la dirección de fase [paper: TorchIO, Pérez-García 2021, CMPB, doi:10.1016/j.cmpb.2021.106236, arXiv 2003.04696; mecanismo descrito también en Shaw].
- **Parámetros.** TorchIO por defecto `num_ghosts=(4, 10)`, `intensity=(0.5, 1)`, `restore=0.02` [no en corpus: verificar doc]. Para rodilla (poca respiración; pulsación poplítea en axial/sagital) usar intensidad baja (0.2–0.5) [inferencia].
- **Implementación.** FFT2 por ventana; multiplicar cada n-ésima fila de k-space (n = num_ghosts) por (1 − intensidad), preservando una franja central de ancho `restore·H`; IFFT, `.abs()`. Por estudio×slot, p=0.15.
```python
def ghost(xb, n_ghosts, inten, restore=0.02, axis=-2):
    k = torch.fft.fftshift(torch.fft.fft2(xb), dim=(-2, -1))
    H = k.shape[axis]; c = H // 2; r = max(1, int(restore * H))
    mask = torch.ones(H, device=xb.device); mask[::n_ghosts] = 1 - inten; mask[c - r:c + r] = 1
    shape = [1] * k.ndim; shape[axis] = H
    k = k * mask.view(shape)
    return torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1))).abs().clamp(0, 1)
```
- **Costo / riesgo / SDD.** Igual que 3.2. Experimento: `aug_ghosting` (padre exp045). **No** juntar con motion en el mismo experimento (regla 2); si ambos dan ≥ 0, un tercer experimento "motion+ghosting apilado" (BigAug-style, §3.6).

### 3.4 Spike / herringbone (`RandomSpike`) — prioridad baja

- **Qué es.** Puntos de k-space corruptos (gradientes a alto duty cycle) → franjas uniformemente espaciadas ("herringbone", "corduroy") [paper: TorchIO §, texto extraído; usado por Shaw para incertidumbre heteroscedástica].
- **Parámetros.** TorchIO `num_spikes=1`, `intensity=(1, 3)` por defecto [no en corpus].
- **Implementación.** FFT2; sumar en 1–2 posiciones aleatorias de k-space un valor `intensity · max|k|`; IFFT. Por ventana de un slot, p=0.05–0.1.
- **Riesgo.** Artefacto raro en clínica moderna; poco probable que exista en test. Valor esperado bajo [inferencia]. **Solo como componente de un apilado BigAug**, no como experimento propio.

### 3.5 Bias field **por secuencia** (`RandomBiasField`, AdvBias) — prioridad 3

- **Qué es.** Inhomogeneidad de intensidad de baja frecuencia (bobina de superficie: en rodilla la intensidad cae lejos de la bobina) [paper: Chen 2020, MICCAI, doi:10.1007/978-3-030-59710-8_65, arXiv 2006.13322: modelo multiplicativo `I_bias = I · bias(c)` con puntos de control]. SynthSeg: volumen pequeño 4³ de gaussiana de media 0 y SD aleatoria, se sube a tamaño completo y se exponencia voxel a voxel [paper].
- **Ya probado parcialmente:** exp046 tenía bias polinómico grado 2 ≤0.15 p0.3, pero **mismo campo para las 5 secuencias** y mezclado con gamma p=1 → no aislado.
- **Diferencia con lo probado.** Un **campo distinto por slot** (cada serie se adquiere con su propia configuración de bobina/ganancia), sin gamma. La versión adversarial (AdvBias: optimizar los puntos de control para maximizar la pérdida) dobla el costo de cada paso; dejarla para después.
- **Snippet (SynthSeg-style, por estudio×slot):**
```python
def rand_bias_field(x, slot, p=0.3, sd_max=0.3, grid=4):
    # slot: (B, K) int; mismo campo para todas las ventanas del mismo (estudio, slot)
    B, K, C, H, W = x.shape
    out = x.clone()
    for b in range(B):
        for s in slot[b].unique().tolist():
            if torch.rand(1).item() >= p:
                continue
            sd = float(torch.empty(1).uniform_(0, sd_max))
            small = torch.randn(1, 1, grid, grid, device=x.device) * sd
            field = torch.exp(F.interpolate(small, size=(H, W), mode="bicubic", align_corners=False))[0, 0]
            m = slot[b] == s
            out[b, m] = (x[b, m] * field).clamp(0, 1)
    return out
```
- **Costo.** Despreciable en GPU. 1 screen ≈ 2.5 GPU-h.
- **Riesgo.** Bajo-medio. Prior baja por exp046, pero la versión por slot y aislada nunca se midió.
- **Experimento SDD.** `aug_bias_por_slot`, padre exp045, única variable. Criterio estándar + harness exp040.

### 3.6 BigAug / Deep Stacked Transformation (apilado) — prioridad 4 (después de tener 1–2 componentes no negativos)

- **Qué es.** n transformaciones apiladas en orden fijo (sharpening, blurring, noise, brightness, contrast, intensity perturbation, rotation, scaling, deformation, crop), cada una con **probabilidad p y magnitud m** [paper: Zhang L. et al., IEEE TMI 39(7):2531–2540, 2020, arXiv 1906.03347]. Con dataset fuente pequeño, en dominios no vistos el Dice cae solo 11 % frente a 39 % con aug convencional [paper, RESUMEN 04].
- **Parámetros en la literatura.** gamma ∈ [0.5, 4.5]; ruido std [0.1, 1.0]; blur gaussiano std [0.25, 1.5]; perturbación de intensidad scale/shift ∈ [−0.1, 0.1]; **p=0.5 por transformación** [paper, citado textual en A2 §2.3]. En RM 3D lo que más pesó fue calidad/apariencia, **sharpening el mayor** [paper, texto extraído].
- **Nota.** BigAug apunta a **dominio no visto**; nuestra vara es in-distribution (A2 §1.4). Su valor solo se ve con harness exp040 o LB.
- **Implementación** [inferencia]: función `stacked_aug(x, slot)` que aplica en orden: sharpen (unsharp mask, p0.2) → blur (σ U(0.25,1.0), p0.2) → ruido (σ U(0, 0.05), p0.2) → bias por slot (p0.2) → gamma por slot (U(0.7,1.5), p0.2) → anisotropía (p0.2) → (motion p0.1). Magnitudes **más suaves** que BigAug (nuestro régimen de 16 épocas penaliza la aug fuerte, A2 §1.1).
- **Costo.** 1 screen 2.5 GPU-h; con padre largo 3.2–3.8.
- **Riesgo.** Alto in-distribution (curva más lenta). Mejor con padre de 20–24 épocas.
- **Experimento SDD.** `aug_bigaug_suave`, padre exp045 (o el padre largo si existe). Una variable = "el bloque apilado completo".

### 3.7 GIN (+ IPA) — domain randomisation de apariencia — prioridad 5

- **Qué es.** Red convolucional superficial con **pesos aleatorios re-muestreados cada iteración** (convs intercaladas con LeakyReLU), salida interpolada linealmente con la imagen original y re-escalada a la misma norma de Frobenius; IPA mezcla espacialmente dos versiones GIN con un mapa de pseudo-correlación interpolado desde puntos de control aleatorios [paper: Ouyang et al., IEEE TMI 2022, doi:10.1109/TMI.2022.3224067, arXiv 2111.12525]. Cambia apariencia/textura local preservando forma. Mejor DG de fuente única en TMI (RESUMEN 04 #5).
- **Parámetros.** Kernel pequeño (1×1 y 3×3), pocas capas, canales intermedios aleatorios, α∈U(0,1) para la interpolación [paper, figura 2; cifras exactas no extraídas hoy → verificar en §III del PDF].
- **Implementación** [inferencia]: por estudio×slot, sobre las 3 "canales" (cortes) como entrada de una conv aleatoria 3→c→3; aplicar a la ventana completa. Cuidado: mezclar canales = mezclar cortes vecinos (aceptable, es local).
```python
def gin(xb, n_layers=4, ch=8, k=3):
    h = xb
    cin = xb.shape[1]
    for i in range(n_layers):
        cout = 3 if i == n_layers - 1 else ch
        w = torch.randn(cout, cin, k, k, device=xb.device) * (1.0 / (cin * k * k) ** 0.5)
        h = F.conv2d(h, w, padding=k // 2)
        if i < n_layers - 1:
            h = F.leaky_relu(h, 0.2)
        cin = cout
    a = torch.rand(xb.shape[0], 1, 1, 1, device=xb.device)
    y = a * h + (1 - a) * xb
    y = y * (xb.flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1).clamp_min(1e-6)).view(-1, 1, 1, 1)
    return y.clamp(0, 1)
```
- **Costo.** Despreciable. **Riesgo.** Medio: puede chocar con el fracaso de intensidad (A5 #E en RESUMEN 04). p=0.3.
- **Experimento SDD.** `aug_gin`, padre exp045.

### 3.8 Gamma/contraste **por slot y p=0.3** (re-prueba de exp046) — prioridad 3 (empata con 3.5)

- Ver A2 §2.3(b). Diferencia con exp046: parámetros **por slot** (no compartidos entre las 5 secuencias) y **p=0.3** (no 1). Rangos: los de sub046_1. Referencia: nnU-Net gamma 0.7–1.5 p0.3 [no en corpus]; AnyMC3D noise p0.25, blur p0.2 [paper].
- Antes de entrenar: pasar `best_f0.pt` de exp046/sub046_1 por el harness exp040 (0.3 GPU-h, A2 ranking #2). Si exp046 ya mejora el sitio croata, sube la prior.

### 3.9 Arreglos de lo que funciona (no son augs nuevas, pero van primero)

| id | Cambio | Por qué | Costo |
|---|---|---|---|
| AUG-F1 | **Afín sobre el corte de 384 y recorte central 320 después** | Hoy el afín mete ceros donde había 32 px reales (A2 §2.1–2.2) | 2.6 GPU-h; ×1.44 bytes CPU→GPU |
| AUG-F2 | Afín fuerte con **p=0.3** sobre 384 (tras F1) | sub045_1 falló con p=1 + ceros | 2.6 GPU-h |
| AUG-F3 | **Resize 384→320** en vez de recorte | Separa FOV de resolución (A2 §2.7) | 2.5 GPU-h |
| AUG-F4 | `padding_mode="border"` o `"reflection"` en `grid_sample` (alternativa barata a F1 si F1 no cabe en I/O) | quita el artefacto de borde negro | 2.5 GPU-h |

### 3.10 Consistencia entre secuencias y cortes — reglas de diseño

- **Geometría:** una transformación **por estudio** (ya lo hacemos; Darragh, C-spine 3.º: *"the same augmentation … was applied to the study"* [Kaggle, `kaggle_2022_cspine_3rd_Darragh.md`]). El espejo debe ser anatómico (invierte M/L en todo el estudio a la vez).
- **Artefactos de adquisición e intensidad:** **por serie (slot)**, porque cada serie es una adquisición separada con su propio contraste, bobina, grosor y movimiento [inferencia, fundamentado en A2 §2.3 punto 2].
- **Eje de cortes:** la misma perturbación para los 3 cortes de una ventana y para todas las ventanas del mismo slot (un artefacto de adquisición es coherente a lo largo de la serie).
- **Espejo en TTA:** si entrenamos con espejo, la TTA espejo aporta menos; medirla (A2 §2.14).

### 3.11 Lo que no recomendamos

- Más gamma/CLAHE/normalización global sin cambio de diseño (A5 "Lo que NO").
- Mixup de píxeles entre bolsas no alineadas (exp048).
- Elastic deformation fuerte: borra la morfología de menisco/ligamento [inferencia]; BigAug la incluye pero en segmentación con 3D isotrópico.
- MedAugment/RandAugment automáticos [paper: Liu Z. 2023, arXiv 2306.17466]: alternativa barata, pero sin control de qué artefacto se simula → no sirve para la ablación del paper [inferencia].

---

## 4. Plan sugerido (orden, padre, costo)

| Orden | Experimento | Padre | GPU-h | Sirve a |
|---|---|---|---|---|
| 0 | Harness exp040 sobre exp046/sub046_1/exp047 (sin entrenar) | — | 0.6 | decide prior de 3.5/3.8 |
| 1 | AUG-F1 afín sobre 384 | exp045 | 2.6 | LB |
| 2 | 3.1 anisotropía in-plane p0.25 | exp045 (o F1 si worked) | 2.5 | paper (robustez), LB bajo |
| 3 | 3.2 motion k-space p0.15 | idem | 2.5 | paper |
| 4 | 3.5 bias por slot p0.3 / 3.8 gamma por slot p0.3 | idem | 2.5 c/u | paper |
| 5 | 3.6 BigAug suave apilado | padre largo 20–24 ép | 3.5 | paper, LB bajo |
| 6 | 3.7 GIN | idem | 2.5 | paper |

Total ≈ 17 GPU-h (~$13 a $0.74/h, A5). Para el paper (C3), la tabla resultante "geométrica vs intensidad vs artefacto de adquisición", con Δ pseudo, Δ sitio croata y Δ LB f0, es una ablación no publicada en rodilla [RESUMEN 04, Huecos #3].
