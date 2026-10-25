# 03 — Arquitectura MIL multi-secuencia

Fecha: 2026-10-12. Fuentes: `local/RSNA_papers/03_mil_multiseq_arch/RESUMEN.md` (24 PDFs + 8 writeups Kaggle), `01_knee_mri_sota/RESUMEN.md`, A2 §2.13, A5 #4 y #10.

## 1. Modelo actual (A2 §2.13)

- CoAtNet-2 (`coatnet_rmlp_2_rw_384`, timm) por ventana 2.5D (3 cortes → RGB), recorte 320.
- `RaptorClassifier`: LayerNorm → atención **tanh no gated** (Linear 256 → Tanh → Dropout 0.2 → Linear 12), **una atención por hallazgo**, softmax sobre K ventanas (12 train / 24 eval), lineal por hallazgo.
- **Sin embedding de slot ni de posición**; las 5 secuencias se mezclan en una bolsa. Supuesto fuerte: la atención infiere plano y secuencia por la apariencia.
- Receta de Kaggle (Raptor/nartaa, K1–K11 en `08_LO_QUE_HACEN_OTROS.md`); cada pieza tiene cita canónica en `referencias.bib` (A1).

## 2. Qué hacen la literatura y los ganadores

- **ABMIL gated** (Ilse 2018, ICML, arXiv 1802.04712): *"tanh(x) is approximately linear for x ∈ [−1, 1]"* → propone gating [paper, A2].
- **CoPAS** (Qiu 2024, Nat Commun, doi:10.1038/s41467-024-51888-4): ResNet3D-18 por plano + cross-attention entre secuencias, 12 anomalías de rodilla; AUC 0.812 interna, ~0.72 externa (no comparable con nuestro LB).
- **Vuskov 2025/26** (Eur Radiol, doi:10.1007/s00330-025-12052-8): ResNet18 2D + **bloques 3D inicializados a cero** + transformer (32 cortes), 23 condiciones.
- **AnyMC3D** (Liu 2025): FM 2D congelado + query pooling por vista, fusión tardía; desaconseja RNN por anisotropía.
- **MST** (Müller-Franzes 2024/25): slice transformer sobre DINOv2; rodilla 0.85 vs 0.69 de ResNet3D.
- **HeMIS** (Havaei 2016): fusión media/varianza por modalidad + drop aleatorio de modalidades.
- Kaggle: Lumbar 2024 1.º ABMIL → +biLSTM + pérdida aux sobre atención (logloss 0.35→0.33 con aux y ensamble); C-spine 2022 3.º congela la CNN y entrena RNN + atención (*"helped a lot"*); Abdominal 2023 2.º RNN sobre salidas congeladas y el mejor privado fue **un solo modelo**. **No conviene:** 3D-CNN desde cero, backbones más grandes (Lumbar 1.º: large < base < small), focal loss, épocas largas (Lumbar 1.º/3.º) [RESUMEN 03 §3].

## 3. Ideas implementables

| id | Idea | Cita | Costo | Riesgo | Nota |
|---|---|---|---|---|---|
| ARC-1 | **Caché de embeddings de corte congelados** por fold → probar agregadores en minutos | C-spine 3.º; Abdominal 2.º; AnyMC3D | ~1 GPU-h/fold extracción | bajo | Habilita ARC-2..5. Embeddings de train vienen de un backbone que **memorizó** esos estudios → entrenar agregador sobre OOF anidado o extraer con aug (A5 #4) |
| ARC-2 | **Embedding de slot aprendible** sumado antes de LayerNorm (5×768 params) | inferencia; prerequisito de slot dropout (A2 §2.4) | 2.5 | bajo | una variable, padre exp045 |
| ARC-3 | **Atención gated** (Ilse) | Ilse 2018 | 2.5 | bajo | |
| ARC-4 | **Agregación jerárquica**: corte → secuencia (query por slot) → estudio (transformer de 5 tokens), query por hallazgo | AnyMC3D; CoPAS; PRISM | 2.5 o minutos con ARC-1 | bajo | interpretabilidad por secuencia (figura 6 del paper) |
| ARC-5 | **Cross-attention con sagital como query** (CoPAS-style) | Qiu 2024 | idem | medio | baseline que pedirá el revisor (A5 R9) |
| ARC-6 | **biGRU por slot + ABMIL + aux sobre atención** | Lumbar 1.º; Abdominal 1.º | ~2.5 | medio | Lumbar 3.º: LSTM no funcionó |
| ARC-7 | **HeMIS** (media/varianza de slots presentes) | Havaei 2016 | 2.5 | medio | protocolos variables / secuencias faltantes |
| ARC-8 | **Bloques 3D a cero** entre etapas de CoAtNet | Vuskov | +10–20 % cómputo | medio | arranca idéntico al padre |
| ARC-9 | **FPN / pooling multiescala** para lesiones pequeñas | MRPyrNet (Dunnhofer 2021) | 2.5 | bajo | fractura/contusión/menisco |
| ARC-10 | **Recorte ROI de rodilla** (coarse-to-fine) + jitter del recorte en train | Xie 2025; Astuto 2021; ganadores de 2 etapas; Lumbar 1.º: jitter "crucial" | 3–4 + reconstruir cache | alto en 10 días | recupera resolución en el modelo de 224 (exp027 −0.007 gold) |
| ARC-11 | **FM congelado (DINOv3, MRI-CORE) + agregador** como baseline | Dong 2025; AnyMC3D; MST | 1–6 | prior baja para LB (exp006 DINOv3 killed) | baseline del paper (R9), después del 22-oct |
| ARC-12 | Edad/sexo concatenados al embedding (si el DICOM lo trae y las reglas lo permiten) | MeniOmni; OrthoDiffusion | muy bajo | verificar metadato | |
| ARC-13 | SSL en dominio (MAE sobre cortes train+test) | KneePreM; SB-SSL | días GPU | alto | después del deadline |

Hueco para el paper: **no hay benchmark controlado de agregadores** para RM de rodilla multi-secuencia multi-etiqueta en datos públicos grandes (RESUMEN 03 §4.1). ARC-1 + ARC-2..7 con el mismo backbone congelado es barato y reproducible.
