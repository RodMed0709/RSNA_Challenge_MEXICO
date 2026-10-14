# Bitácora exp025_perfil_raptor

- 2026-10-11 · `kernels pull nartaa/rsna-knee-0949-anatomical-mirror` (1 celda de código de 95 KB con
  `_fastread` en base64, variante `study_lut`, FR_HDR/FR_PX = 32/16, `nproc = max(4, min(96, cpu−8))`).
  El velci-kcd25-20 trae **el mismo** `_fastread` (md5 idéntico) y el mismo brazo Raptor; sus otros
  miembros (estudiante 224) no se perfilan aquí.
- Pesos bajados a `/workspace/data/public_assets/nartaa__rsna-knee-publication-swa-weights-20261007/`
  (r384 y c08_r224, 293 MB cada uno). Claim `/workspace/users/.claims/exp025_perfil_raptor`.
- Smoke 4 estudios: lectores instrumentados idénticos byte a byte al notebook. `posix_fadvise(DONTNEED)`
  no cambia los tiempos: MooseFS no deja caché de página, toda lectura va a la red (como en frío).
- Corridas en rsna-d (scripts y JSON crudos en `/workspace/users/rodrigo/_exp025/`): serial 112 h,
  serial 4 h (`taskset -c 0-3`), pool 112, pool 4, gpubench. ~20 min en total, sin procesos colgados.
- Resultado: lectura+decode+norm = 41 % del total serie con `study_lut` → hipótesis rechazada; en
  Kaggle (T4) el forward domina todavía más. Detalle en la SPEC §3.

## Optimizaciones priorizadas (ahorro estimado en s de reloj por estudio en Kaggle, 2×T4 + 4 CPU)

Supuestos: forward TTA ≈ 1.6 s/estudio por T4 (2 T4 → 0.8 s de reloj), prep 0.3 s/estudio solapada.
×1300 estudios para pasar a minutos. Solo cuenta el lado que es cuello de botella (GPU).

1. **channels_last** en modelo y ventanas: −12 % medido en PRO 4500 → ≈ −0.10 s/estudio (≈ −2 min).
   Riesgo de AUC nulo (misma aritmética fp16). Primero de la cola: exp026/027.
2. **TTA espejo parcial o sin TTA**: sin TTA = −50 % del forward → ≈ −0.40 s/estudio (≈ −9 min). Cuesta
   AUC (medir Δ en gold/pseudo; el autor eligió el espejo por LB). Intermedio: espejo solo en las
   ventanas sagitales o solo en un subconjunto de ventanas → −0.15 a −0.25 s.
3. **Menos ventanas** (K_EVAL 94 → 47, paso 2): −50 % del forward → ≈ −0.40 s/estudio; el MIL ve la
   mitad de cortes, riesgo de AUC moderado. Combinable con 2 solo si el AUC aguanta.
4. **TensorRT / torch.compile** (si el contenedor offline los trae): típico −20–30 % en T4 →
   ≈ −0.15–0.25 s/estudio. Validar que compile no se coma el ahorro en arranque.
5. **Destilar a un estudiante más barato** (p. ej. el 224 de velci o ConvNeXt compacto) → −50–80 % del
   forward; es otro proyecto, no un flag.
6. Lado CPU (solo si Kaggle resulta tener I/O ≥ 2.5× más lento que RunPod, ver exp026):
   - resize + normalización en GPU (subir el recorte uint16 y aplicar la tabla allí): −0.08 s CPU.
   - percentil por submuestra en vez de `bincount` de 65 536 por corte: −0.03 s CPU (rompe identidad).
   - leer solo los cortes elegidos ya está hecho (89 de 151 archivos); las cabeceras son 4 KB por
     pread → poco margen (0.05 s).
   - nproc: con 4 CPU el notebook usa 4 procesos × 32/16 hilos; más procesos empeoran en RunPod.
7. Descartados por la medida: trozos más grandes (32/47/94 son más lentos que 16) y meter normal+espejo
   en un solo batch (0.405 vs 0.402 s, igual).
