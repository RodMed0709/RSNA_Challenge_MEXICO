# Paso 3 · Entrada v2 (diseño)

Pregunta del paso: ¿qué entrada le damos al modelo? Se corre sobre el mejor resultado del paso 2 (`PADRE_P2`).
Fuentes: `docs/papers/RANKING.md`, `docs/papers/07_kaggle_ganadores`, `docs/papers/08_preprocesamiento_mri`, `docs/08_LO_QUE_HACEN_OTROS.md`.

## Metadatos por serie (`meta_v2.csv`)

Pasada solo de cabeceras (`stop_before_pixels=True`).

- **Plano**: normal = fila × columna de `ImageOrientationPatient`; mayor componente x/y/z → sagital/coronal/axial;
  < 0.8 → oblicua. No se usa el texto (12 idiomas).
- **Ponderación**: T1 si TR < 1000 y TE < 30; PD si TR ≥ 1500 y TE < 60; T2 si TR ≥ 1500 y TE ≥ 60;
  STIR si TI 100–250 ms o IR.
- **Supresión grasa**: `ScanOptions` (FS, SFS, SPAIR…), STIR, luego texto multilingüe, luego columna del csv.
- **Espaciado**: `PixelSpacing`; entre cortes, mediana de las diferencias de posición proyectada (no `SliceThickness`).
- **Lateralidad** en cascada: tag `Laterality`/`ImageLaterality` → texto L/R → geometría (centro físico en x fuera
  del isocentro, ±15 mm) → voto del estudio → `unknown` (no se espeja). Se valida contra el lado que dice el informe.

## Cache v2 (`/workspace/data/cache/v2_05mm/`)

- 0.5 mm/px en el plano, ventana de 160 mm (320 px) centrada en el centro del estudio, cortes nativos ordenados por
  posición (no por `InstanceNumber`), normalización igual que v1.
- Centro del estudio = mediana de los centros de FOV de sus series; respaldo: máscara Otsu. Un localizador solo si el
  QA de 100 estudios muestra > 5 % de fallos.
- Al leer: recorte (130 mm, jitter ±10 mm), espejado a rodilla derecha (coronal/axial: volteo horizontal; sagital:
  invertir orden de cortes) y resolución final. Así 3b–3d no reconstruyen cache.
- ≈ 84 GB. 1–3 h de reloj con 16 procesos; reanudable.

**Slots (6):** SAG líquido FS · COR líquido FS · AX líquido FS (resp. AX PD/T2) · SAG PD/T2 sin FS · COR T1 (resp.
COR PD) · SAG T1. Desempate: ORIGINAL > DERIVED, menor PixelSpacing, 15–60 cortes, mayor SeriesNumber. Fuera:
localizadores, < 5 cortes, orientación inconsistente.

`KneeDataset` gana `selector`, `mirror`, `crop_mm`, `center` con valores por defecto iguales a v1.

## Experimentos (una variable cada uno)

| Exp | Variable | Hipótesis | Cache |
|---|---|---|---|
| 3a | Selección de series: prioridad v1 → slots v2 | Series correctas suben la AUC y leen menos DICOM | v1 + meta_v2 |
| 3b | Espejado a rodilla derecha | Sube Medial/Lateral | v1 + meta_v2 |
| 3c0 | Paridad cache v2 sin recorte | Δ ≈ 0 (control) | v2 |
| 3c | Recorte físico 130 mm | Más detalle útil por píxel | v2 |
| 3d | Resolución 224 → 192 → 160 sobre el recorte | Misma AUC con 30–50 % menos cómputo | v2 |
| 3e | Centro por máscara o localizador (opcional) | Robustez del recorte | v2 |
| 3f | Recorte en profundidad ±65 mm (opcional) | Menos cortes inútiles | v2 |

Cada uno usa como padre el mejor anterior del paso.

## Qué medir para el paper

AUC por etiqueta (pares medial/lateral con y sin 3b), prueba de espejado sin intercambiar etiquetas, precisión de la
lateralidad por método, cobertura de slots por idioma del informe, DICOM leídos por estudio y costo por resolución.
