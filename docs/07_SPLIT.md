# Split canónico `split_v1_seed42`

Un solo split para todo el equipo durante toda la competencia. Lo genera `experiments/exp001_preparacion/04_split_v1.ipynb`
y vive en git como `SPLITS/split_v1_seed42.csv` + `SPLITS/split_v1_seed42.json` (hash, conteos, método).
`load_split()` revisa el hash: si alguien edita el CSV, el notebook se detiene.

## Qué contiene

| fold | Estudios | Uso |
|---|---:|---|
| 0–4 | 865–877 cada uno | Entrenamiento y validación cruzada |
| -1 | 58 | Gold (etiqueta experta). Nunca se entrena; solo veta |
| -2 | 1 | Agrupado con un gold (probable mismo paciente). Fuera de todo |

Columnas: `StudyInstanceUID, fold, is_gold, group_id, vendor, field, lang, PatientSex, ManufacturerModelName`.

## Por qué así

- **Pacientes:** `PatientID` es distinto en cada uno de los 4,407 estudios y edad, fechas, peso e institución vienen
  vacíos, así que no se puede agrupar por paciente. Como protección unimos los informes largos (>400 caracteres)
  casi idénticos (coseno TF-IDF ≥ 0.9) con el mismo equipo y el mismo sexo: 89 pares, 31 grupos. Incluye pares de
  rodilla izquierda/derecha de la misma persona. Cada grupo cae completo en un solo fold.
- **Estratificación:** `MultilabelStratifiedKFold` sobre grupos con los positivos de las 12 pseudo-etiquetas
  (0.5 y NaN cuentan como 0), fabricante (Siemens, Philips, GE, otro), campo (1.5T, 3T, otro) e idioma del informe
  (en, es, tr, fr, otro), que aproxima el hospital. Resultado: la prevalencia de cada etiqueta varía ±0.003 entre folds
  y la mezcla de equipos e idiomas es casi idéntica.
- **Gold fuera antes de partir**, para que los folds queden parejos y gold nunca se filtre.

## Reglas

- No se edita. Si cambian las etiquetas o el agrupamiento, se crea `split_v2_*` con otro notebook y se anota aquí.
- Resultados con splits distintos no se comparan: se vuelve a correr la referencia con el split nuevo.
- La semilla del split es fija (42). La semilla de entrenamiento (`SEED`) puede cambiar en la etapa final de ensemble.
