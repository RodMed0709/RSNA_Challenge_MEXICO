# Lo que hacen otros equipos (notebooks públicos, 24 sep)

Revisamos tres notebooks públicos. Ninguno entrena: cargan pesos que otros publicaron como datasets de Kaggle y los combinan.
Aun así muestran qué funciona.

## Dónde está el nivel

- Leaderboard público (premio de eficiencia): el top 100 está entre **0.93 y 0.958** de AUC; ~0.94 es lo normal.
- Nuestra referencia exp002: gold **0.777** (fold 0). La diferencia no se cierra afinando hiperparámetros: hay que cambiar datos, entrada y cabeza.
- `rsna-knee-speedy-raptors-coatnet-d4-0943`: 0.943 público combinando ~36 checkpoints públicos (CoAtNet + DINOv2 + RadImageNet), en menos de 30 min.

## Qué hacen distinto (por impacto esperado)

| # | Qué | Ellos | Nosotros hoy |
|---|---|---|---|
| 1 | Resolución y recorte | Recorte físico de 130 mm al centro, 336–384 px (~0.39 mm/px) | Imagen completa reducida a 224 px |
| 2 | Izquierda / derecha | Voltean rodillas derechas (coronal/axial en horizontal, sagital invirtiendo el orden de cortes). Lado por `Laterality` o por la posición x del centro (falta en la mitad) | Sin alinear |
| 3 | Selección de series | 6 "slots" fijos: SAG/COR/AX líquido con supresión de grasa, SAG líquido sin supresión, COR T1, SAG T1. Tipo de secuencia sacado de TR/TE y descripción | Primeras 5 series por prioridad simple |
| 4 | Entrada 2.5D | Cada imagen = 3 cortes vecinos como RGB; muchas ventanas por estudio (62–94) | 16 cortes sueltos por serie, canal repetido |
| 5 | Cabeza | Atención **por etiqueta** sobre ventanas/slots (el LCA mira sagital, el LCM coronal); prior por etiqueta opcional | Un solo vector para las 12 etiquetas |
| 6 | Pooling en test | Por etiqueta: `max` para fractura, contusión, meniscos, Baker; `top2` para LCA/LCM; media para el resto | Media |
| 7 | Etiquetas | "No mencionado" = 0.28 con peso bajo; pesos por confianza; gold con peso 3 (no lo copiamos: gold es nuestra única validación) | "No mencionado" = 0.5, sin pesos |
| 8 | Optimización | lr cabeza 1e-3, backbone 8e-6, OneCycle, solo 6 bloques de DINOv2 entrenables | cabeza 1e-4, backbone 1e-5, todo entrenable |
| 9 | Ensemble | Promedio de checkpoints (top-3 épocas / SWA), mezcla por ranking, TTA de ventanas y volteo de profundidad | Un checkpoint, promedio de probabilidades |
| 10 | Backbone | CoAtNet `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (timm) y DINOv2-small | DINOv2-small |
| 11 | Velocidad | Bloques congelados calculados una vez para todos los miembros; 2 GPUs T4 en paralelo; bf16 | Sin medir |

Lo que ya hacemos bien: etiquetas con LLM (su propio notebook dice que el LLM le gana a sus reglas), más cortes por serie,
split con agrupación de informes casi idénticos y gold separado.

## Cómo usarlo

Cada fila es candidata a experimento. El backlog priorizado está en `docs/05_PLAN_DE_MODELOS.md`.
Una fila = una variable = un experimento con su SPEC (`docs/06_SDD.md`).
