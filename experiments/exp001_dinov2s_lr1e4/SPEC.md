# exp001_dinov2s_lr1e4

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | — |
| Estado | didnt |

Registrado después de correr (antes de adoptar SDD). Split viejo: fold 0 generado dentro del run.

## 1. SPEC
Primer baseline: DINOv2-small por corte, atención sobre cortes y luego sobre series, 5 series, 16 cortes, 224 px,
lr 1e-4 para todo el modelo, 1 época con backbone congelado.

## 3. VERDICT
| Métrica | Valor |
|---|---|
| gold AUC | 0.64 congelado → 0.60 al descongelar |

**Qué aprendimos:** con lr 1e-4 el backbone se degrada al descongelarlo. Necesita un lr propio más bajo (exp002).
Pesos en `results/exp001_dinov2s_lr1e4/`.
