# exp002_dinov2s_bblr1e5

| Campo | Valor |
|---|---|
| Autor | rodrigo |
| Padre | exp001 |
| Estado | worked |

Registrado después de correr (antes de adoptar SDD). Split viejo: fold 0 generado dentro del run.
El notebook `exp002_dinov2s_bblr1e5_legacy.ipynb` es el original y usaba `src/knee/train.py`, que ya no existe;
se conserva como registro. Su receta vive ahora en exp003.

## 1. SPEC
**Variable cambiada:** lr del backbone 1e-4 → 1e-5 (cabeza sigue en 1e-4); épocas 5 → 8.

## 3. VERDICT
| Métrica | Padre | Este |
|---|---|---|
| pseudo AUC fold 0 | — | 0.828 |
| gold AUC | 0.60 | 0.781 |
| Tiempo | | ~24 min en RTX PRO 6000 |

**Qué aprendimos:** lr bajo en el backbone estabiliza el fine-tuning. Es la receta base.
Pesos en `results/exp002_dinov2s_bblr1e5/`.
