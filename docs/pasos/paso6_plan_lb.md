# Paso 6 — Plan para subir el LB (sesión autónoma 2026-10-11, ~30 h)

Mandato de Rodrigo: usar lo que ya gana (receta pública "Raptor" y notebooks de 0.949–0.954) como base, montarle todo lo que
nos funcionó, subir a Kaggle por cuenta propia y competir también en eficiencia. Publicabilidad no es prioridad en este paso.

## Recursos y reglas de la sesión

- Hasta 3 pods idénticos (RTX PRO 4500 32 GB, $0.34/h; si no hay, PRO 6000 96 GB con 2 runs por GPU). EU-RO-1 (volumen).
  Sin stock: reintentar cada 3 min.
- Vigía de saldo RunPod: si `clientBalance` < $4 se apagan todos los pods (Rodrigo recarga al recibir el correo de $5).
- Hasta 15 agentes en paralelo. Waiters por cada run.
- Envíos a Kaggle libres dentro del límite diario. Cada envío se registra en `docs/pasos/envios_kaggle.md`.
- SDD se mantiene: cada experimento con SPEC (hipótesis, una variable, criterio) y verdict. Los forks de Kaggle también.
- Sub-experimentos `subNNN_k` cuando un experimento deja de mejorar: barrido de hiperparámetros sobre el ganador (mismo SPEC
  padre, una variable por sub).

## Lo que aprendimos de los notebooks líderes (ver análisis en la sesión)

Raptor (`dreaddevelopment/raptor-knee-widedense`, 0.924 un modelo; `nartaa/rsna-knee-0949-anatomical-mirror` ~0.949):
etiquetas blandas multi-LLM (L2 + L3 + JEV + propias, incierto ≈ 0.8); 5 slots por plano/fluido con 96 cortes (2–98 %);
recorte 140 mm, 384 → 320 px, percentiles 2–98; 2.5D (3 cortes como canales, 94 ventanas); CoAtNet-2 384; atención MIL por
hallazgo; SWA + fine-tune con todos los datos; TTA espejo L/R anatómico. Ensamble por rangos con ConvNeXt-small
(LR-loss con prior) aporta ~+0.001–0.005. Cuello de botella en inferencia: decodificar DICOM (~40 s/estudio en el ConvNeXt).

Top del LB de premio: 0.963–0.965. Top 10 de eficiencia: AUC 0.952–0.960, el orden lo decide el runtime.

## Fases y experimentos

### Fase A — Base que gana (Kaggle, sin GPU propia)
| exp | qué | decide |
|---|---|---|
| exp023 | Reproducir los notebooks públicos tal cual (velci-kcd25-20, blend-mine, 0949-anatomical-mirror): LB y runtime en nuestro envío | referencia de LB y runtime |
| exp024 | Raptor solo (sin blend) con TTA espejo: AUC y runtime de un solo modelo | base de eficiencia |

### Fase B — Eficiencia sobre la base (perfilado en pod + Kaggle)
| exp | qué | decide |
|---|---|---|
| exp025 | Perfilado por etapa de la inferencia Raptor (decode, normalización, ventanas, modelo) | dónde está el tiempo |
| exp026 | Preprocesado rápido: normalización por percentiles aproximada/vectorizada, decode con menos copias, menos cortes leídos | runtime ↓ a igual AUC |
| exp027 | Menos ventanas / menos cortes por slot (96 → 64 → 48) | curva runtime–AUC |
| exp028 | fp16 + `channels_last` + `torch.compile` / ONNX-TensorRT en T4 | runtime de modelo |
| sub026_k–sub028_k | barridos de cortes por slot, resolución (320/288/256), hilos | punto óptimo de la métrica de eficiencia |

### Fase C — Entrenar la receta ganadora nosotros (pods)
| exp | qué | decide |
|---|---|---|
| exp029 | Reproducir el entrenamiento de Raptor con sus etiquetas públicas (fold 0) | ¿llegamos a ~0.92 LB? |
| exp030 | Raptor + nuestras etiquetas: blend L2+L3+L4(+L7) blandas | LB |
| exp031 | Raptor con entrada de nuestro cache v2 (recorte físico + alineado L/R) | LB / pseudo L2 |
| exp032 | Raptor con 12–16 épocas (exp015/exp020: la receta corta se queda corta) | LB |
| exp033 | SWA + fine-tune todos los datos sobre el ganador | LB |
| exp034 | Backbone alterno para diversidad (ConvNeXt-small / EfficientNetV2) en la misma receta | LB del ensamble |
| sub030_k–sub034_k | lr, pesos de etiquetas, prior de incierto (0.7/0.8/0.9), drop_path | LB |

### Fase D — Nuestras propias líneas en paralelo (lo que ya funcionó)
| exp | qué | decide |
|---|---|---|
| exp035 | Etiquetas combinadas L2+L3+L4 blandas sobre la receta exp021 (barata) | LB vs 0.836 |
| exp036 | Referencia propia v2 + mejores etiquetas + 12 épocas | LB |
| exp037 | exp036 + 96 cortes 2.5D | LB |

### Fase E — Ensamble y cierre
| exp | qué | decide |
|---|---|---|
| exp038–exp040 | Rank-blend de los mejores (pesos fijos elegidos con pseudo L2, nunca con gold) | LB |
| exp041–exp050 | Reservados para lo que salga (sub-experimentos, destilación a un estudiante rápido para eficiencia) | — |

## Validación interna

La pseudo contra v1 está desalineada con el test (exp021). La vara interna nueva es pseudo AUC contra etiquetas L2
binarizadas (> 0.5) en fold 0; gold solo veta. El LB público decide lo que va a envío final, cuidando no sobreajustar al
público: el veredicto final prefiere cambios que suban LB **y** pseudo L2.
