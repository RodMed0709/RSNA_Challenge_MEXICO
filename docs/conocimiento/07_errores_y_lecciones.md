# 07 — Errores y lecciones (para no repetirlos)

Fecha: 2026-10-12. Fuentes: A1 (hallazgos clave), A2 (§1, §2), A4, A6. Cada punto dice qué pasó, por qué importa y la regla que sale.

## 1. "SWA" sin recalcular BN (A2 §2.12)
- **Qué:** promediamos los 3 mejores `state_dict` (ép. 13–15) con LR de OneCycle ≈ 0 y **copiamos los buffers BN promediados sin `update_bn`**. CoAtNet tiene BN en las etapas MBConv.
- **Efecto:** Δ SWA − mejor época entre −0.0004 y +0.0005 en todos; solo exp033 (24 ép, épocas separadas) dio +0.0020. Es un seguro de selección de época, no SWA de Izmailov.
- **Regla:** tras promediar pesos, `torch.optim.swa_utils.update_bn(loader, model)` con ~500 estudios de train; y promediar puntos **separados** (LR cíclica/constante). Re-chequear los pesos all-data ya enviados (ENS-3).

## 2. exp052 diseñado nulo por construcción (A2 §2.9)
- **Qué:** muestrear una fuente por estudio y época. La BCE es lineal en y → mismo gradiente esperado que la media, más varianza.
- **Regla:** antes de correr, preguntar "¿cambia el gradiente esperado?". Si no, no es un experimento. Alternativas con fundamento: cabezas por fuente, DS, pérdida no lineal en y (01_etiquetas_llm LAB-2/3/8).

## 3. Mixup entre planos (A2 §2.5)
- **Qué:** se mezclaba la ventana k de A con la ventana k de B; las ventanas se barajan entre 5 slots → **sagital × axial**. Imagen sin sentido; etiqueta mezclada no corresponde a la evidencia que la atención encuentra.
- **Regla:** cualquier mezcla entre estudios debe estar **alineada por slot y posición**, o hacerse en el espacio de rasgos pooled.

## 4. Afín fuerte con p=1 y relleno con ceros (A2 §2.2)
- **Qué:** sub045_1 subió magnitud **y** mantuvo p=1, aplicado después del recorte 320 con `padding_mode="zeros"` → marco negro de hasta 20 %.
- **Regla:** magnitudes fuertes van con p 0.2–0.5 (AnyMC3D, Lumbar 1.º); aplicar el afín sobre el corte completo (384) y recortar después, o `padding_mode="border"`. Lo probado es "afín fuerte con p=1, ceros y 16 épocas no sirve", no "afín fuerte no sirve".
- Lo mismo para intensidad (exp046): **gamma/contraste p=1** + mismos parámetros para las 5 secuencias.

## 5. Padre exp031 seed 42 salió bajo (A2 §1.3)
- **Qué:** seed 42 = 0.8836, seed 7 = 0.8852. Contra la media (0.8844), los "didnt" exp046–052, sub031_4, sub046_1 quedan entre −0.0008 y +0.0012: **9 de 10 deltas positivos eran regresión a la media**. exp045 sí es real (+0.0055 sobre la media).
- **Regla:** para efectos esperados < +0.003, padre con **2 semillas** (o comparar contra la media de semillas). El ruido de semilla 0.0013 sale de solo 2 semillas: no es una SD fiable.

## 6. Regularizadores probados sin sobreajuste (A2 §1.1)
- Mejor época = última o penúltima de 16 → drop_path, mixup, intensidad, slot dropout pagaron el retraso de convergencia sin cobrar nada. exp033 muestra sobreajuste después de la ép. 14 a 24 épocas.
- **Regla:** probar regularizadores sobre un **padre largo** (20–24 ép) — un experimento para el padre largo, otro para el regularizador.

## 7. Todos contra exp031 en paralelo (A2 §1.2)
- exp046–052 nunca se midieron sobre exp045 (que ya regulariza). **Regla:** las re-pruebas van contra el padre vigente (exp045 o el combo).

## 8. Dos variables en un experimento (exp047; exp056)
- exp047 = slot dropout + jitter ±1 de centros. exp056-alldata-s42s7 mezcla combo y semillas (A6). **Regla 2 del repo**; si se hace por coste, marcar como "informativo, no causal".

## 9. Nombre engañoso: "res384" (exp050, A2 §2.7)
- No subió la resolución (mm/píxel fijo): **amplió el campo de visión**. Conclusión correcta: "el borde que quita el recorte no aporta". **Regla:** nombrar por lo que cambia físicamente.

## 10. Registro roto (A1 hallazgo 1 y 3)
- `main:docs/EXPERIMENTOS.md` solo tiene exp001–exp010 y exp023; ~50 filas viven en ramas sin mergear; exp038/039/042/055 solo en `tablero_paso7.md`.
- Inconsistencias: sub031_2 con pseudo y gold invertidos (0.918/0.8852 vs tablero 0.8852/0.9178); exp012, exp015 `running`; exp010, exp011 en `spec`; exp044 en `run` sin verdict; padre de exp056 distinto en fila y tablero; numeración de `paso6_plan_lb.md` ≠ lo ejecutado.
- **Regla:** mergear el registro a `main` antes de citar números en el paper; verdict obligatorio (regla 9) aunque falle.

## 11. Interno vs LB (A4)
- Offsets estables (LB ≈ gold + 0.02; ≈ pseudo + 0.06), pero Δ < 0.003 en público = empate. exp030 vs exp029 (−0.003) era ruido + circularidad (pseudo L2 premia entrenar con L2).
- Peso 0.35 del bloque propio bajó el LB (0.952, 0.950): con ρ 0.90–0.95 era previsible (A4 §4).
- Pseudo v1 vs LB ρ −0.40: **la definición de etiqueta importa más que el modelo**.
- **Regla:** fijar pesos y umbrales antes de ver el LB; nunca barrer; el LB decide entre familias solo con Δ ≥ 0.003.

## 12. Evaluación que no mide el objetivo (A2 §1.4)
- Las augs de robustez se evaluaron solo in-distribution; el harness exp040 (ruido, sitio croata) existía y no se usó.
- **Regla:** toda técnica de robustez reporta además el harness exp040 en la sección 3 del SPEC.

## 13. Ninguna SPEC cita literatura (A1 hallazgo 2)
- Todo origen es Kaggle (K1–K11) o interno. **Regla:** la sección 1 de cada SPEC nueva lleva cita (clave de `referencias.bib`) o "inferencia".

## 14. Pesos públicos con OAI (A6 §4; A1)
- nartaa (blend-mine, velci, exp042/044) usa supervisión OAI. Riesgo compartido con ~400 cuentas. Para la final eficiente, preferir modelo propio si empata.
