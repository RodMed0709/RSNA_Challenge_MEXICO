# exp047_raptor_slot_dropout — bitácora

- 2026-10-12 · claim creado; pod rsna-d2 (tras sub031_3). Slots del corpus por posición fija (build_meta.json), presencia 1.00/0.973/1.00/0.745/1.00.
- Simulación del muestreo train (3000 estudios, K 12): fracción de estudios con 0/1/2/3 slots presentes sin ninguna ventana — padre 0.707/0.275/0.019/0; exp047 0.338/0.460/0.184/0.018. Ventanas repetidas por bolsa: padre 0, exp047 0.58 (jitter). Slots ausentes (cortes a cero dentro del rango) se siguen muestreando como en el padre.
- 2026-10-12 05:16 UTC · run fold 0 terminado (rc 0): mejor ep14 pseudo L2 0.8844, SWA [14,15,11] 0.8849, gold 0.9139, 2.50 GPU-h → didnt.
- 2026-10-12 · eval de robustez (solo inferencia, 48 ventanas, rsna-d2, ~25 min): 3 slots exp047 −0.003 pseudo / −0.005 gold vs exp031 −0.002/−0.002; sin sagital exp047 −0.025/−0.032 vs exp031 −0.029/−0.045. Tabla en SPEC sección 4.
