# exp051_raptor_norm — bitácora

- 2026-10-12 · SPEC y notebook preparados; claim creado.
- Verificación: `build_corpus_norm.py` con `PCT=(2, 98)` reproduce **bit a bit** las filas 0, 1000 y 3000 de `/root/corpus96_384`. Con 0.5–99.5: media de cortes válidos 114→91, 103→84, 82→69; fracción saturada a 255: 1.0–1.6 % → 0.2–0.4 %.
- Costo: ~1 s de pared por estudio en un proceso (8 hilos de lectura) → ~1.2 h secuencial; en paralelo (24 procesos, nice/ionice) ~21 min de pared en rsna-e2. Por debajo del umbral de 3 h → se hace. Corpus en `/root/corpus96_384_p05` (rsna-e2).
- Encadenado tras exp049 en `/root/t3_chain.sh` de rsna-e2.
- Corpus construido: 1178 s de pared (24 procesos), 0 errores, cortes válidos medios 92.34 (= padre). Smoke OK (03:27).
- Fin: mejor época 14, pseudo L2 0.8842 (Δ +0.0005), gold 0.9146, 2.47 GPU-h → `didnt`. Corpus en rsna-e2 `/root/corpus96_384_p05`.
