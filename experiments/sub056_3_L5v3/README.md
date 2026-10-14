# L5v3 (sub056_3) — etiquetas LLM, 3.ª ronda
1. Qué: L5v3 = 5 lecturas de L5v2 (caché A–E) + F_focus3 (gpt-5.4: Contusion/Synovitis/Baker's) + G_quote (gpt-5.4-mini: 12 hallazgos con cita), media por celda.
2. Cuándo: solo si sub056_2 = worked y existe la marca `GO` aquí (la crea Rodrigo o el orquestador cuando él confirme la recarga de $30).
3. Dónde: desde rsna-cron (clave en /root/.openai_key, chmod 600; nunca en /workspace). Usa /workspace/venv312 (sin paquete openai: HTTP directo).
4. Ejecutar: `bash run_l5v3.sh --pod <PUERTO> <HOST>` (pod de GPU encendido con /root/corpus96_384; máximo 2 pods encendidos).
5. Pasos: piloto de 20 (proyecta el costo, para si > $29) → todo (caché y reanudación en cache/) → build → validación → entreno.
6. Salidas: /workspace/data/labels/ours/L5v3_<fecha>.csv (+ copia L5v3.csv y lecturas sueltas), L5v3_validacion.json, STATUS, logs/.
7. Marcas: L5V3_NO_QUOTA (cuota insuficiente: recargar y relanzar, reanuda), L5V3_OVER_BUDGET, L5V3_UNHEALTHY (no entrena), L5V3_READY, L5V3_POD_BUSY.
8. Si el pod estaba ocupado: `bash run_l5v3.sh --launch-only --pod <PUERTO> <HOST>`.
9. Entreno: train_sub056_3.sh en el pod (smoke → fold 0) en wt_sub056_3_L5v3; estado en results/sub056_3_L5v3/chain_status.
10. Criterio (SPEC de la rama sub056_3_L5v3): worked si pseudo L2 fold 0 ≥ sub056_2 + 0.003 y gold ≥ gold sub056_2 − 0.02. Prueba: `bash run_l5v3.sh --dry-run --pod ...`.
