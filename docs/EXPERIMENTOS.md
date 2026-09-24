# Registro de experimentos

**Referencia actual: exp003_ref_split_v1** (pendiente de correr sobre `split_v1_seed42`).

Una fila por experimento. `nuevo_experimento()` agrega la fila con estado `spec`; su autor la completa en el verdict.
Estados: `spec`, `running`, `worked`, `didnt`, `killed`. Los números salen de `results/expNNN_tag/metrics.json`.
exp001 y exp002 se corrieron con el split viejo (fold 0 generado en cada run): no se comparan directo con los demás.

| exp | tag | autor | padre | variable cambiada | estado | pseudo f0 | gold | OOF 5f | aprendizaje |
|---|---|---|---|---|---|---|---|---|---|
| exp001 | dinov2s_lr1e4 | rodrigo | — | baseline, lr 1e-4 para todo | didnt | 0.708 | 0.641 | — | Con lr alto el backbone se degrada al descongelarlo: gold 0.64 → 0.60 |
| exp002 | dinov2s_bblr1e5 | rodrigo | exp001 | backbone lr 1e-5, 8 épocas | worked | 0.828 | 0.781 | — | lr bajo en el backbone estabiliza el fine-tuning |
| exp003 | ref_split_v1 | rodrigo | exp002 | split canónico + semillas + entrenamiento en celdas | running | | | | |
