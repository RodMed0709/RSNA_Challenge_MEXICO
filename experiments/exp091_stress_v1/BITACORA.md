# Bitácora — exp091_stress_v1

## 2026-10-21 — claim y auditoría previa

- Claim global creado con `nuevo_experimento("stress_v1", padre="exp040")`: `exp091_stress_v1`.
- Pod `fohtxofhx4utp1` (`rsna-stress`, RTX PRO 4500 Blackwell, USD 0.72/h); `guard_skip/rsna-stress` presente.
- RunPod al inicio: saldo USD 25.2403389251; gasto global USD 0.165/h antes de crear la GPU.
- `main`/`origin/main` del clon del volumen estaban obsoletos y el fetch privado no tenía credenciales. Se transfirió el
  snapshot comprobable `origin/main` local `a139cc8` (15-oct, contiene exp040-exp073) por bundle y se creó un worktree
  aislado. No hubo push ni merge. Riesgo: el orquestador deberá rebasar antes de integrar si GitHub tiene un main posterior.
- Auditoría de pesos/OOF/metrics cerrada antes de implementar. Paneles y exclusiones quedaron fijados en `SPEC.md`.
- Fase SDD al cerrar este bloque: SPEC/PLAN; gate siguiente: commit de SPEC/PLAN, tests RED→GREEN, smoke.
