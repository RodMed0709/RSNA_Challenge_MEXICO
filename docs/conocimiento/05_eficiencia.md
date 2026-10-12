# 05 — Eficiencia (premio de eficiencia)

Fecha: 2026-10-12. Fuentes: `RSNA_papers/05_efficiency_isbi_novelty/RESUMEN.md` (23 PDFs), A5 C2 y ★1–★2, A4 §3.

## 1. Lo que sabemos (números propios, A5 C2 / A4)

- Perfilado exp025: **I/O 41 %** del tiempo; lectura DICOM ≈ 60–70 % del runtime en Kaggle (A4 §3.3).
- Sin TTA: −50 % de tiempo (exp026). 224 px: −48 % de tiempo, −0.007 gold (exp027). Poda de slots solo en inferencia: 3 slots ≈ −0.002.
- Ensamble 5 folds: gold 0.924 vs 0.920 de un modelo.
- Ch = **pesos públicos nartaa** 224 half sin TTA: LB 0.945 en ~8 min. **No es nuestro modelo** (A5 aviso): el claim de eficiencia del paper debe salir de exp055/exp057 y del student.
- Fórmula: 0.01 AUC ≈ 697 s (A5) / 0.001 ≈ 70 s (A4).

## 2. Ideas implementables

| id | Idea | Cita | Costo | Riesgo |
|---|---|---|---|---|
| EFF-1 | **Destilar ensamble (exp045+exp054 OOF) → CoAtNet 224 / 3 slots / sin TTA / fp16**; objetivo α·teacher_OOF + (1−α)·consenso, α=0.5 fijado antes. Padre exp057 | Hinton 2015 (arXiv 1503.02531); Beyer 2022 (CVPR, doi:10.1109/CVPR52688.2022.01065: teacher *patient & consistent*, mismo crop); Noothout 2022 (JMI, doi:10.1117/1.JMI.9.5.052407); Yoon 2024 | 4–5 GPU-h | medio: teacher offline < online (Beyer); fuga leve OOF (declararla) |
| EFF-2 | **Fold soup** (= ENS-1): 1 modelo al costo de 1 | Wortsman 2022 | 0.5 | medio |
| EFF-3 | **Curva de presupuesto de cortes** (8/12/16/24 por serie) y de secuencias por hallazgo, solo inferencia | FLARE (Ma 2022/2023); Chiu 2026 (2D-VoCo); Yoon 2024 | bajo | lesiones periféricas caen; medir por hallazgo |
| EFF-4 | **Atacar I/O**: decodificación DICOM en paralelo, leer solo cortes usados | FLARE21/22: medir de punta a punta | bajo | ninguno de modelo |
| EFF-5 | Enviar **2 veces** cada candidato eficiente (varianza del runtime) | A4 §3.1 | 2 envíos | — |
| EFF-6 | INT8 TensorRT/ORT en T4 | MedPTQ (Qu 2026; hasta 2.74× vs FP32) | medio | ganancia vs fp16 menor; entorno offline. **No antes del 22-oct** |
| EFF-7 | Student ligero (RepViT, FastViT, MobileNetV3) destilado | Wang 2024; Vasu 2023; Howard 2019 | medio | latencias de móvil no transfieren a T4; si domina I/O, no importa |
| EFF-8 | Cascada / early exit | Agarwal 2023; FLARE | medio | alto: con 12 etiquetas casi todo estudio es incierto |
| EFF-9 | Recorte ROI antes de 224 (= ARC-10) | Xie 2025 | alto | ver 03 |

## 3. Para el paper

- Frontera Pareto AUC (OOF + LB privado) vs s/estudio en T4 de punta a punta, con curvas iso-score de la fórmula (A5 figura 4). **Medir runtimes oficiales antes del 22-oct** (después no se puede igual, A5 R6).
- Ningún trabajo multi-hallazgo de rodilla (CoPAS, Vuskov, Xie, OrthoDiffusion) reporta coste (A5 C2). ELNet argumenta ligereza con 1–3 etiquetas.
