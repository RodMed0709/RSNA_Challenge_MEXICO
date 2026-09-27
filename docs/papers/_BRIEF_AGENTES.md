# Brief común para agentes de investigación

## Contexto del problema
- Kaggle "RSNA Knee Abnormality Detection" (2026). Por estudio de resonancia de rodilla (varias series: sagital, coronal,
  axial; PD/T2 con y sin supresión grasa, T1), predecir 12 hallazgos: ACL, MCL, Medial/Lateral Meniscus,
  Medial/Lateral/PF OA, Effusion, Synovitis, Baker's cyst, Contusion (bone bruise), Fracture. Métrica: AUC media.
- Train: 4,407 estudios, ~820k DICOM; solo 58 con etiqueta de experto. El resto trae informe radiológico
  (≈12 idiomas) → pseudo-etiquetas con LLM (Qwen2.5-7B, acuerdo ~0.80 con gold).
- Test sin informes. Notebook de Kaggle sin internet, < 9 h. Hay premio/track de EFICIENCIA.
- Nuestra referencia: DINOv2-small 224px, 5 series × 16 cortes, atención cortes→series. Gold AUC ≈ 0.78.
  Notebooks públicos ~0.94 (CoAtNet-384 + DINOv2, recorte 130 mm, 2.5D, atención por etiqueta, ensembles).

## Objetivo del equipo
NO ganar el leaderboard. Ganar/ser fuertes en el **track de eficiencia** y publicar un **paper de journal** con un
modelo eficiente, con novedad y buen AUC. Prioriza: AUC por unidad de cómputo, novedad publicable, evidencia sólida.

## Qué entregas
1. Busca papers publicados (journal/conferencia peer-reviewed; arXiv solo si es muy relevante o muy citado),
   preferentemente 2021–2026. Verifica que cada paper EXISTE (URL real). Cero inventos.
2. Descarga PDFs de acceso abierto con `curl -L -A "Mozilla/5.0" -o <ruta> <url>` en tu carpeta
   `docs/papers/<tu_tema>/`, nombre `AAAA_PrimerAutor_TituloCorto.pdf`. Comprueba que empiece con `%PDF`
   (`head -c 4`), si no, bórralo. Máx 25 MB por PDF. Si es de pago, no lo descargues: solo el enlace.
   Meta: 8–15 PDFs útiles, calidad sobre cantidad.
3. Escribe `docs/papers/<tu_tema>/README.md` (español) con una tabla rankeada:
   `rank | paper (año, venue) | PDF sí/no + enlace | puntaje 1–10 | por qué nos sirve | números clave`.
   Puntaje = relevancia para nuestro problema × eficiencia × novedad × solidez de evidencia.
   Después: "Ideas para nosotros" (3–6 viñetas cortas, cada una ligada a un paper) y "Qué NO haría" (1–3 viñetas).
4. No toques nada fuera de tu carpeta. No hagas git commit.
5. Mensaje final (≤ 300 palabras): top 5 papers con una línea cada uno, y las 3 ideas más fuertes para el track de
   eficiencia con su riesgo.
