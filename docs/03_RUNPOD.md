# RunPod

## Pod recomendado

- Imagen: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- Network Volume: `RSNA-KNEE`
- Región: `US-NE-1`
- Montaje: `/workspace`

GPU, de mayor a menor preferencia práctica:

| GPU | VRAM | Cuándo usarla |
|---|---:|---|
| RTX PRO 6000 | 96 GB | Entrenamientos grandes y mejor margen para batch. |
| H100 | 80 GB | Muy rápida; úsala cuando el costo por hora tenga sentido. |
| L40S | 48 GB | Buena opción normal para DINOv2-small. |
| A6000 | 48 GB | Opción estable si la L40S no está disponible. |

## Preparación

Desde una terminal, dentro de tu carpeta:

```bash
cd /workspace/users/<tu_nombre>
bash scripts/setup_pod.sh
```

`setup_pod.sh` elige el ambiente según la versión de Python de la imagen (3.11 → `/workspace/venv`, 3.12 → `/workspace/venv312`),
lo crea si falta o está roto, instala `requirements.txt` y registra el kernel `Python (rsna)`. Tu usuario sale del nombre de tu carpeta.

## Dejar un entrenamiento corriendo

```bash
cd /workspace/users/<tu_nombre>
bash scripts/run_notebook.sh experiments/exp023_dinov2_base/exp023_dinov2_base.ipynb
bash scripts/run_notebook.sh experiments/exp023_dinov2_base/exp023_dinov2_base.ipynb -y "FOLDS: [0, 1, 2, 3, 4]"
```

Puedes cerrar el navegador: `nohup` mantiene el proceso. El comando imprime el PID y la ruta del log.

Para revisar el avance:

```bash
tail -f results/exp023_dinov2_base/run_*.log
nvidia-smi
```

Sal de `tail` con `Ctrl+C`; no detiene el entrenamiento.

## Costos y convivencia

Detén el pod desde la consola de RunPod apenas termine el trabajo. Detenerlo libera la GPU; el Network Volume permanece y se cobra aparte. Confirma que `results/expNNN_tag/metrics.json` exista antes de terminar el pod.

El volumen es compartido y los DICOM son cientos de miles de archivos. No abras ni recorras carpetas grandes mientras otras personas entrenan: puede volver lenta la lectura para todos. Usa el cache `.npy` para entrenar.

## Tiempos de referencia

exp003 (DINOv2-small, 224 px, 5 series × 16 cortes, 8 épocas) tarda ~90 min por fold en RTX PRO 6000:
fold 0 ≈ 1.5 h, 5 folds ≈ 7.5 h. Haz screening con `FOLDS = [0]` y solo confirma con 5 folds lo que promete.
