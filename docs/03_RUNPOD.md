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

Desde una terminal, dentro del repo:

```bash
export RSNA_USER=ana
bash scripts/setup_pod.sh
nvidia-smi
```

`setup_pod.sh` crea `/workspace/venv` solo una vez y siempre registra el kernel `rsna` para el pod actual.

## Dejar un entrenamiento corriendo

```bash
export RSNA_USER=ana
cd /workspace/users/$RSNA_USER/RSNA_Challenge
bash scripts/run_notebook.sh notebooks/03_baseline_dinov2.ipynb ana_dinov2s_lr1e4_v1 -p FOLD 0 -p EPOCHS 5
```

Puedes cerrar el navegador: `nohup` mantiene el proceso. El comando imprime el PID y la ruta del log.

Para revisar el avance:

```bash
tail -f /workspace/users/$RSNA_USER/runs/ana_dinov2s_lr1e4_v1/train.log
nvidia-smi
```

Sal de `tail` con `Ctrl+C`; no detiene el entrenamiento.

## Costos y convivencia

Detén el pod desde la consola de RunPod apenas termine el trabajo. Detenerlo libera la GPU; el Network Volume permanece y se cobra aparte. Confirma que los resultados estén en `/workspace/users/<nombre>/runs/` antes de terminar el pod.

El volumen es compartido y los DICOM son cientos de miles de archivos. No abras ni recorras carpetas grandes mientras otras personas entrenan: puede volver lenta la lectura para todos. Usa el cache `.npy` para entrenar.
