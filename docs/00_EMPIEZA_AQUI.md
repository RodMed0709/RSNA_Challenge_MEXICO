# Empieza aquí

## Checklist del primer día

1. En RunPod crea un pod con esta imagen:

   ```text
   runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
   ```

   Elige una GPU de 48 GB o más, de preferencia `RTX PRO 6000 96GB`. Adjunta el Network Volume `RSNA-KNEE` en la región `US-NE-1` y móntalo en `/workspace`.

2. Inicia el pod y abre JupyterLab con el botón `Connect`.

3. Abre una terminal, pon tu nombre en minúsculas y clona tu propia copia:

   ```bash
   export RSNA_USER=ana
   mkdir -p /workspace/users/$RSNA_USER
   cd /workspace/users/$RSNA_USER
   git clone <URL_DEL_REPO> RSNA_Challenge
   cd RSNA_Challenge
   ```

4. Prepara el ambiente compartido:

   ```bash
   bash scripts/setup_pod.sh
   ```

   La primera vez instala los paquetes. En pods nuevos solo vuelve a registrar el kernel porque el volumen conserva `/workspace/venv`.

5. Regresa a JupyterLab, abre `notebooks/00_explorar_datos.ipynb` y elige `Kernel > Change Kernel > Python (rsna)`. Corre todas las celdas.

6. Antes de una sesión nueva vuelve a definir tu usuario:

   ```bash
   export RSNA_USER=ana
   ```

Si el notebook no ve los datos, confirma que existe `/workspace/data/raw/train.csv`. No copies el dataset a tu carpeta personal.
