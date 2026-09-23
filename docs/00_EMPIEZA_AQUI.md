# Empieza aquí

## Checklist del primer día

0. Pídele a Rodrigo acceso al repo privado de GitHub y al equipo de RunPod.

1. En RunPod ve a `Storage`, elige el Network Volume `RSNA-KNEE` (región `US-NE-1`) y dale `Deploy`. Usa el template `RSNA-KNEE`; ya trae esta imagen, los puertos y el disco:

   ```text
   runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
   ```

   Elige una GPU de 48 GB o más, de preferencia `RTX PRO 6000 96GB`. El volumen queda montado en `/workspace`.

2. Inicia el pod y abre JupyterLab con el botón `Connect`.

3. Abre una terminal, pon tu nombre en minúsculas y clona tu propia copia:

   ```bash
   export RSNA_USER=ana
   mkdir -p /workspace/users/$RSNA_USER
   cd /workspace/users/$RSNA_USER
   git clone https://github.com/RodMed0709/RSNA_Challenge_MEXICO.git RSNA_Challenge
   cd RSNA_Challenge
   ```

4. Prepara el ambiente compartido:

   ```bash
   bash scripts/setup_pod.sh
   ```

   La primera vez instala los paquetes. En pods nuevos solo vuelve a registrar el kernel porque el volumen conserva `/workspace/venv`.

5. Regresa a JupyterLab, abre `notebooks/00_explorar_datos.ipynb` y elige `Kernel > Change Kernel > Python (rsna)`. Corre todas las celdas.

6. Los notebooks saben quién eres por la carpeta donde está tu copia (`/workspace/users/ana/...`). En la terminal sí hay que definirlo cada sesión antes de usar `scripts/run_notebook.sh`:

   ```bash
   export RSNA_USER=ana
   ```

   GitHub te pedirá usuario y un token personal (Settings > Developer settings > Personal access tokens) la primera vez que clones o hagas push.

Si el notebook no ve los datos, confirma que existe `/workspace/data/raw/train.csv`. No copies el dataset a tu carpeta personal.
