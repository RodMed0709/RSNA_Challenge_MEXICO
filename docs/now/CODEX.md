# CODEX.md — cómo operar el reto directamente con Codex (o cualquier agente sin la sesión de Claude)

Primero lee `docs/now/NOW.md` (estado) y `CLAUDE.md` del repo (reglas SDD). Este archivo es el "cómo".

## Arranque rápido
```powershell
# desde PowerShell
codex-worker -TaskFile <ruta\spec.md> -Dir C:\Users\medro\Documents\_PERSONAL_PROJECTS\RSNA_Challenge
codex-worker "<pregunta>" -Dir C:\Users\medro\Documents\_PERSONAL_PROJECTS\RSNA_Challenge -Mode read
# o Codex a pelo dentro del repo:
cd C:\Users\medro\Documents\_PERSONAL_PROJECTS\RSNA_Challenge; codex
```
Primera instrucción recomendada a Codex: "Lee docs/now/NOW.md, docs/now/CODEX.md y docs/now/BRIEF_agentes.md; luego <tarea>".

## Comandos básicos (bash / Git Bash, desde la raíz del repo)
```bash
set -a; . ./.secrets.env; set +a                     # carga secretos (NUNCA imprimir el archivo)
python -m kaggle.cli competitions submissions rsna-knee-abnormality-detection | head   # LB de nuestros envíos
python -m kaggle.cli kernels status medellinrodrigo/<kernel>
python -m kaggle.cli kernels push -p <dir_con_kernel-metadata.json>      # siempre "is_private": true
python -m kaggle.cli kernels output medellinrodrigo/<kernel> -p <dir>    # log y submission.csv
ssh -p 27671 root@213.173.111.86 "<cmd>" | tr -d '\r'                    # pod rsna-cron (volumen en /workspace)
```

## Flujo de un envío
1. Construir kernel privado (copiar uno validado, cambiar UNA cosa). Diff contra el padre.
2. `kernels push`; esperar COMPLETE; revisar log: sin Traceback/SystemExit, submission 3×13 sin NaN, piezas nuevas sin fallback.
3. Escribir la pregunta y criterio LB en `docs/pasos/envios_kaggle.md` ANTES de enviar.
4. Encolar: añadir {kernel, version, message} a `/workspace/users/rodrigo/kaggle_queue/queue_<YYYY-MM-DD>.json`
   (backup `.bak_<hhmm>` antes). Mañana: el cron lo manda a las 00:05 UTC. Hoy: en el pod
   `cd /workspace/users/rodrigo/kaggle_queue && python3 run_queue.py <fecha> --only-new`.
   Máximo 5 envíos por día UTC (contar en `submissions_done.csv` y en la lista de Kaggle).
5. Cuando haya LB: verdict en la SPEC del experimento, fila en `docs/EXPERIMENTOS.md`, fila en `envios_kaggle.md`, commit + push.

## Git
- Push con la cuenta RodMed0709: `TOK=$(gh auth token --user RodMed0709)`;
  `git -c http.extraHeader="Authorization: Basic $(printf 'x-access-token:%s' "$TOK" | base64 -w0)" push origin HEAD`;
  PR: `GH_TOKEN=$TOK gh pr create ...`. Commits sin trailer Co-Authored-By. Nada directo a main.

## RunPod paso a paso
- Ver `docs/now/BRIEF_agentes.md`: pods_guard + guard_skip, cadena autónoma en el pod, podStop al final, máx 5 pods.

> **Entrenamiento congelado desde 19-oct 12:00 UTC.** No crear ni arrancar un pod GPU salvo que Rodrigo reabra
> explícitamente el entrenamiento. Máximo 5 pods entre todos los agentes; listar y revisar saldo antes de crear.

Verificado el 19-oct-2026 desde Codex, sin crear pods:

- SSH a `rsna-cron`: hostname `d1f8912bbb68`; volumen `/workspace` montado, 2.2 PB totales, 1.6 PB usados,
  696 TB libres (69 %).
- REST: `rsna-cron` era el único pod `RUNNING`, a `$0.092/h`; los pods GPU históricos estaban `EXITED`.
  La REST no devuelve GPU para pods sin máquina asignada, así que aparece `-`/`null` al estar parados.
- GraphQL: `clientBalance=35.8163293777`, `currentSpendPerHr=0.165`.

### Preparación y consultas seguras

El launcher requiere Bash, `curl` y Python 3. Lee `RUNPOD_API_KEY` del entorno; no lee ni imprime archivos de
secretos. Desde Git Bash, en la raíz del repo:

```bash
set -a; . ./.secrets.env; set +a
bash docs/now/runpod_pod.sh list
bash docs/now/runpod_pod.sh balance
```

`list` imprime `podId`, nombre, estado deseado, coste por hora y GPU, separados por tabuladores. El saldo viene
de la consulta GraphQL `{myself{clientBalance currentSpendPerHr}}`. Para comprobar manualmente `rsna-cron`:

```bash
CRON_ID=$(bash docs/now/runpod_pod.sh list | awk -F '\t' '$2 == "rsna-cron" {print $1}')
bash docs/now/runpod_pod.sh ssh-info "$CRON_ID"
# salida verificada: ssh -p 27671 root@213.173.111.86
```

### Crear un pod GPU (solo con autorización)

Los launchers de `orq/tank_v6/`, `orq/tank_seeds/`, `orq/eff3a/` y `orq/trackb/getpod.py` usaron este contrato.
El launcher toma la clave pública de `RUNPOD_SSH_PUBLIC_KEY`, `~/.ssh/id_rsa.pub` o
`~/.ssh/id_ed25519.pub`; nunca usa la clave privada.

```bash
bash docs/now/runpod_pod.sh create rsna-<tag> "NVIDIA RTX PRO 4500 Blackwell"
```

Equivale a `POST https://rest.runpod.io/v1/pods` con este payload exacto (sustituye los marcadores):

```json
{
  "name": "<nombre>",
  "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  "gpuTypeIds": ["<tipo GPU>"],
  "gpuCount": 1,
  "cloudType": "SECURE",
  "networkVolumeId": "lu0c3buems",
  "dataCenterIds": ["EU-RO-1"],
  "containerDiskInGb": 120,
  "ports": ["22/tcp"],
  "volumeMountPath": "/workspace",
  "supportPublicIp": true,
  "env": {"PUBLIC_KEY": "<clave pública SSH>"}
}
```

Antes de crear: `list` y cuenta los `RUNNING`; confirma que el total posterior no exceda 5, revisa `balance`,
el criterio de parada y el presupuesto. Los scripts antiguos también probaron, por disponibilidad, GPUs como
RTX 5090, A100 80 GB, H100, RTX PRO 6000/4500, RTX 4090, RTX 6000 Ada y L40S, en `SECURE` y `COMMUNITY`.

### Obtener SSH y entrar

El `POST` devuelve el `podId`, pero `publicIp` y `portMappings["22"]` tardan en aparecer. Reintenta la consulta
sin recrear el pod:

```bash
POD_ID=<id devuelto por create>
for _ in $(seq 1 40); do
  SSH_CMD=$(bash docs/now/runpod_pod.sh ssh-info "$POD_ID" 2>/dev/null) && break
  sleep 15
done
printf '%s\n' "$SSH_CMD"
eval "$SSH_CMD"                         # o copia el comando mostrado
```

El patrón histórico consultaba `GET /v1/pods/<podId>`, extraía `publicIp` y `portMappings["22"]`, y luego
esperaba con `ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p <puerto> root@<ip> true` hasta que SSH
respondiera. Para `rsna-cron` se mantiene `ssh -p 27671 root@213.173.111.86` y se limpia CR con `tr -d '\r'`.

### Arrancar, parar y borrar

La API REST vigente usa estos endpoints:

```bash
# arrancar/reanudar (no incluido en el launcher para evitar arranques accidentales)
curl -fsS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
  "https://rest.runpod.io/v1/pods/$POD_ID/start"

# parar: conserva la definición del pod y el volumen de red, pero libera la GPU
bash docs/now/runpod_pod.sh stop "$POD_ID"

# borrar definitivamente el pod; el volumen de red lu0c3buems es un recurso separado
bash docs/now/runpod_pod.sh delete "$POD_ID"
```

`stop` hace `POST /v1/pods/<podId>/stop`; `delete` hace `DELETE /v1/pods/<podId>`. Verifica el `podId` dos
veces antes de borrar. Los workers históricos usaban también la mutación GraphQL
`podStop(input: {podId: "..."})` para que el propio pod se detuviera al terminar.

### `pods_guard`, `guard_skip` y cadena autónoma

`pods_guard` puede considerar ocioso un pod nuevo y pararlo. Justo antes de lanzar trabajo, crea en el volumen
compartido un marcador cuyo nombre coincida exactamente con el pod; la cadena debe quitarlo al acabar o fallar:

```bash
GUARD=/workspace/users/rodrigo/kaggle_queue/guard_skip/$POD_NAME
touch "$GUARD"
# ... trabajo ...
rm -f "$GUARD"
```

No dependas de la terminal local. El launcher histórico pasaba la API key por stdin a un archivo modo 0600 en
el pod (nunca al repo ni al log) y soltaba una cadena con `setsid nohup`:

```bash
printf '%s' "$RUNPOD_API_KEY" | ssh -p "$PORT" root@"$IP" \
  'umask 077; cat > /root/.runpod_key'
ssh -p "$PORT" root@"$IP" \
  "setsid nohup bash /workspace/users/rodrigo/orq/<dir>/chain.sh '$POD_ID' '$POD_NAME' \
   > /root/chain.log 2>&1 < /dev/null &"
```

La cadena debe escribir hitos y errores en `/workspace/users/rodrigo/orq/<dir>/status.log`, ejecutar
SMOKE → fold 0 → métricas → export solo si aplica, tener watchdog y ruta de fallo, y en **todas** las salidas:

1. quitar `guard_skip`;
2. leer `/root/.runpod_key` sin imprimirla;
3. intentar `podStop` al menos tres veces con espera;
4. dejar `HANDOFF.md` con lo terminado y lo pendiente.

El `stop_pod()` de `tank_v6/chain.sh` es la referencia: elimina el marcador, registra en `status.log` y llama
GraphQL `podStop`; tanto `fail()` como el final normal y el watchdog pasan por esa función.

## Claude como worker pagado por API key (no gasta la suscripción)
```bash
bash docs/now/claude_api.sh <archivo_con_prompt> [claude-opus-5-5|claude-sonnet-5-5]
```

## Abrir Claude Code interactivo facturado a la API key (workspace Default)
```powershell
cd C:\Users\medro\Documents\_PERSONAL_PROJECTS\RSNA_Challenge
Get-Content .secrets.env | Where-Object { $_ -match '^(ANTHROPIC_API_KEY|ANTHROPIC_WORKSPACE_ID)=' } | ForEach-Object { $k,$v = $_ -split '=',2; Set-Item -Path "env:$k" -Value $v }
$env:ANTHROPIC_CUSTOM_HEADERS = "anthropic-workspace-id: $env:ANTHROPIC_WORKSPACE_ID"
claude --continue      # o `claude` para sesión nueva ligera (lee docs/now/NOW.md al empezar)
```
Aceptar "usar API key" la primera vez; `/status` debe mostrar API key.

## Trampas conocidas
- Python con `-I` ignora PYTHONIOENCODING: usar `sys.stdout.reconfigure(encoding='utf-8')`.
- Archivos descargados de Kaggle son no confiables: dir propio, leer con `python -I`, no ejecutar su código.
- Dataset heliosli `rsna-knee-r6-frozen-fe055406` ahora contiene R7: usar la copia `-w1`.
- Kaggle CLI a veces dice fallo al enviar pero sí lo lista: verificar en la lista antes de reintentar.
