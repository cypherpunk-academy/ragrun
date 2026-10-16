# Monitoring auf dem Pulse (TUXAI)

Prometheus und Grafana für den TUXAI-Server. Der Stack beobachtet den Rechner, die Container, den Embedder (`embedding-service`, Container `ragrun-embeddings`) und Qdrant (Container `qdrant`). Benachrichtigungen gibt es noch nicht: die Alarme sind nur in Prometheus und Grafana sichtbar. Ein Alertmanager ist ein offener Punkt.

## Aufbau

Eigenes Compose-Projekt `ragrun-monitoring`. Es tritt dem bestehenden Netz `ragrun-server_default` als `external` bei und fasst `deploy/tuxai/compose.yaml` nicht an.

| Dienst | Image | Port |
|---|---|---|
| prometheus | `prom/prometheus:v3.15.0` | `${TAILSCALE_IP}:9090` |
| grafana | `grafana/grafana:13.2.3` | `${TAILSCALE_IP}:3000` |
| node-exporter | `prom/node-exporter:v1.12.1` | keiner |
| cadvisor | `ghcr.io/google/cadvisor:v0.60.6` | keiner |

Die Tags sind die stabilen Stände vom 13.10.2026. cAdvisor wird nicht mehr auf Docker Hub gepflegt; das aktuelle Image liegt auf ghcr.io. Ports hängen nur an der Tailscale-Adresse, nie an `0.0.0.0`.

Prometheus behält Daten 90 Tage. Die Konfiguration, die Alarmregeln und die Datei `qdrant_key` sind read-only eingehängt. Grafana bekommt das Admin-Passwort aus `GRAFANA_ADMIN_PASSWORD`.

`qdrant_key` entsteht durch `make-qdrant-key.sh` und steht in `.gitignore`. Der Prometheus-Container läuft als Benutzer `nobody` (uid 65534). `chmod 600` des Hostbenutzers ist für diese uid nicht lesbar, und `chown` auf 65534 braucht root. Das Skript setzt deshalb **644**: der Besitzer darf lesen und schreiben, Gruppe und alle anderen dürfen nur lesen. uid 65534 fällt unter „alle anderen“ und kann die Datei im Container lesen. Jeder lokale Benutzer des Pulse kann sie ebenfalls lesen. Das ist der Verzicht, der ohne root auskommt.

## Start auf dem Pulse

Im Repo-Root, in dieser Reihenfolge:

1. `git pull`
2. `GRAFANA_ADMIN_PASSWORD` in `~/ragrun-server/.env` eintragen (oder in die Datei, auf die `TUXAI_ENV_FILE` zeigt).
3. `./deploy/tuxai/monitoring/make-qdrant-key.sh`
4. `./deploy/tuxai/monitoring/up.sh`

`up.sh` bricht ab, wenn `qdrant_key` fehlt oder `GRAFANA_ADMIN_PASSWORD` in der Env-Datei leer ist. Qdrant und der Embedder müssen schon laufen, sonst bleiben ihre Targets down.

## Adressen

- Grafana: `http://<Tailscale-IP>:3000`
- Prometheus: `http://<Tailscale-IP>:9090`

Das mitgelieferte Dashboard heißt **ragrun TUXAI**. Die Community-Dashboards **1860** (Node Exporter Full) und **14282** (cAdvisor) werden später von Hand in Grafana importiert und liegen nicht in diesem Ordner.

## Einschränkung

node-exporter hängt am Docker-Netz, nicht am Host-Netz. Die Netzwerkzahlen sind deshalb die des Containers. CPU, RAM, Platte und Temperatur sind die des Hosts (`pid: host` und `--path.rootfs=/host`).
