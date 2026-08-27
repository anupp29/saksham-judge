#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# keepalive.sh — cron job that pings your HF Space every 25 seconds
# so the container NEVER sleeps.
#
# SETUP (run once on any always-on machine / free server):
#   chmod +x keepalive.sh
#   crontab -e
#   # paste this line (pings every minute; script loops internally):
#   * * * * * /path/to/keepalive.sh >> /tmp/boxing-ping.log 2>&1
#
# OR run standalone in background:
#   nohup ./keepalive.sh &
# ─────────────────────────────────────────────────────────────────

SPACE_URL="https://YOUR_USERNAME-boxing-judge.hf.space/ping"
INTERVAL=25   # seconds between pings (HF sleeps after ~60s of inactivity)

echo "[$(date)] keepalive started → $SPACE_URL"

while true; do
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$SPACE_URL")
    echo "[$(date)] ping → HTTP $RESPONSE"
    sleep $INTERVAL
done