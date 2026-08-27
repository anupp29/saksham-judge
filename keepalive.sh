#!/bin/bash
# Hit your Render URL every 25s to prevent free-tier sleep (sleeps after 15min)
# Run: nohup ./keepalive.sh &
# Or crontab: * * * * * /path/to/keepalive.sh >> /tmp/ping.log 2>&1

URL="https://boxing-judge.onrender.com/ping"   # ← replace with your Render URL

echo "[$(date)] keepalive started"
while true; do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$URL")
    echo "[$(date)] ping $CODE"
    sleep 25
done