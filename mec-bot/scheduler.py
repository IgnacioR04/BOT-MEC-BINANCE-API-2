"""
scheduler.py
============
Loop infinito que ejecuta bot.py cada 60 segundos.
Se usa en Railway como alternativa a GitHub Actions para
conseguir latencia <5s en lugar de los ~30-60s del cron nativo.

Railway mantiene este proceso corriendo 24/7.
El bot se ejecuta, espera 60s, y vuelve a ejecutar.
"""

import subprocess
import sys
import time
from datetime import datetime, timezone


INTERVAL_SECONDS = 60


def run_bot():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[scheduler] {ts} UTC — lanzando bot.py", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "bot.py"],
            timeout=300,
        )
        if result.returncode != 0:
            print(f"[scheduler] bot.py termino con codigo {result.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        print("[scheduler] TIMEOUT: bot.py tardo mas de 5 minutos", flush=True)
    except Exception as e:
        print(f"[scheduler] ERROR: {e}", flush=True)


def main():
    print("[scheduler] Iniciando MEC Bot scheduler en Railway", flush=True)
    print(f"[scheduler] Intervalo: {INTERVAL_SECONDS}s", flush=True)

    while True:
        start = time.time()
        run_bot()
        elapsed = time.time() - start
        wait    = max(0, INTERVAL_SECONDS - elapsed)
        if wait > 0:
            print(f"[scheduler] Proxima ejecucion en {wait:.1f}s", flush=True)
            time.sleep(wait)


if __name__ == "__main__":
    main()
