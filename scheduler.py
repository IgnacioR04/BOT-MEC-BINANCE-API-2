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
import signal
import logging
from datetime import datetime, timezone

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("scheduler")


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
    logger.info("Iniciando MEC Bot scheduler en Railway (Hardened Mode)")
    logger.info(f"Intervalo: {INTERVAL_SECONDS}s")

    # Signal handling for graceful shutdown
    stop_event = False
    def handle_exit(signum, frame):
        nonlocal stop_event
        logger.info(f"Signal {signum} received. Shutting down gracefully...")
        stop_event = True

    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)

    while not stop_event:
        start = time.time()
        run_bot()
        elapsed = time.time() - start
        wait = max(0, INTERVAL_SECONDS - elapsed)
        
        if wait > 0:
            logger.info(f"Proxima ejecucion en {wait:.1f}s")
            # Sleep in small chunks to remain responsive to signals
            for _ in range(int(wait)):
                if stop_event: break
                time.sleep(1)
            time.sleep(wait % 1)
    
    logger.info("Scheduler stopped.")

if __name__ == "__main__":
    main()
