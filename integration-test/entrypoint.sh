#!/bin/bash

set -e

WORKER_PID=""
SHUTDOWN_INITIATED=0

cleanup() {
    if [ $SHUTDOWN_INITIATED -eq 1 ]; then
        return
    fi
    SHUTDOWN_INITIATED=1

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Cleanup initiated"

    if [ -n "$WORKER_PID" ] && kill -0 $WORKER_PID 2>/dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Sending SIGTERM to worker (PID: $WORKER_PID)"
        kill -TERM $WORKER_PID 2>/dev/null || true

        GRACE_PERIOD=${CASOLA_SHUTDOWN_GRACE_PERIOD:-5}
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Waiting up to ${GRACE_PERIOD}s for worker to shutdown gracefully"

        for i in $(seq 1 $GRACE_PERIOD); do
            if ! kill -0 $WORKER_PID 2>/dev/null; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') - Worker shutdown gracefully"
                break
            fi
            sleep 1
        done

        if kill -0 $WORKER_PID 2>/dev/null; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Force killing worker"
            kill -9 $WORKER_PID 2>/dev/null || true
        fi
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Cleanup complete"
}

trap cleanup SIGTERM SIGINT EXIT

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting worker"
python -u /app/worker.py &
WORKER_PID=$!
echo "$(date '+%Y-%m-%d %H:%M:%S') - Worker started (PID: $WORKER_PID)"

# Wait for worker process to exit
wait $WORKER_PID
WORKER_EXIT_CODE=$?
echo "$(date '+%Y-%m-%d %H:%M:%S') - Worker exited with code: $WORKER_EXIT_CODE"
exit $WORKER_EXIT_CODE
