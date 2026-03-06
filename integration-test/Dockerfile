FROM python:3.11-slim

WORKDIR /app

COPY integration-test/requirements.txt .
COPY casola_worker/requirements.txt ./worker-common-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r worker-common-requirements.txt

COPY casola_worker/ ./casola_worker/
COPY --chmod=+x integration-test/worker.py .
COPY --chmod=+x integration-test/entrypoint.sh .

ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
