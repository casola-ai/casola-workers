FROM vllm/vllm-omni:v0.14.0

WORKDIR /app

COPY vllm/requirements.txt .
COPY casola_worker/requirements.txt ./worker-common-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r worker-common-requirements.txt

COPY casola_worker/ ./casola_worker/
COPY vllm/worker.py .
COPY vllm/entrypoint.sh .
RUN chmod +x worker.py entrypoint.sh

ENV VLLM_HOST=127.0.0.1
ENV VLLM_PORT=8000
ENV VLLM_SERVE_COMMAND=vllm-omni

# Symlink vllm_omni package into /app so relative --stage-configs-path resolves correctly
RUN VLLM_OMNI_PATH="$(python3 -c 'import vllm_omni; print(vllm_omni.__path__[0])' 2>/dev/null | tail -1)" && \
    test -n "$VLLM_OMNI_PATH" && \
    ln -s "$VLLM_OMNI_PATH" /app/vllm_omni

ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
