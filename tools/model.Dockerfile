FROM python:3.11-slim AS download
RUN pip install --no-cache-dir huggingface_hub
ARG MODEL_ID
ARG HF_TOKEN
RUN hf download $MODEL_ID ${HF_TOKEN:+--token "$HF_TOKEN"}

FROM scratch
COPY --from=download /root/.cache/huggingface /root/.cache/huggingface
