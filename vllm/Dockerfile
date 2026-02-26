FROM vllm/vllm-openai:v0.15.1

WORKDIR /app

# Remove CUDA compat ldconfig registration from the base image. The compat libs
# (libcuda.so 575.x) are for forward-compat when the HOST driver is too old, but
# on hosts with newer drivers (580+) they shadow the host's libcuda and cause
# "Error 803: unsupported display driver / cuda driver combination". The NVIDIA
# container runtime injects the host driver into /lib/x86_64-linux-gnu/ which
# ldconfig picks up naturally.
RUN rm -f /etc/ld.so.conf.d/*cuda-compat*.conf && ldconfig

COPY vllm/requirements.txt .
COPY casola_worker/requirements.txt ./worker-common-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r worker-common-requirements.txt

COPY casola_worker/ ./casola_worker/
COPY vllm/worker.py .
COPY vllm/entrypoint.sh .
RUN chmod +x worker.py entrypoint.sh

ENV VLLM_HOST=127.0.0.1
ENV VLLM_PORT=8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
