FROM vllm/vllm-omni:v0.16.0

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

# SM120 (RTX 5090) -> SM100 FP4 monkey-patch — see #700
COPY vllm/sm120_fp4_patch.py /tmp/_sm120_fp4_patch.py
RUN site="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    cp /tmp/_sm120_fp4_patch.py "$site/_sm120_fp4_patch.py" && \
    echo "import _sm120_fp4_patch" > "$site/sm120-fp4-patch.pth"

# Symlink vllm_omni package into /app so relative --stage-configs-path resolves correctly.
# Use site-packages path instead of `import vllm_omni` to avoid CUDA init during build.
RUN site="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    test -d "$site/vllm_omni" && \
    ln -s "$site/vllm_omni" /app/vllm_omni

# Worker code — changes most frequently, keep last for layer caching
COPY casola_worker/ ./casola_worker/
COPY --chmod=+x vllm/worker.py .
COPY --chmod=+x shared/entrypoint.sh ./shared/
COPY --chmod=+x vllm/entrypoint.sh .

ENV VLLM_HOST=127.0.0.1
ENV VLLM_PORT=8000
ENV VLLM_SERVE_COMMAND=vllm-omni

ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
