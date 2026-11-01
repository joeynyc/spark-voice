# Ming 3.0 ASR (tiny) for DGX Spark (GB10, sm_121). NGC PyTorch keeps the Blackwell build;
# the model's code needs Transformers 5 and fla-core (linear-attention kernels, Triton).
FROM nvcr.io/nvidia/pytorch:26.07-py3
RUN pip install --no-cache-dir transformers==5.12.1 fla-core==0.5.2 accelerate==1.15.0 einops soundfile \
      fastapi uvicorn huggingface_hub \
 && pip uninstall -y torchao
RUN pip install --no-cache-dir python-multipart
COPY shim /opt/shim
COPY ming-asr/server.py /opt/server.py
ENV PYTHONPATH=/opt/shim PORT=7797 HOST=127.0.0.1
WORKDIR /opt
# HOST is 127.0.0.1 unless compose sets BIND (Spark 2 in a two-Spark setup).
CMD uvicorn server:app --host ${HOST} --port ${PORT}
