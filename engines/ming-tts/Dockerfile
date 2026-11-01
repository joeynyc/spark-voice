# Ming 3.0 TTS (0.5B) for DGX Spark (GB10, sm_121). NGC PyTorch keeps the Blackwell build;
# the base image's PIP_CONSTRAINT stops pip from replacing it.
FROM nvcr.io/nvidia/pytorch:26.07-py3
# transformers 4.46 / diffusers 0.33 break on NGC's newer torchao at import; Ming doesn't use it.
RUN pip install --no-cache-dir transformers==4.46.3 accelerate==1.3.0 x_transformers==2.9.2 \
      diffusers==0.33.0 onnxruntime inflect einops soundfile loguru fastapi uvicorn \
 && pip uninstall -y torchao
# inclusionAI's inference code (MIT), pinned; the weights download from Hugging Face on first start.
ARG MING_TTS_COMMIT=028c1eace193e7acdb256453da89b6e9207e052b
RUN git clone -q https://github.com/inclusionAI/Ming-3.0-tts /opt/ming-tts \
 && git -C /opt/ming-tts checkout -q ${MING_TTS_COMMIT} && rm -rf /opt/ming-tts/.git
# torchaudio has no aarch64 CUDA wheel that matches NGC torch; Ming only needs two small pieces of it.
COPY shim /opt/shim
COPY ming-tts/server.py /opt/server.py
ENV PYTHONPATH=/opt/shim:/opt/ming-tts PORT=7798 HOST=127.0.0.1
WORKDIR /opt
# HOST is 127.0.0.1 unless compose sets BIND (Spark 2 in a two-Spark setup).
CMD uvicorn server:app --host ${HOST} --port ${PORT}
