FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

RUN pip install pandas pyarrow lightning scikit-learn matplotlib omegaconf gcsfs

COPY entrypoint.py .

ENV PYTHONUNBUFFERED=1

CMD ["bash"]
