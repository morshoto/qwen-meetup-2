FROM python:3.12-alpine@sha256:e6dae589c1c05976188fd3bd019afd091445fc533ed2ea29001beddc7368a9c8
COPY runner.py /opt/runner.py
USER 65534:65534
ENTRYPOINT ["python", "-I", "-B", "/opt/runner.py"]
