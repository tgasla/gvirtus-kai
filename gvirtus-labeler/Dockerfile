FROM python:3.13-slim
WORKDIR /app
COPY labeler.py .
RUN pip install --no-cache-dir kubernetes
CMD ["python", "labeler.py"]