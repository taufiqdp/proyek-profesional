FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements-docker.txt --no-cache-dir

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]