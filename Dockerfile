FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip

EXPOSE 8501 

CMD ["streamlit", "run", "n.py"]
