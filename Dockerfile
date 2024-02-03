FROM python:3.10

WORKDIR /app

COPY ./requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install wsgi

COPY ./src /app

CMD ["wsgi", "main:app", "--host", "0.0.0.0", "--port", "80"]