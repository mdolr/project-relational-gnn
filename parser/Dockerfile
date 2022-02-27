FROM python:3.9-buster

WORKDIR /app/parser

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

# Create a data directory for the output
RUN mkdir -p /app/data 

CMD ["python","parser.py"]