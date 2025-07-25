FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt 

COPY . .

# AWS LAMBDA WILL USE THIS TO COMMUNICATE
EXPOSE 8000 

#RUN UVICORN ON HOST 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
