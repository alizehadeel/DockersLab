FROM python:3.10 
#base image, starting point of the container

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py . 
#first copy req, install them and then run app.py

EXPOSE 5000
#Tells Docker “this container will listen on port 5000”.

CMD python app.py