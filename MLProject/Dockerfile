FROM python:3.12

WORKDIR /app

COPY . .

RUN pip install mlflow==2.19.0
RUN pip install pandas
RUN pip install numpy
RUN pip install scikit-learn

CMD ["python", "modelling.py"]