FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY SVM_Classifier.py .
COPY iris.csv .
CMD ["python", "SVM_Classifier.py"]