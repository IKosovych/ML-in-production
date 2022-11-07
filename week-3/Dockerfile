FROM python:3.10
WORKDIR /news-classification
COPY requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
ADD news-classification/data /news-classification/data
ADD news-classification/scripts /news-classification/scripts
ADD news-classification/config /news-classification/config
ADD news-classification/tests /news-classification/tests
CMD ["python", "scripts/main.py"]
