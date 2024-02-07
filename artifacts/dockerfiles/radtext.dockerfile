# medsapcy environment

FROM python:3.8.0

#RUN apt update && apt install build-essential default-java

RUN pip install --upgrade pip && \
    pip install pandas numpy matplotlib seaborn medspacy spacy


#    \
#    radtext && \
#    python -m spacy download en_core_web_sm && \
#    radtext-download --all



RUN pip install radtext

RUN python -m spacy download en_core_web_sm

RUN pip install nltk

RUN python -m nltk.downloader stopwords