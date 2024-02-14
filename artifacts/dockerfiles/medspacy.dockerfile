# medsapcy environment

FROM python:3.11.0


RUN pip install --upgrade pip && \
    pip install pandas numpy matplotlib seaborn medspacy openpyxl && \
    pip install text-generation



