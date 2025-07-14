FROM python:3.11.0


RUN pip install --upgrade pip && \
    pip install pandas numpy==1.26.4 matplotlib seaborn medspacy openpyxl pydantic==1.10.10 pyarrow