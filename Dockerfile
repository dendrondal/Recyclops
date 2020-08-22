FROM python:3.8
WORKDIR /recyclops
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run app/app.py --server.enableCORS=false --server.port=80
