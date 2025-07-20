# 2012 Kindle with 256MB RAM, 0.3 CPU, Python 3.10

FROM python:3.10

RUN apt update && apt install -y git

RUN git clone --depth=1 https://github.com/akhilvreddy/emnist-on-a-potato.git

WORKDIR /emnist-on-a-potato

RUN pip install -r requirements.txt
# RUN pip install -r kindle_sim/requirements.txt

# CMD ["python", "run_kindle.py"]