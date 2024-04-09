# You may need to specify the --platform depending on cloud provider
FROM --platform=linux/amd64 python:3.9

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY . .

ARG PRODIGY_KEY
RUN python -m pip install prodigy -f https://${PRODIGY_KEY}@download.prodi.gy
