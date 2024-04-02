# You may need to specify the --platform depending on cloud provider
FROM --platform=linux/amd64 python:3.9

RUN python -m pip install --upgrade pip

COPY . .

ARG PRODIGY_KEY
RUN python -m pip install prodigy -f https://${PRODIGY_KEY}@download.prodi.gy
