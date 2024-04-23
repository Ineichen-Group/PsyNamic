# Base image
FROM python:3.9

# Set time zone to Europe/Zurich
ENV TZ=Europe/Zurich
RUN apt-get update && \
    apt-get install -y tzdata && \
    apt-get clean
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
echo $TZ > /etc/timezone

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Get the Prodigy key from the docker_compose.yml file
ARG PRODIGY_KEY
RUN python -m pip install prodigy -f https://${PRODIGY_KEY}@download.prodi.gy
