FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Allow statements and log messages to immediately appear in the Cloud Run logs
ARG TEST
ENV PYTHONUNBUFFERED True

ARG PORT
ENV PORT=$PORT

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install Python and pip
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11 python3-pip

# Install supervisord
RUN apt-get install -y supervisor spell

# Install libpq-dev for psycopg
RUN apt-get update && apt-get install -y libpq-dev

# Clean up
RUN rm -rf /var/lib/apt/lists/*

# Copy model files (assuming they are in the 'models' directory)
COPY models/ models/

# Copy setup files, requirements, and scripts
COPY setup.py requirements.txt celeryworker.sh ./

# Make celeryworker.sh executable
RUN chmod +x ./celeryworker.sh

# Install dependencies
RUN pip install --upgrade pip==23.0.1
RUN pip install -r requirements.txt

# Copy source code
COPY src/ src/
COPY pyproject.toml .

# Copy the supervisord.conf file into the container
COPY supervisord.conf /etc/supervisord.conf

RUN pip install --default-timeout=120 -e .

ENV FLASK_APP=src.seer.app

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
