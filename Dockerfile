FROM python:3.11

# Allow statements and log messages to immediately appear in the Cloud Run logs
ARG TEST
ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy model files (assuming they are in the 'models' directory)
COPY models/ models/

# Copy setup files and requirements
COPY setup.py requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip==23.0.1
RUN pip install -r requirements.txt

# Copy source code
COPY src/ src/
COPY pyproject.toml .

RUN pip install --default-timeout=120 -e .
RUN mypy

# The number of gunicorn workers is selected by ops based on k8s configuration.
CMD exec gunicorn --bind :$PORT --worker-class sync --threads 1 --timeout 0 src.seer.app:run
