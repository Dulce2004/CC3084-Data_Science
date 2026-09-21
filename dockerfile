# Fija Debian a bookworm para tener openjdk-17 disponible
FROM python:3.11-slim-bookworm

# Instalar Java 17 y utilidades
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless ca-certificates curl tini && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV PYSPARK_PYTHON=/usr/local/bin/python

WORKDIR /opt/app

# PySpark + Jupyter + Librerías de Data Science
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        pyspark==3.5.1 \
        notebook \
        jupyterlab \
        ipykernel \
        findspark \
        pandas \
        pyarrow \
        matplotlib \
        seaborn

# Usuario no root
RUN useradd -ms /bin/bash spark && chown -R spark:spark /opt/app
USER spark

EXPOSE 8888
CMD ["bash"]
