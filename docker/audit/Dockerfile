FROM postgres:15

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && ln -s /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*
