FROM debian:stable-slim
LABEL maintainer="Magnus Wahlberg <magnus.wahlberg@dkfz-heidelberg.de>"

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=C.UTF-8 \
    LANGUAGE=en_US.UTF-8

RUN apt-get update \
    && apt-get -y --no-install-recommends install  \
        build-essential \
        cpanminus perl git default-libmysqlclient-dev  \
        perl-doc zlib1g-dev libbz2-dev liblzma-dev  \
        ca-certificates wget curl dh-autoreconf && \
    rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj bin/micromamba
# curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba


RUN git clone --depth 1 --branch release/110 https://github.com/Ensembl/ensembl-vep.git 
RUN cpanm --installdeps --with-recommends --notest --cpanfile ensembl-vep/cpanfile .
RUN cd ensembl-vep && mkdir cache

RUN wget --quiet https://github.com/samtools/bcftools/releases/download/1.15.1/bcftools-1.15.1.tar.bz2 && \
    tar -xjf bcftools-1.15.1.tar.bz2 && rm bcftools-1.15.1.tar.bz2 && \
    cd bcftools-1.15.1 && \
    ./configure >/dev/null && \
    make >/dev/null && \
    make install >/dev/null && \
    cd .. && rm -r bcftools-1.15.1

RUN wget --quiet https://github.com/samtools/htslib/releases/download/1.15.1/htslib-1.15.1.tar.bz2 \
 && tar -xjf htslib-1.15.1.tar.bz2 && rm htslib-1.15.1.tar.bz2 \
 && cd htslib-1.15.1 \
 && autoreconf -i \
 && ./configure \
 && make \
 && make install \
 && cd .. && rm -r htslib-1.15.1

RUN git clone --depth 1 --branch v0.1.0 https://github.com/HealthML/faatpipe && \
    git clone --depth 1 --branch main https://github.com/kipoi/kipoi-veff2 && \
    git clone --depth 1 --branch release/110 https://github.com/Ensembl/VEP_plugins && \
    git clone --depth 1 --branch v1.0.0 https://github.com/gagneurlab/absplice.git

# TODO REMOVE :D
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj bin/micromamba

RUN cd absplice micromamba env create -f environment.yaml && micromamba activate absplice && pip install -e .

RUN cd /kipoi-veff2 && micromamba  env create -f environment.minimal.linux.yml && micromamba activate kipoi-veff2 &&  python -m pip install .

#RUN cd ensembl-vep && perl INSTALL.pl --NO_HTSLIB --NO_TEST --AUTO c --ASSEMBLY GRCh38 --CACHEDIR cache --species homo_sapiens --CACHE_VERSION 110

ENTRYPOINT ["/bin/bash"]
