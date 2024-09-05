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
        ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*


RUN git clone --depth 1 --branch release/110 https://github.com/Ensembl/ensembl-vep.git 
RUN cpanm --installdeps --with-recommends --notest --cpanfile ensembl-vep/cpanfile .
RUN cd ensembl-vep && mkdir cache

#RUN cd ensembl-vep && perl INSTALL.pl --NO_HTSLIB --NO_TEST --AUTO c --ASSEMBLY GRCh38 --CACHEDIR cache --species homo_sapiens --CACHE_VERSION 110

ENTRYPOINT ["/bin/bash"]
