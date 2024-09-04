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
        cpanminus perl git default-libmysqlclient-dev 


# Install cpan modules
RUN cpanm Archive::Zip
RUN cpanm DBD::mysql
RUN cpanm Bundle::DBI

RUN git clone https://github.com/Ensembl/ensembl-vep.git

RUN cd ensembl-vep && perl INSTALL.pl

