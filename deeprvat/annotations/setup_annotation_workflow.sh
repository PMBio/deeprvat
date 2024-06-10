#!/usr/bin/env bash

set -e
set -o pipefail

# Prerequisites: mamba, git, Perl with DBI, Bioperl, DBD::mysql modules

REPO_DIR="$1"
TO_INSTALL="$2" # valid values: ensembl-vep absplice kipoi-veff2 faatpipe vep-plugins
VEP_CACHEDIR=$REPO_DIR/ensembl-vep/cache
VEP_PLUGINDIR=$REPO_DIR/ensembl-vep/Plugins

if [ -z "$REPO_DIR" ]; then 
    echo "You need to specify the repo base path $0 <REPO_DIR> <TO_INSTALL>"
    exit 1
fi

if [ -z ""$TO_INSTALL"" ]; then 
    echo "You need to specify the tools to install $0 $REPO_DIR <TO_INSTALL>"
    echo "Example: $0 $REPO_DIR ensembl-vep,absplice,faatpipe,vep-plugins"
    echo "Valid values for TO_INSTALL: ensembl-vep absplice kipoi-veff2 faatpipe vep-plugins"
    exit 1
fi

if [ -z "$MAMBA_EXE" ]; then 
    echo "You need mamba installed"
    exit 1
fi

if ! command -v perl &> /dev/null
then
    echo "perl could not be found"
    echo "Please install togheter with DBI, Bioperl, DBD::mysql modules"
    exit 1
fi

if ! command -v git &> /dev/null
then
    echo "git could not be found"
    exit 1
fi

echo "Downloading necessary repos and installing conda environments for: $TO_INSTALL"

tool="ensembl-vep"
if [[ "$TO_INSTALL" == *$tool* ]]; then
    echo "Installing $tool"

    perl -MCPAN -e 'install Bundle::DBI'
    git clone https://github.com/Ensembl/ensembl-vep.git $REPO_DIR/ensembl-vep 
    cd $REPO_DIR/ensembl-vep
    git checkout release/111
    perl INSTALL.pl --AUTO ac --ASSEMBLY GRCh38 --CACHEDIR $VEP_CACHEDIR --species homo_sapiens
    tree $VEP_CACHEDIR
fi


tool="absplice"
if [[ "$TO_INSTALL" == *$tool* ]]; then
    echo "Installing $tool"
    mkdir -p $REPO_DIR/absplice
    git clone https://github.com/gagneurlab/absplice.git $REPO_DIR/absplice
    cd $REPO_DIR/absplice
    $MAMBA_EXE env create -f environment.yaml
    $MAMBA_EXE activate absplice
    pip install -e .
fi

tool="kipoi-veff2"
if [[ "$TO_INSTALL" == *$tool* ]]; then
    echo "Installing $tool"
    mkdir -p $REPO_DIR/kipoi-veff2
    git clone https://github.com/kipoi/kipoi-veff2.git $REPO_DIR/kipoi-veff2
    cd $REPO_DIR/kipoi-veff2
    $MAMBA_EXE env create -f environment.minimal.linux.yml
    $MAMBA_EXE activate kipoi-veff2
    python -m pip install .
fi

tool="faatpipe"
if [[ "$TO_INSTALL" == *$tool* ]]; then
    echo "Installing $tool"
    mkdir -p $REPO_DIR/faatpipe
    git clone https://github.com/HealthML/faatpipe.git $REPO_DIR/faatpipe
fi

tool="vep-plugins"
if [[ "$TO_INSTALL" == *$tool* ]]; then
    echo "Installing $tool"
    mkdir -p $VEP_PLUGINDIR
    git clone https://github.com/Ensembl/VEP_plugins.git $VEP_PLUGINDIR
fi

echo "DONE!"