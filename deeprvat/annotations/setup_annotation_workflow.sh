#prerequisites: mamba, git, Per with modules Bioperl,  DBD::mysql 
VEP_CACHEDIR=$1
VEP_PLUGINDIR=$2
REPO_DIR=$3

echo "downloading necessary repos and installing conda environments"

echo "- vep"
mkdir -p $REPO_DIR/ensembl-vep
git clone https://github.com/Ensembl/ensembl-vep.git $REPO_DIR/ensembl-vep 
cd $REPO_DIR/ensembl-vep
perl INSTALL.pl --AUTO acfp --ASSEMBLY GRCh38 --CACHEDIR $VEP_CACHEDIR --PLUGINS CADD, SpliceAI, PrimateAI --PLUGINSDIR $VEP_PLUGINDIR --species homo_sapiens

echo "- AbSplice"
mkdir $REPO_DIR/absplice
git clone https://github.com/gagneurlab/absplice.git $REPO_DIR/absplice
cd $REPO_DIR/absplice
mamba env create -f environment.yaml
mamba activate absplice
pip install -e .

echo "- DeepSea(kipoi-veff2)"
mkdir $REPO_DIR/kipoi-veff2
git clone https://github.com/kipoi/kipoi-veff2.git $REPO_DIR/kipoi-veff2
cd $REPO_DIR/kipoi-veff2
mamba env create -f environment.minimal.linux.yml
mamba activate kipoi-veff2
python -m pip install .

echo "- DeepRiPe(faatpipe)"
mkdir $REPO_DIR/faatpipe
git clone https://github.com/HealthML/faatpipe.git $REPO_DIR/faatpipe

touch $REPO_DIR/annotation-workflow-setup.done