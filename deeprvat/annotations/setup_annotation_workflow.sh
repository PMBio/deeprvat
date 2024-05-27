#prerequisites: mamba, git, Per with DBI, Bioperl, DBD::mysql modules
VEP_CACHEDIR=$1
VEP_PLUGINDIR=$2
REPO_DIR=$3
CLEANUP=$4

echo "downloading necessary repos and installing conda environments"
perl -MCPAN -e 'install Bundle::DBI'
echo "- vep"
mkdir -p $REPO_DIR/ensembl-vep
git clone https://github.com/Ensembl/ensembl-vep.git $REPO_DIR/ensembl-vep 
cd $REPO_DIR/ensembl-vep
git checkout release/111
perl INSTALL.pl --AUTO acfp --ASSEMBLY GRCh38 --CACHEDIR $VEP_CACHEDIR --PLUGINS CADD, SpliceAI, PrimateAI --PLUGINSDIR $VEP_PLUGINDIR --species homo_sapiens
cd ../..

echo "- DeepSea(kipoi-veff2)"
mkdir -p $REPO_DIR/kipoi-veff2
git clone https://github.com/kipoi/kipoi-veff2.git $REPO_DIR/kipoi-veff2
cd $REPO_DIR/kipoi-veff2
$MAMBA_EXE env create -f environment.minimal.linux.yml
$MAMBA_EXE activate kipoi-veff2
python -m pip install .
cd ../..

echo "- DeepRiPe(faatpipe)"
mkdir -p $REPO_DIR/faatpipe
git clone https://github.com/HealthML/faatpipe.git $REPO_DIR/faatpipe

echo vep plugins
mkdir -p $REPO_DIR/VEP_plugins
git clone https://github.com/Ensembl/VEP_plugins.git $REPO_DIR/VEP_plugins
##returning to main environment

$MAMBA_EXE activate deeprvat_annotations

# Cleanup mamba
if [ -n "$CLEANUP" ]; then
  $MAMBA_EXE clean --all --yes
fi

##create token output file
touch $REPO_DIR/annotation-workflow-setup.done

