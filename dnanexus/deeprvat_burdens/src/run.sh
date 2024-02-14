main() {
    echo "Mounting via dxfuse"
    mkdir -pv /mnt/project
    dxfuse -verbose 2 /mnt/project DeepRVAT
    sleep 3
    echo "----------"

    echo "Unpacking conda env"
    echo "mkdir -p deeprvat_env"
    mkdir -p deeprvat_env
    tar -xzf /mnt/project/DeepRVAT/DeepRVAT/deeprvat_env.tar.gz -C deeprvat_env
    # ./deeprvat_non_cuda/bin/python
    source deeprvat_env/bin/activate
    conda-unpack

    echo "Installing deeprvat package"
    mkdir -p out/results
    cd out/results
    tar -xzf /mnt/project/DeepRVAT/DeepRVAT/deeprvat.tar.gz
    echo "pip install -e deeprvat"
    pip install -e deeprvat

    echo "Downloading data"
    echo "dx download DeepRVAT/workdir/preprocessed/genotypes.h5"
    cp /mnt/project/DeepRVAT/DeepRVAT/workdir/preprocessed/genotypes.h5 .
    echo "dx download DeepRVAT/data/variants_90pct10dp_qc.parquet"
    cp /mnt/project/DeepRVAT/DeepRVAT/data/variants_90pct10dp_qc.parquet .
    ln -s variants_90pct10dp_qc.parquet variants.parquet
    echo "dx download DeepRVAT/data/phenotypes.parquet"
    cp /mnt/project/DeepRVAT/DeepRVAT/data/phenotypes.parquet .
    echo "dx download DeepRVAT/data/annotations.parquet"
    cp /mnt/project/DeepRVAT/DeepRVAT/data/annotations.parquet .
    echo "dx download DeepRVAT/data/protein_coding_genes.parquet"
    cp /mnt/project/DeepRVAT/DeepRVAT/data/protein_coding_genes.parquet .

    echo "Executing command: $command using config $config"
    echo "dx download $config"
    cp /mnt/project/DeepRVAT/$config .
    echo "eval $command"
    eval $command

    echo "Uploading outputs"
    echo "rm config.yaml"
    rm -rf config.yaml genotypes.h5 variants_90pct10dp_qc.parquet phenotypes.parquet annotations.parquet protein_coding_genes.parquet deeprvat
    echo "dx-upload-all-outputs"
    dx-upload-all-outputs

    echo "DONE!"
}
