main() {
    BASE=/mnt/project/DeepRVAT/DeepRVAT
    WORKDIR=workdir/pretrained_scoring # TODO: Change

    echo "Mounting via dxfuse"
    mkdir -pv /mnt/project
    dxfuse -verbose 2 /mnt/project DeepRVAT
    sleep 3
    echo "----------"

    echo "Unpacking conda env"
    echo "mkdir -p deeprvat_env"
    mkdir -p deeprvat_env
    tar -xzf $BASE/deeprvat_env.tar.gz -C deeprvat_env
    # ./deeprvat_non_cuda/bin/python
    source deeprvat_env/bin/activate
    conda-unpack

    echo "Installing deeprvat package"
    mkdir -p out/results
    cd out/results
    tar -xzf $BASE/deeprvat.tar.gz
    echo "pip install -e deeprvat"
    pip install -e deeprvat

    echo "Downloading data"
    echo "dx download DeepRVAT/workdir/preprocessed/genotypes.h5"
    cp $BASE/workdir/preprocessed/genotypes.h5 .
    # cp $BASE/data/genotypes-head1000.h5 .
    # mv genotypes-head1000.h5 genotypes.h5
    echo "dx download DeepRVAT/data/variants_90pct10dp_qc.parquet"
    cp $BASE/data/variants_90pct10dp_qc.parquet .
    echo "dx download DeepRVAT/data/phenotypes.parquet"
    cp $BASE/data/phenotypes.parquet .
    # cp $BASE/data/phenotypes-head1000.parquet .
    # mv phenotypes-head1000.parquet phenotypes.parquet
    echo "dx download DeepRVAT/data/annotations.parquet"
    cp $BASE/data/annotations.parquet .
    echo "dx download DeepRVAT/data/protein_coding_genes.parquet"
    cp $BASE/data/protein_coding_genes.parquet .

    echo "Executing command: $command using config $config"
    echo "dx download $config"
    mkdir -p Calcium/deeprvat/burdens_chunk_$chunk
    python deeprvat/deeprvat/deeprvat/associate.py compute-burdens \
           --n-chunks $n_chunks --chunk $chunk \
           --dataset-file $BASE/$WORKDIR/Calcium/deeprvat/association_dataset.pkl \
           $BASE/workdir/pretrained_scoring/Calcium/deeprvat/hpopt_config.yaml \
           $BASE/workdir/pretrained_scoring/pretrained_models/config.yaml \
           $BASE/workdir/pretrained_scoring/pretrained_models/repeat_0/best/bag_0.ckpt $BASE/workdir/pretrained_scoring/pretrained_models/repeat_1/best/bag_0.ckpt $BASE/workdir/pretrained_scoring/pretrained_models/repeat_2/best/bag_0.ckpt $BASE/workdir/pretrained_scoring/pretrained_models/repeat_3/best/bag_0.ckpt $BASE/workdir/pretrained_scoring/pretrained_models/repeat_4/best/bag_0.ckpt $BASE/workdir/pretrained_scoring/pretrained_models/repeat_5/best/bag_0.ckpt \
           Calcium/deeprvat/burdens_chunk_$chunk

    echo "Uploading outputs"
    echo "rm config.yaml"
    rm -rf deeprvat config.yaml genotypes.h5 variants_90pct10dp_qc.parquet phenotypes.parquet annotations.parquet protein_coding_genes.parquet
    echo "dx-upload-all-outputs"
    dx-upload-all-outputs

    echo "DONE!"
}
