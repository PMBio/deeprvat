main() {
    echo "Mounting via dxfuse"
    mkdir -pv /mnt/project
    dxfuse -verbose 2 /mnt/project DeepRVAT
    sleep 3
    echo "----------"

    echo "Unpacking conda env"
    mkdir -p deeprvat_non_cuda
    tar -xzf /mnt/project/DeepRVAT/DeepRVAT/deeprvat_non_cuda.tar.gz -C deeprvat_non_cuda
    ./deeprvat_non_cuda/bin/python
    source deeprvat_non_cuda/bin/activate
    conda-unpack

    echo "Installing deeprvat package"
    tar -xzf /mnt/project/DeepRVAT/DeepRVAT/deeprvat.tar.gz
    pip install -e deeprvat

    echo "Executing command: $command using config $config"
    mkdir -p out/results
    cd out/results
    cp /mnt/project/DeepRVAT/$config .
    eval $command

    echo "Uploading outputs"
    rm config.yaml
    dx-upload-all-outputs

    echo "DONE!"
}
