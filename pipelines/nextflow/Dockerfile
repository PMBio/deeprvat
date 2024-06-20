FROM mambaorg/micromamba:1.5.8
COPY --chown=$MAMBA_USER:$MAMBA_USER deeprvat_preprocessing_env.yml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
