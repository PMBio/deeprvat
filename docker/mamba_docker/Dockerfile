FROM mambaorg/micromamba:1.5.9-bookworm-slim

COPY --chown=$MAMBA_USER:$MAMBA_USER deeprvat_annotations.yml /tmp/deeprvat_annotations.yml
RUN micromamba create -y -f /tmp/deeprvat_annotations.yml && \
    micromamba clean --all --yes
