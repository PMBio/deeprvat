# Annotation docker images

## Build all docker images

```shell
docker buildx bake --file docker-bake.hcl build_annotations
```

## Build the apptainer

```
    apptainer build deeprvat_annotations_environment.sif micromamba_apptainer.def
```
