# docker-bake.hcl


group "build_annotations" {
  targets = [
    "annotations_base","annotations_absplice"
  ]
}

target "annotations_base" {
    dockerfile = "Dockerfile.annotations"
    tags = ["deeprvat/annotations_base:1.0"]
    platforms = ["linux/amd64"]
}

target "annotations_absplice" {
    dockerfile = "Dockerfile.absplice"
    tags = ["deeprvat/annotations_absplice:1.0"]
    platforms = ["linux/amd64"]
}

target "annotations_kipoi-veff2" {
    dockerfile = "Dockerfile.kipoi-veff2"
    tags = ["deeprvat/annotations_kipoi-veff2:1.0"]
    platforms = ["linux/amd64"]
}

target "annotations_faatpipe" {
    dockerfile = "Dockerfile.faatpipe"
    tags = ["deeprvat/annotations_faatpipe:1.0"]
    platforms = ["linux/amd64"]
}
