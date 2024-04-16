#!/usr/bin/env bash

SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"


generate_rule_graph() {
  SNAKEFILE="$1"
  DIRECTORY="$2"
  CONFIG="$3"
  OUTPUT_FILE="$4"

  echo "Generating rule graph: $OUTPUT_FILE"


  snakemake -n --snakefile "$SNAKEFILE" --directory "$DIRECTORY" --configfile "$CONFIG" --forceall --rulegraph \
    | tail -n +2  | awk '!/color=/{gsub(/color = "[^"]+",/, "");} {gsub(/,$/, "");} 1' | awk '{$1=$1}1' \
    | dot -Tsvg > "$OUTPUT_FILE"
}


generate_dag_graph() {
  SNAKEFILE="$1"
  DIRECTORY="$2"
  CONFIG="$3"
  OUTPUT_FILE="$4"
  echo "Generating dag graph: $OUTPUT_FILE"
  snakemake -n --snakefile "$SNAKEFILE" --directory "$DIRECTORY" --configfile "$CONFIG" --forceall --dag \
    | tail -n +2 | dot -Tsvg > "$OUTPUT_FILE"
}




PIPELINE_DIR="$SCRIPT_PATH/../pipelines"
CONFIG_DIR="$PIPELINE_DIR/config"
WORK_DIR="$SCRIPT_PATH/../example"
STATIC_DIR="$SCRIPT_PATH/_static"

generate_rule_graph "$PIPELINE_DIR/preprocess_with_qc.snakefile" "$WORK_DIR/preprocess" "$CONFIG_DIR/deeprvat_preprocess_config.yaml" "$STATIC_DIR/preprocess_rulegraph_with_qc.svg"
generate_rule_graph "$PIPELINE_DIR/preprocess_no_qc.snakefile" "$WORK_DIR/preprocess" "$CONFIG_DIR/deeprvat_preprocess_config.yaml" "$STATIC_DIR/preprocess_rulegraph_no_qc.svg"

generate_rule_graph "$PIPELINE_DIR/annotations.snakefile"  "$WORK_DIR/annotations" "$CONFIG_DIR/deeprvat_annotation_config.yaml" "$STATIC_DIR/annotation_rulegraph.svg"
generate_dag_graph "$PIPELINE_DIR/annotations.snakefile"  "$WORK_DIR/annotations" "$CONFIG_DIR/deeprvat_annotation_config.yaml" "$STATIC_DIR/annotation_pipeline_dag.svg"

