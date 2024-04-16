#!/usr/bin/env bash

SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

PIPELINE_DIR="$SCRIPT_PATH/../pipelines"
CONFIG_DIR="$PIPELINE_DIR/config"
WORK_DIR="$SCRIPT_PATH/../example"
STATIC_DIR="$SCRIPT_PATH/_static"

if ! command -v "snakemake" &> /dev/null; then
    echo "Please install snakemake (did you forget to activate the conda env?)"
    exit 1
fi

check_result_file(){
  file_path="$1"
  if [ -e "$file_path" ]; then
      if [ -s "$file_path" ]; then
          echo "Ok graph exists :)"
      else
          # Remove the file
          rm "$file_path"
          echo "Fail: Graph was empty, removed."
      fi
  fi
}

generate_rule_graph() {
  SNAKEFILE="$PIPELINE_DIR/$1"
  DIRECTORY="$2"
  CONFIG="$3"
  OUTPUT_FILE="$STATIC_DIR/$(basename "$SNAKEFILE" |cut -f 1 -d'.')_rulegraph.svg"

  echo "Generating rule graph: $OUTPUT_FILE"

  snakemake -n --snakefile "$SNAKEFILE" --directory "$DIRECTORY" --configfile "$CONFIG" --forceall --rulegraph \
    | sed -n '/digraph/,$p' | awk '!/color=/{gsub(/color = "[^"]+",/, "");} {gsub(/,$/, "");} 1' | awk '{$1=$1}1' \
    | dot -Tsvg > "$OUTPUT_FILE"
  check_result_file "$OUTPUT_FILE"
  echo -e "---------\n"
}


generate_dag_graph() {
  SNAKEFILE="$PIPELINE_DIR/$1"
  DIRECTORY="$2"
  CONFIG="$3"
  OUTPUT_FILE="$STATIC_DIR/$(basename "$SNAKEFILE" |cut -f 1 -d'.')_dag.svg"

  echo "Generating dag graph: $OUTPUT_FILE"
  snakemake -n --snakefile "$SNAKEFILE" --directory "$DIRECTORY" --configfile "$CONFIG" --forceall --dag \
    | sed -n '/digraph/,$p' | dot -Tsvg > "$OUTPUT_FILE"
  check_result_file "$OUTPUT_FILE"
}


generate_rule_graph "preprocess_with_qc.snakefile" "$WORK_DIR/preprocess" "$CONFIG_DIR/deeprvat_preprocess_config.yaml"
generate_rule_graph "preprocess_no_qc.snakefile" "$WORK_DIR/preprocess" "$CONFIG_DIR/deeprvat_preprocess_config.yaml"

generate_rule_graph "annotations.snakefile"  "$WORK_DIR/annotations" "$CONFIG_DIR/deeprvat_annotation_config.yaml"
generate_dag_graph "annotations.snakefile"  "$WORK_DIR/annotations" "$CONFIG_DIR/deeprvat_annotation_config.yaml"

generate_rule_graph "association_testing_pretrained.snakefile" "$WORK_DIR" "$WORK_DIR/config.yaml"

generate_rule_graph "association_testing_pretrained_regenie.snakefile" "$WORK_DIR" "$WORK_DIR/config.yaml"

generate_rule_graph "run_training.snakefile" "$WORK_DIR" "$WORK_DIR/config.yaml"

generate_rule_graph "seed_gene_discovery.snakefile" "$WORK_DIR" "$WORK_DIR/config.yaml"

generate_rule_graph "training_association_testing.snakefile" "$WORK_DIR" "$WORK_DIR/config.yaml"

generate_rule_graph "training_association_testing_regenie.snakefile" "$WORK_DIR" "$WORK_DIR/config.yaml"


echo "Done!"