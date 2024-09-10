from pathlib import Path
from deeprvat.deeprvat.config import create_main_config

create_main_config("deeprvat_input_config.yaml")

configfile: 'deeprvat_config.yaml'

debug_flag = config.get('debug', False)
phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes
training_phenotypes = config["training"].get("phenotypes", phenotypes)
training_phenotypes = list(training_phenotypes.keys()) if type(training_phenotypes) == dict else training_phenotypes

n_burden_chunks = config.get('n_burden_chunks', 1) if not debug_flag else 2

n_regression_chunks = 1
n_bags = config['training']['n_bags'] if not debug_flag else 3
n_repeats = config['n_repeats']
debug = '--debug ' if debug_flag else ''
do_scoretest = '--do-scoretest ' if config.get('do_scoretest', False) else ''


DEEPRVAT_ANALYSIS_DIR=os.environ['DEEPRVAT_ANALYSIS_DIR']
DEEPRVAT_DIR=os.environ['DEEPRVAT_DIR']

py_deeprvat_analysis= f'python {DEEPRVAT_ANALYSIS_DIR}'
py_deeprvat = f'python {DEEPRVAT_DIR}/deeprvat/deeprvat'

wildcard_constraints:
    repeat="\d+",
    trial="\d+",

cv_exp = True if os.path.exists('cv_split0/') else False
config_file_prefix = 'cv_split0/deeprvat/' if cv_exp else '' #needed in case we analyse a CV experiment
print(config_file_prefix)
cv_splits = 5 #TODO make this more robust for non-cv experiment 


burden_agg_fct = 'mean'
n_avg_repeats = 6
use_seed = 'wo_seed'
combi = 0


phenotypes = training_phenotypes


## specific for common variants
phecode_dict = {'Apolipoprotein_A': 30630,
 'Apolipoprotein_B': 30640,
 'Calcium': 30680,
 'Cholesterol': 30690,
 'HDL_cholesterol': 30760,
 'IGF_1': 30770,
 'LDL_direct': 30780,
 'Lymphocyte_percentage': 30180,
 'Mean_corpuscular_volume': 30040,
 'Mean_platelet_thrombocyte_volume_(MPTV)': 30100,
 'Mean_reticulocyte_volume': 30260,
 'Neutrophill_count': 30140,
 'Platelet_count': 30080,
 'Platelet_crit': 30090,
 'Platelet_distribution_width': 30110,
 'Red_blood_cell_(erythrocyte)_count': 30010,
 'SHBG': 30830,
 'Standing_height': 50,
 'Total_bilirubin': 30840,
 'Triglycerides': 30870,
 'Urate': 30880,
 'Body_mass_index_BMI': 21001,
 'Glucose': 30740,
 'Vitamin_D': 30890,
 'Albumin': 30600,
 'Total_protein': 30860,
 'Cystatin_C': 30720,
 'Gamma_glutamyltransferase': 30730,
 'Alkaline_phosphatase': 30610,
 'Creatinine': 30700,
 'Whole_body_fat_free_mass': 23101,
 'Forced_expiratory_volume_in_1_second_FEV1': 20153,
 'Glycated_haemoglobin_HbA1c': 30750,
 'Mean_platelet_thrombocyte_volume': 30100,
 'Red_blood_cell_erythrocyte_count': 30010}



gtf_file = 'gencode.v34lift37.annotation.gtf.gz'
genotype_base_dir = 'genotypes/'
padding = 500

burden_phenotype = phenotypes[0]

print('missing phenotypes')
print(set(phenotypes) - set(phecode_dict.keys()))
phenotypes = set(phenotypes).intersection(set(phecode_dict.keys()))
print(f'number of kept phenotypes: {len(phenotypes)}')
print(phenotypes)



rule all_regression_correct_common:
    input:
        expand(f'{{phenotype}}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/combi_{combi}/burden_associations_common_variant_corrected.parquet',
            phenotype = phenotypes
        )



rule regression_correct_common:
    input:
        data_config = f"{config_file_prefix}{{phenotype}}/deeprvat/config.yaml",
        chunks = lambda wildcards: (
            [] if wildcards.phenotype == phenotypes[0]
            else expand('{{phenotype}}/deeprvat/burdens/chunk{chunk}.linked',
                        chunk=range(n_burden_chunks))
        ) if not cv_exp  else '{phenotype}/deeprvat/burdens/merging.finished',
        genes_to_keep = '{phenotype}/deeprvat/burdens/significant_genes_restest.npy',
        common_variants = '{phenotype}/deeprvat/burdens/prepare_genotypes_per_gene.finished'
    output:
        '{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/combi_{combi}/burden_associations_common_variant_corrected.parquet',
    threads: 2
    resources:
        mem_mb = lambda wildcards, attempt: 28676  + (attempt - 1) * 4098,
    params:
        burden_file = f'{burden_phenotype}/deeprvat/burdens/burdens_{{burden_agg_fct}}_{{n_avg_repeats}}_{{combi}}.zarr',
        burden_dir = '{phenotype}/deeprvat/burdens/',
        out_dir = '{phenotype}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/combi_{combi}',
        common_genotype_prefix = '{phenotype}/deeprvat/burdens/genotypes_gene'
    shell:
        'deeprvat_associate regress-common '
        + debug +
        '--chunk 0 '
        '--n-chunks  1 ' 
        '--use-bias '
        '--repeat 0 '
        '--burden-file {params.burden_file} '
        '--common-genotype-prefix {params.common_genotype_prefix} '
        '--genes-to-keep {input.genes_to_keep} '
        + do_scoretest +
        '{input.data_config} '
        '{params.burden_dir} ' 
        '{output}'

rule all_data:
    input:
        expand('{phenotype}/deeprvat/burdens/prepare_genotypes_per_gene.finished',
            phenotype = phenotypes
        )

rule prepare_genotypes_per_gene:
    conda:
        "prs" #TODO upgrade deeprvat environment pyarrow to version 6.0.1. to make DeepRVAT env work 
    input:
        significant_genes = '{phenotype}/deeprvat/eval/significant_genes_restest.parquet',
        data_config = 'deeprvat_config.yaml', 
        genotype_file = lambda wildcards: f'{genotype_base_dir}/GWAS_variants_clumped_mac_{phecode_dict[wildcards.phenotype]}.parquet',
        sample_file = '{phenotype}/deeprvat/burdens/sample_ids.finished'
    params: 
        out_dir = '{phenotype}/deeprvat/burdens/',
        sample_file = '{phenotype}/deeprvat/burdens/sample_ids.zarr'
    output:
        '{phenotype}/deeprvat/burdens/prepare_genotypes_per_gene.finished'
    threads: 16
    resources:
        mem_mb = 60000
        # mem_mb = lambda wildcards, attempt: 60000  + (attempt - 1) * 4098,
    shell:
        ' && '.join([
            (f'{py_deeprvat}/common_variant_condition_utils.py prepare-genotypes-per-gene '
            '--gtf-file '+ str(gtf_file) + ' '
            '--padding '+ str(padding) + ' '
            '--standardize '
            '{input.data_config} '
            '{input.significant_genes} '
            '{input.genotype_file} '
            '{params.sample_file} '
            '{params.out_dir} '),
            'touch {output}'
        ])


rule get_significant_genes:
    input:
        res_file = f"{{phenotype}}/deeprvat/{burden_agg_fct}_agg_results/{n_avg_repeats}_repeats/eval/{use_seed}/all_results.parquet",
        data_config = 'deeprvat_config.yaml' 
    output:
        out_parquet = '{phenotype}/deeprvat/eval/significant_genes_restest.parquet',
        out_npy = '{phenotype}/deeprvat/burdens/significant_genes_restest.npy'
    threads: 2
    resources:
        mem_mb = lambda wildcards, attempt: 8000  + (attempt - 1) * 4098,
    shell:
        py_deeprvat + '/common_variant_condition_utils.py get-significant-genes ' 
        '--pval-correction-method Bonferroni '
        # f'{debug_flag} '
        '{input.data_config} '
        '{input.res_file} '
        '{output.out_parquet} '
        '{output.out_npy} '

# this will be redundant in future since it is in newer versions of associate.pya

rule get_ordered_sample_ids:
    input: 
        dataset_pickle = expand('cv_split{split}/deeprvat/{{phenotype}}/deeprvat/association_dataset.pkl', split = range(cv_splits))
    output:
        '{phenotype}/deeprvat/burdens/sample_ids.finished'
    params:
        dataset_files = lambda wildcards, input: ''.join([
            f'--dataset-files {f} '
            for f in input.dataset_pickle
        ]),
        out_file = '{phenotype}/deeprvat/burdens/sample_ids.zarr'
    threads: 8
    resources:
        mem_mb = lambda wildcards, attempt: 32000  + (attempt - 1) * 4098,
    shell:
        ' && '.join([ 
            (py_deeprvat + '/get_ordered_sample_ids.py get-ordered-sample-ids ' 
            '{params.dataset_files} '
            '{params.out_file} '),
            'touch {output}'
        ])

