from pathlib import Path

genome = absplice_main_conf['genome']
p = Path(vcf_dir)
vcf_ids = [f.name for f in p.glob('*_variants_header.vcf.gz')] 


def splicemap5(wildcards):
    path = Path(absplice_download_dir) / config_download['splicemap_dir']
    splicemaps = [
        path / f'{tissue}_splicemap_psi5_method=kn_event_filter=median_cutoff.csv.gz' 
        for tissue in absplice_main_conf['splicemap_tissues']
        ]
    splicemaps = [str(x) for x in splicemaps]
    return splicemaps

def splicemap3(wildcards):
    path = Path(absplice_download_dir) / config_download['splicemap_dir']
    splicemaps = [
        path / f'{tissue}_splicemap_psi3_method=kn_event_filter=median_cutoff.csv.gz' 
        for tissue in absplice_main_conf['splicemap_tissues']
        ]
    splicemaps = [str(x) for x in splicemaps]
    return splicemaps


rule mmsplice_splicemap:
    input:
        vcf = vcf_id,
        fasta = Path(absplice_download_dir) / config_download['fasta'][genome]['file'],
        splicemap_5 = splicemap5,
        splicemap_3 = splicemap3
    resources:
        mem_mb = 30_000,
        threads = 4
    conda: 
        "absplice"
    output:
        result = Path(absplice_output_dir)/ config_pred['splicing_pred']['mmsplice_splicemap']
    script:
        "./mmsplice_splicemap.py"




if absplice_main_conf['use_rocksdb'] == True:
    genome_mapper = {
        'hg38': 'grch38',
        'hg19': 'grch37',
    }

    def dict_path(wildcards):
        paths = {}
        genome = wildcards['genome']
        for chr in config_download['chromosomes']:
            paths[chr] = str(Path(absplice_download_dir) / config_download['spliceai_rocksdb'][genome].format(chromosome=chr))
        return paths

    rule spliceai:
        resources:
            mem_mb = lambda wildcards, attempt: attempt * 16000,
            threads = 1,
            gpu = 1,
        input:
            vcf = vcf_id,
            fasta = str(Path(absplice_download_dir) / config_download['fasta'][genome]['file']),
            spliceai_rocksdb = expand(Path(absplice_download_dir) / config_download['spliceai_rocksdb'][genome],
                                    chromosome=config_download['chromosomes'])
        params:
            db_path = dict_path,
            lookup_only = False,
            genome = genome_mapper[absplice_main_conf['genome']]
        conda:
            f"./environment_spliceai_rocksdb.yaml"
        output:
            result = Path(absplice_output_dir)/config_pred['splicing_pred']['spliceai']
        script:
            "./spliceai.py"
else:
    genome_mapper = {
        'hg38': 'grch38',
        'hg19': 'grch37',
    }
    rule spliceai:
        resources:
            mem_mb = lambda wildcards, attempt: attempt * 16000,
            threads = 1,
            gpu = 1,
        input:
            vcf = vcf_id,
            fasta = Path(absplice_download_dir) /config_download['fasta'][genome]['file']
        params:
            genome = genome_mapper[absplice_main_conf['genome']]
        conda:
            f"./environment_spliceai_rocksdb.yaml"
        output:
            result = config_pred['splicing_pred']['spliceai_vcf']
        shell:
            'spliceai -I {input.vcf} -O {output.result} -R {input.fasta} -A {params.genome}'
    
    rule spliceai_vcf_to_csv:
        input:
            spliceai_vcf = Path(absplice_output_dir) /  config_pred['splicing_pred']['spliceai_vcf'],
        output:
            spliceai_csv = Path(absplice_output_dir) / config_pred['splicing_pred']['spliceai'],
        conda:
            "absplice"
        run:
            from absplice.utils import read_spliceai_vcf
            df = read_spliceai_vcf(input.spliceai_vcf)
            df.to_csv(output.spliceai_csv, index=False)

rule absplice_dna:
    resources: 
        mem_mb = lambda wildcards, attempt: attempt * 16_000
    input:
        mmsplice_splicemap = Path(absplice_output_dir) / config_pred['splicing_pred']['mmsplice_splicemap'],
        spliceai = Path(absplice_output_dir) / config_pred['splicing_pred']['spliceai'],
    params:
        extra_info = absplice_main_conf['extra_info_dna']
    conda:
        "absplice"
    output:
        absplice_dna = absplice_output_dir / '{genome}' / 'dna' / '{vcf_id}_AbSplice_DNA.csv'
    script:
        "./absplice_dna.py"

rule all_predict_dna:
    input:
        #expand([absplice_output_dir / absplice_main_conf['genome'] / 'dna' / source_variant_file_pattern],zip, chr=chromosomes, block=block)
        expand([absplice_output_dir / absplice_main_conf['genome'] / 'dna' / '{file_stem}_AbSplice_DNA.csv'],file_stem=file_stems)

del splicemap5
del splicemap3
