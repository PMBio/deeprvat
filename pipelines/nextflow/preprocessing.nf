params.qc = false
process extract_samples {

    input:
    path vcf_file

    shell:
    '''
    bcftools query --list-samples !{vcf_file} > samples_chr.csv
    '''

    output:
    path 'samples_chr.csv'

}

process index_fasta {

    input:
    path fasta_file

    output:
    path "${fasta_file}.fai"

    shell:
    '''
    samtools faidx !{fasta_file}
    '''
}
process normalize {
    input:
    path vcf_file
    path sample_file
    path fasta_file
    path fasta_index_file

    output:
    path "${vcf_file}.bcf"

    shell:
    '''
    bcftools view --samples-file !{sample_file} --output-type u !{vcf_file} \
    | bcftools view --include 'COUNT(GT="alt") > 0' --output-type u \
    | bcftools norm -m-both -f !{fasta_file} --output-type b --output !{vcf_file}.bcf
    '''

}

process variants {

    input:
    path bcf_file

    output:
    path "${bcf_file}.tar.gz"

    shell:
    '''
    bcftools query --format '%CHROM\t%POS\t%REF\t%ALT\n' !{bcf_file} | gzip > !{bcf_file}.tar.gz
    '''
}

process sparsify {

    input:
    path bcf_file

    output:
    path "${bcf_file}_sparse.tsv.gz"

    shell:
    '''
    bcftools query --format '[%CHROM\t%POS\t%REF\t%ALT\t%SAMPLE\t%GT\n]' --include 'GT!="RR" & GT!="mis"' !{bcf_file} \
    | sed 's/0[/,|]1/1/; s/1[/,|]0/1/; s/1[/,|]1/2/; s/0[/,|]0/0/' | gzip > !{bcf_file}_sparse.tsv.gz
    '''

}

process concatenate_variants {
    input:
    path 'variant_files'

    output: 
    path "variants_no_id.tsv.gz"

    shell:
    '''
    zcat variant_files* | gzip > variants_no_id.tsv.gz
    '''
}

process create_parquet_variant_ids {

    publishDir './example/workdir', mode: 'copy', overwrite: true

    input:
    path "variants_no_id.tsv.gz"

    output:
    path "variants.parquet", emit: variants
    path "duplicates.parquet", emit: duplicates

    shell:
    '''
    deeprvat_preprocess add-variant-ids --chromosomes 21,22 "variants_no_id.tsv.gz" "variants.parquet" "duplicates.parquet"
    '''
}


process add_variant_ids {

    publishDir './example/workdir', mode: 'copy', overwrite: true

    input:
    path "variants_no_id.tsv.gz"

    output:
    path "variants.tsv.gz", emit: variants
    path "duplicates.tsv", emit: duplicates 

    shell:
    '''
    deeprvat_preprocess add-variant-ids --chromosomes 21,22  "variants_no_id.tsv.gz" "variants.tsv.gz" "duplicates.tsv"
    '''
}

process preprocess {
    
    

    input:
    path "variants.parquet"
    path "duplicates.parquet"
    path "samples_chr.csv"
    path "sparse/sparse.tsv.gz"

    output:
    path "genotypes_chr*.h5"
    shell:
    '''
   
    deeprvat_preprocess process-sparse-gt variants.parquet samples_chr.csv sparse genotypes
    
    '''

}

process combine_genotypes{

    publishDir './example/workdir', mode: 'copy', overwrite: true
    
    input: 
    path 'genotype_files'

    output:
    path 'genotypes.h5'
    shell:
    '''
    deeprvat_preprocess combine-genotypes  genotype_files* genotypes.h5
    '''
}



process qc_read_depth{

    input:
    path 'infile.bcf'
    output:
    path 'read_depth.tsv.gz'
    shell:
    '''
    bcftools query --format '[%CHROM\\t%POS\\t%REF\\t%ALT\\t%SAMPLE\\n]' --include '(GT!="RR" & GT!="mis" & TYPE="snp" & FORMAT/DP < 7) | (GT!="RR" & GT!="mis" & TYPE="indel" & FORMAT/DP < 10)' infile.bcf | gzip > read_depth.tsv.gz
    '''

}

process qc_varmiss {
    input:
    path "infile.bcf"
    output: 
    path "varmiss.tsv.gz"
    shell:
    '''
     bcftools query --format "%CHROM\t%POS\t%REF\t%ALT\n" --include "F_MISSING >= 0.1" infile.bcf | gzip > varmiss.tsv.gz
    '''
}

process qc_allelic_imbalance {
    input:
    path "infile.bcf"
    output: 
    path "allelic_imbalance.tsv.gz"
    shell:
    '''
     bcftools query --format "%CHROM\t%POS\t%REF\t%ALT\n" --include "F_MISSING >= 0.1" infile.bcf | gzip > allelic_imbalance.tsv.gz
    '''
}

process qc_hwe {
    input:
    path "infile.bcf"
    output:
    path "hwe.tsv.gz"
    shell:
    '''
    bcftools +fill-tags --output-type u infile.bcf --tags HWE | bcftools query --format "%CHROM\t%POS\t%REF\t%ALT\n" --include "INFO/HWE <= 1e-15" | gzip > hwe.tsv.gz
    '''

}

process qc_indmiss {
    
    publishDir './example/workdir/qc', mode: 'copy', overwrite: true
    
    input:
    path "infile.bcf"
    output: 
    path "indmiss_stats.stats", emit: stats
    path "indmiss_samples.tsv", emit: samples
    path "indmiss_sites.tsv", emit: sites

    shell:
    '''
    bcftools +smpl-stats --output indmiss_stats.stats infile.bcf && grep "^FLT0" indmiss_stats.stats > indmiss_samples.tsv && grep "^SITE0" indmiss_stats.stats > indmiss_sites.tsv
    '''
}

process process_individual_missingness {
    debug true
    input:
    path "ind_miss/sites/indmiss?.tsv"
    path "ind_miss/samples/indmiss?.tsv"
    
    output:
    path "indmiss_samples.csv"
    shell:
    '''
    ls ind_miss/samples > files.txt
    deeprvat_preprocess process-individual-missingness files.txt ind_miss indmiss_samples.csv
    '''

}


process preprocess_qc {
    input:
    path "variants.parquet"
    path "duplicates.parquet"
    path "variants.tsv.gz"
    path "qc/duplicates/duplicates.tsv"
    path "samples_chr.csv"
    path "sparse/sparse.tsv.gz"
    path "qc/read_depth/exclude?.tsv.gz"
    path "qc/varmiss/exclude?.tsv.gz"
    path "qc/allelic_imbalance/exclude?.tsv.gz"
    path "qc/hwe/exclude?.tsv.gz"
    path "qc/indmiss/samples.csv"
    output:
    path "genotypes_chr*.h5"
    shell:
    '''
    deeprvat_preprocess process-sparse-gt --exclude-calls qc/read_depth --exclude-samples qc/indmiss --exclude-variants qc/duplicates --exclude-variants qc/hwe --exclude-variants qc/varmiss --exclude-variants qc/allelic_imbalance variants.parquet samples_chr.csv sparse genotypes
    '''

}


workflow  {
    // Define the path to the text file containing the paths
    def pathFile = 'example/vcf_files_list.txt'
    def fasta_file = Channel.fromPath("example/workdir/reference/GRCh38.primary_assembly.genome.fa")
    def vcf_files = Channel.fromPath(pathFile).splitText().map{file(it.trim())}
    def pathFilePath = Channel.fromPath(pathFile).first()
    def samples = extract_samples(vcf_files.first())
    def normalized = normalize(vcf_files, 
                                samples,
                                fasta_file.first(), index_fasta(fasta_file).first())
    
    def sparsified = sparsify(normalized)
    def concatenated_variants = concatenate_variants(variants(normalized).collect())
    def variants = create_parquet_variant_ids(concatenated_variants)
    def variants_tsv = add_variant_ids(concatenated_variants)
    if ( params.qc ) {
        def read_depth_exclude_files = qc_read_depth(normalized).collect()
        def varmiss_exclude_files = qc_varmiss(normalized).collect()
        def allelic_imbalance_exclude_files = qc_allelic_imbalance(normalized).collect()
        def hwe_exclude_files = qc_hwe(normalized).collect()
        qc_indmiss(normalized)
        def indmiss_sites = qc_indmiss.out.sites.collect()
        def indmiss_samples = qc_indmiss.out.samples.collect()
        def indmiss_stats = qc_indmiss.out.stats.collect()
        def indmiss_files = process_individual_missingness(indmiss_sites,  indmiss_samples)
        preprocessed = preprocess_qc(variants, variants_tsv ,samples, sparsified, read_depth_exclude_files, varmiss_exclude_files, allelic_imbalance_exclude_files, hwe_exclude_files, indmiss_files)
    }
    else {
        preprocessed = preprocess(variants,samples, sparsified )

        
    }
    
    combine_genotypes(preprocessed.collect())
    
}
