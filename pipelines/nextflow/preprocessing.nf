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
    publishDir './example/workdir/qc', mode: 'copy', overwrite: true
    input:
    path 'infile.bcf'
    output:
    path 'read_depth.tsv.gz'
    shell:
    '''
    bcftools query --format '[%CHROM\\t%POS\\t%REF\\t%ALT\\t%SAMPLE\\n]' --include '(GT!="RR" & GT!="mis" & TYPE="snp" & FORMAT/DP < 7) | (GT!="RR" & GT!="mis" & TYPE="indel" & FORMAT/DP < 10)' infile.bcf | gzip > read_depth.tsv.gz
    '''

}

process preprocess_qc {
    
    input:
    path "variants.parquet"
    path "duplicates.parquet"
    path "samples_chr.csv"
    path "sparse/sparse.tsv.gz"
    path "qc/read_depth/exclude?.tsv.gz"

    output:
    path "genotypes_chr*.h5"
    shell:
    '''
    ls -lah qc/read_depth/
    deeprvat_preprocess process-sparse-gt --exclude-variants qc/read_depth variants.parquet samples_chr.csv sparse genotypes
    
    '''

}


workflow  {
    // Define the path to the text file containing the paths
    def pathFile = 'example/vcf_files_list.txt'
    def fasta_file = Channel.fromPath("example/workdir/reference/GRCh38.primary_assembly.genome.fa")
    def vcf_files = Channel.fromPath(pathFile).splitText().map{file(it.trim())}
    
    def samples = extract_samples(vcf_files.first())
    def normalized = normalize(vcf_files, 
                                samples,
                                fasta_file.first(), index_fasta(fasta_file).first())
    
    def sparsified = sparsify(normalized)
    def concatenated_variants = concatenate_variants(variants(normalized).collect())
    def variants = create_parquet_variant_ids(concatenated_variants)

    if ( params.qc ) {
        def read_depth_exclude_files = qc_read_depth(normalized).collect()
        preprocessed = preprocess_qc(variants,samples, sparsified, read_depth_exclude_files)
    }
    else {
        preprocessed = preprocess(variants,samples, sparsified )

        
    }
    add_variant_ids(concatenated_variants)
    combine_genotypes(preprocessed.collect())
    
}
