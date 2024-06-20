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
    input:
    path "variants_no_id.tsv.gz"

    output:
    path "variants.parquet"
    path "duplicates.parquet"

    shell:
    '''
    deeprvat_preprocess add-variant-ids --chromosomes 21,22 "variants_no_id.tsv.gz" "variants.parquet" "duplicates.parquet"
    '''
}


process add_variant_ids {

    publishDir '/Users/b260-admin/Desktop', mode: 'copy', overwrite: false

    input:
    path "variants_no_id.tsv.gz"

    output:
    path "variants.tsv.gz"
    path "duplicates.tsv"

    shell:
    '''
    deeprvat_preprocess add-variant-ids --chromosomes 21,22  "variants_no_id.tsv.gz" "variants.tsv.gz" "duplicates.tsv"
    '''
}



workflow  {
    // Define the path to the text file containing the paths
    def pathFile = 'example/vcf_files_list.txt'
    def fasta_file = Channel.fromPath("example/workdir/reference/GRCh38.primary_assembly.genome.fa")
    def vcf_files = Channel.fromPath(pathFile).splitText().map{file(it.trim())}

    def normalized = normalize(vcf_files, 
                                extract_samples(vcf_files.first()),
                                fasta_file,index_fasta(fasta_file))
    
    def concatenated_variants = concatenate_variants(variants(normalized))

    sparsify(normalized)    
    create_parquet_variant_ids(concatenated_variants)
    add_variant_ids(concatenated_variants)
}
