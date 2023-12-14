
rule link_config:
    input:
        'models/repeat_0/config.yaml'
    output:
        "models/config.yaml"
    threads: 1
    shell:
        "ln -s repeat_0/config.yaml {output}"


rule best_training_run:
    input:
        expand('models/repeat_{{repeat}}/trial{trial_number}/config.yaml',
               trial_number=range(n_trials)),
    output:
        checkpoints = expand('models/repeat_{{repeat}}/best/bag_{bag}.ckpt',
                             bag=range(n_bags)),
        config = 'models/repeat_{repeat}/config.yaml'
    threads: 1
    shell:
        (
            'deeprvat_train best-training-run '
            + debug +
            'models/repeat_{wildcards.repeat} '
            'models/repeat_{wildcards.repeat}/best '
            'models/repeat_{wildcards.repeat}/hyperparameter_optimization.db '
            '{output.config}'
        )

rule train:
    input:
        config = expand('{phenotype}/deeprvat/hpopt_config.yaml',
                        phenotype=phenotypes),
        input_tensor = expand('{phenotype}/deeprvat/input_tensor.zarr',
                              phenotype=phenotypes),
        covariates = expand('{phenotype}/deeprvat/covariates.zarr',
                            phenotype=phenotypes),
        y = expand('{phenotype}/deeprvat/y.zarr',
                   phenotype=phenotypes),
    output:
        config = 'models/repeat_{repeat}/trial{trial_number}/config.yaml',
        finished = 'models/repeat_{repeat}/trial{trial_number}/finished.tmp'
    params:
        phenotypes = " ".join(
            [f"--phenotype {p} "
             f"{p}/deeprvat/input_tensor.zarr "
             f"{p}/deeprvat/covariates.zarr "
             f"{p}/deeprvat/y.zarr"
             for p in phenotypes])
    shell:
        ' && '.join([
            'deeprvat_train train '
            + debug +
            '--trial-id {wildcards.trial_number} '
            "{params.phenotypes} "
            'config.yaml '
            'models/repeat_{wildcards.repeat}/trial{wildcards.trial_number} '
            'models/repeat_{wildcards.repeat}/hyperparameter_optimization.db',
            'touch {output.finished}'
        ])
