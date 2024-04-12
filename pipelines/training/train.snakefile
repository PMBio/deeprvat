rule link_config:
    input:
        model_path / 'repeat_0/config.yaml'
    output:
        model_path / 'config.yaml'
    threads: 1
    shell:
        "ln -rfs {input} {output}"
        # "ln -s repeat_0/config.yaml {output}"

rule best_training_run:
    input:
        expand(model_path / 'repeat_{{repeat}}/trial{trial_number}/config.yaml',
               trial_number=range(n_trials)),
    output:
        checkpoints = expand(model_path / 'repeat_{{repeat}}/best/bag_{bag}.ckpt',
                             bag=range(n_bags)),
        config = model_path / 'repeat_{repeat}/config.yaml'
    params:
        prefix = '.'
    threads: 1
    resources:
        mem_mb = 2048,
        load = 2000
    shell:
        (
            'deeprvat_train best-training-run '
            + debug +
            '{params.prefix}/{model_path}/repeat_{wildcards.repeat} '
            '{params.prefix}/{model_path}/repeat_{wildcards.repeat}/best '
            '{params.prefix}/{model_path}/repeat_{wildcards.repeat}/hyperparameter_optimization.db '
            '{output.config}'
        )

rule train:
    input:
        config = expand('{phenotype}/deeprvat/hpopt_config.yaml',
                        phenotype=training_phenotypes),
        input_tensor = expand('{phenotype}/deeprvat/input_tensor.zarr',
                              phenotype=training_phenotypes),
        covariates = expand('{phenotype}/deeprvat/covariates.zarr',
                            phenotype=training_phenotypes),
        y = expand('{phenotype}/deeprvat/y.zarr',
                   phenotype=training_phenotypes),
    output:
        expand(model_path / 'repeat_{repeat}/trial{trial_number}/config.yaml',
               repeat=range(n_repeats), trial_number=range(n_trials)),
        expand(model_path / 'repeat_{repeat}/trial{trial_number}/finished.tmp',
               repeat=range(n_repeats), trial_number=range(n_trials))
    params:
        phenotypes = " ".join(
            [f"--phenotype {p} "
             f"{p}/deeprvat/input_tensor.zarr "
             f"{p}/deeprvat/covariates.zarr "
             f"{p}/deeprvat/y.zarr"
             for p in training_phenotypes]),
        prefix = '.',
    priority: 1000
    resources:
        mem_mb = 2000000,        # Using this value will tell our modified lsf.profile not to set a memory resource
        load = 8000,
        gpus = 1
    shell:
        f"parallel --jobs {n_parallel_training_jobs} --halt now,fail=1 --results train_repeat{{{{1}}}}_trial{{{{2}}}}/ "
        'deeprvat_train train '
        + debug +
        '--trial-id {{2}} '
        "{params.phenotypes} "
        'config.yaml '
        '{params.prefix}/{model_path}/repeat_{{1}}/trial{{2}} '
        "{params.prefix}/{model_path}/repeat_{{1}}/hyperparameter_optimization.db '&&' "
        "touch {params.prefix}/{model_path}/repeat_{{1}}/trial{{2}}/finished.tmp "
        "::: " + " ".join(map(str, range(n_repeats))) + " "
        "::: " + " ".join(map(str, range(n_trials)))
