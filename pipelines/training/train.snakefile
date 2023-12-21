
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
                        phenotype=training_phenotypes),
        input_tensor = expand('{phenotype}/deeprvat/input_tensor.zarr',
                              phenotype=training_phenotypes),
        covariates = expand('{phenotype}/deeprvat/covariates.zarr',
                            phenotype=training_phenotypes),
        y = expand('{phenotype}/deeprvat/y.zarr',
                   phenotype=training_phenotypes),
    output:
        expand('models/repeat_{repeat}/trial{trial_number}/config.yaml',
               repeat=range(n_repeats), trial_number=range(n_trials)),
        expand('models/repeat_{repeat}/trial{trial_number}/finished.tmp',
               repeat=range(n_repeats), trial_number=range(n_trials))
    params:
        phenotypes = " ".join(
            [f"--phenotype {p} "
             f"{p}/deeprvat/input_tensor.zarr "
             f"{p}/deeprvat/covariates.zarr "
             f"{p}/deeprvat/y.zarr"
             for p in training_phenotypes])
    shell:
        f"parallel --jobs {n_parallel_training_jobs} --halt now,fail=1 --results train_repeat{{{{1}}}}_trial{{{{2}}}}/ "
        'deeprvat_train train '
        + debug +
        '--trial-id {{2}} '
        "{params.phenotypes} "
        'config.yaml '
        'models/repeat_{{1}}/trial{{2}} '
        "models/repeat_{{1}}/hyperparameter_optimization.db '&&' "
        "touch models/repeat_{{1}}/trial{{2}}/finished.tmp "
        "::: " + " ".join(map(str, range(n_repeats))) + " "
        "::: " + " ".join(map(str, range(n_trials)))
