
rule link_config:
    input:
        model_path / 'repeat_0/config.yaml'
    output:
        model_path / 'config.yaml'
    threads: 1
    shell:
        "ln -s repeat_0/config.yaml {output}"


rule best_training_run:
    input:
        expand(model_path / 'repeat_{{repeat}}/trial{trial_number}/config.yaml',
               trial_number=range(n_trials)),
    output:
        checkpoints = expand(model_path / 'repeat_{{repeat}}/best/bag_{bag}.ckpt',
                             bag=range(n_bags)),
        config = model_path / 'repeat_{repeat}/config.yaml'
    threads: 1
    shell:
        (
            'deeprvat_train best-training-run '
            + debug +
            '{model_path}/repeat_{wildcards.repeat} '
            '{model_path}/repeat_{wildcards.repeat}/best '
            '{model_path}/repeat_{wildcards.repeat}/hyperparameter_optimization.db '
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
             for p in training_phenotypes])
    shell:
        f"parallel --jobs {n_parallel_training_jobs} --halt now,fail=1 --results train_repeat{{{{1}}}}_trial{{{{2}}}}/ "
        'deeprvat_train train '
        + debug +
        '--trial-id {{2}} '
        "{params.phenotypes} "
        'config.yaml '
        '{model_path}/repeat_{{1}}/trial{{2}} '
        '{model_path}/repeat_{{1}}/hyperparameter_optimization.db "&&" '
        'touch {model_path}/repeat_{{1}}/trial{{2}}/finished.tmp '
        "::: " + " ".join(map(str, range(n_repeats))) + " "
        "::: " + " ".join(map(str, range(n_trials)))
