rule link_config:
    input:
        model_path / 'repeat_0/model_config.yaml'
    output:
        model_path / 'model_config.yaml'
    threads: 1
    shell:
        "ln -rfs {input} {output} "
        # "ln -s repeat_0/model_config.yaml {output}"

rule best_training_run:
    input:
        expand(model_path / 'repeat_{{repeat}}/trial{trial_number}/model_config.yaml',
               trial_number=range(n_trials)),
        expand(model_path / 'repeat_{{repeat}}/trial{trial_number}/finished.tmp',
               trial_number=range(n_trials))
    output:
        checkpoints = expand(model_path / 'repeat_{{repeat}}/best/bag_{bag}.ckpt',
                             bag=range(n_bags)),
        model_config = model_path / 'repeat_{repeat}/model_config.yaml'
    params:
        prefix = '.'
    threads: 1
    resources:
        mem_mb = 2048,
    log:
        stdout="logs/best_training_run/repeat_{repeat}.stdout", 
        stderr="logs/best_training_run/repeat_{repeat}.stderr"
    shell:
        (
            'deeprvat_train best-training-run '
            + debug +
            '{params.prefix}/{model_path}/repeat_{wildcards.repeat} '
            '{params.prefix}/{model_path}/repeat_{wildcards.repeat}/best '
            '{params.prefix}/{model_path}/repeat_{wildcards.repeat}/hyperparameter_optimization.db '
            '{output.model_config} '
            + logging_redirct
        )

rule train:
    input:
        data_config = expand('{phenotype}/deeprvat/config.yaml',
                        phenotype=training_phenotypes),
        input_tensor = expand('{phenotype}/deeprvat/input_tensor.zarr',
                              phenotype=training_phenotypes),
        covariates = expand('{phenotype}/deeprvat/covariates.zarr',
                            phenotype=training_phenotypes),
        y = expand('{phenotype}/deeprvat/y.zarr',
                   phenotype=training_phenotypes),
    output:
        expand(model_path / 'repeat_{repeat}/trial{trial_number}/model_config.yaml',
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
        mem_mb = 20000,
        gpus = 1
    shell:
        f"parallel --jobs {n_parallel_training_jobs} --halt now,fail=1 --results {{params.prefix}}/train_repeat{{{{1}}}}_trial{{{{2}}}}/ "
        'deeprvat_train train ' +
        debug +
        deterministic +
        '--trial-id {{2}} '
        "{params.phenotypes} "
        '{params.prefix}/deeprvat_config.yaml '
        '{params.prefix}/{model_path}/repeat_{{1}}/trial{{2}} '
        "{params.prefix}/{model_path}/repeat_{{1}}/hyperparameter_optimization.db '&&' "
        "touch {params.prefix}/{model_path}/repeat_{{1}}/trial{{2}}/finished.tmp "
        "::: " + " ".join(map(str, range(n_repeats))) + " "
        "::: " + " ".join(map(str, range(n_trials)))
