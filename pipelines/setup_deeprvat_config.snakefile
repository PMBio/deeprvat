

rule create_main_deeprvat_config:
    input:
        config_file = 'deeprvat_input_pretrained_models_config.yaml' #'deeprvat/example/config/deeprvat_input_config.yaml',
    output:
        'deeprvat_config.yaml'
    shell:
        (
            "deeprvat_config create-main-config "
            "{input.config_file} "
        )