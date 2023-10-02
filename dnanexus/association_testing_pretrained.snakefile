from pathlib import Path
from typing import Iterable, Union

configfile: 'config.yaml'

debug_flag = config.get('debug', False)
phenotypes = config['phenotypes']
phenotypes = list(phenotypes.keys()) if type(phenotypes) == dict else phenotypes

n_burden_chunks = config.get('n_burden_chunks', 1) if not debug_flag else 2
n_repeats = config['n_repeats']
debug = '--debug ' if debug_flag else ''
pretrained_model_path = Path(config.get("pretrained_model_path", "pretrained_models"))

dnanexus_destination = Path(config["dnanexus"]["destination"])
dnanexus_applet = config["dnanexus"]["applet"]
dnanexus_priority = config["dnanexus"].get("priority", "low")
dnanexus_configfile = config["dnanexus"]["configfile"]

model_checkpoints = [
    pretrained_model_path / f'repeat_{repeat}/best/bag_0.ckpt'
    for repeat in range(n_repeats)
]


def dx_run(
        command: str,
        mkdirs: Union[str, Iterable[str]],
        instance_type: str,
        dx_configfile: str = dnanexus_configfile,
        cost_limit: float = 1.00,
        destination: str = dnanexus_destination,
        applet: str = dnanexus_applet,
        dx_priority: str = dnanexus_priority,
):
    if isinstance(mkdirs, str):
        mkdirs = [mkdirs]

    mkdir_string = " && ".join(f"mkdir -p {d}" for d in mkdirs)

    dx_run_shell = f"dx run {applet} "
    dx_run_shell += f"--instance-type {instance_type} "
    dx_run_shell += f"--priority {dx_priority} "
    dx_run_shell += f"--cost-limit {cost_limit} "
    dx_run_shell += f"-iconfig={dx_configfile} "
    dx_run_shell += f"-icommand='" + mkdir_string
    dx_run_shell += f" && {command}' "
    dx_run_shell += f"--destination {destination} "
    dx_run_shell += f"--watch "
    dx_run_shell += f"-y "

    return dx_run_shell

wildcard_constraints:
    repeat="\d+",
    trial="\d+",


rule all:
    input:
        expand("{phenotype}/deeprvat/burdens/chunk{chunk}.finished",
               phenotype=phenotypes,
               chunk=n_burden_chunks)

rule compute_burdens:
    priority: 10
    input:
        dataset = '{phenotype}/deeprvat/association_dataset.pkl',
        data_config = '{phenotype}/deeprvat/hpopt_config.yaml',
        model_config = pretrained_model_path / 'config.yaml',
    output:
        '{phenotype}/deeprvat/burdens/chunk{chunk}.finished'
    threads: 1
    params:
        dx_run = lambda wildcards, input, output: dx_run(
            command=(
                'deeprvat_associate compute-burdens '
                + debug +
                f' --n-chunks {n_burden_chunks} '
                f'--chunk {wildcards.chunk} '
                f'--dataset-file ' + str("/mnt/project/DeepRVAT" / dnanexus_destination / input.dataset) + " " +
                str("/mnt/project/DeepRVAT" / dnanexus_destination / f'{input.data_config} ') +
                str("/mnt/project/DeepRVAT" / dnanexus_destination / f'{input.model_config} ') +
                " ".join([
                    str("/mnt/project/DeepRVAT" / dnanexus_destination / x)
                    for x in model_checkpoints
                ]) +
                f' {wildcards.phenotype}/deeprvat/burdens'
            ),
            mkdirs=f"{wildcards.phenotype}/deeprvat/burdens",
            instance_type="mem2_ssd1_gpu_x16",
            dx_priority = "high",
            cost_limit=20,
        ),
    shell:
        " && ".join([
            "{params.dx_run}",
            "touch {output}"
        ])
    # shell:
    #     ' && '.join([
    #         ('deeprvat_associate compute-burdens '
    #          + debug +
    #          ' --n-chunks '+ str(n_burden_chunks) + ' '
    #          '--chunk {wildcards.chunk} '
    #          '--dataset-file {input.dataset} '
    #          '{input.data_config} '
    #          '{input.model_config} '
    #          '{input.checkpoints} '
    #          '{wildcards.phenotype}/deeprvat/burdens'),
    #         'touch {output}'
    #     ])

# rule all_association_dataset:
#     input:
#         expand('{phenotype}/deeprvat/association_dataset.pkl',
#                phenotype=phenotypes)

# rule association_dataset:
#     input:
#         config = '{phenotype}/deeprvat/hpopt_config.yaml'
#     output:
#         '{phenotype}/deeprvat/association_dataset.pkl'
#     threads: 1
#     params:
#         dx_run = lambda wildcards, input, output: dx_run(
#             command=(
#                 'deeprvat_associate make-dataset '
#                 + debug +
#                 str("/mnt/project/DeepRVAT" / dnanexus_destination / f'{input.config} ') +
#                 f'{output}'
#             ),
#             mkdirs=f"{wildcards.phenotype}/deeprvat",
#             instance_type="mem3_ssd1_v2_x4",
#             cost_limit=1,
#         ),
#     shell:
#         " && ".join([
#             "{params.dx_run}",
#             "touch {output}"
#         ])

# rule all_config:
#     input:
#         config = expand('{phenotype}/deeprvat/hpopt_config.yaml',
#                         phenotype=phenotypes),

# rule config:
#     input:
#         config = 'config.yaml',
#     output:
#         config = '{phenotype}/deeprvat/hpopt_config.yaml',
#     params:
#         dx_run = lambda wildcards, input, output: dx_run(
#             command=(
#                 'deeprvat_config update-config '
#                 f'--phenotype {wildcards.phenotype} '
#                 f'{input.config} '
#                 f'{output.config}'
#             ),
#             mkdirs=f"{wildcards.phenotype}/deeprvat",
#             instance_type="mem1_ssd1_v2_x2",
#             cost_limit=0.10,
#         ),
#     threads: 1
#     shell:
#         " && ".join([
#             "dx mkdir -p DeepRVAT/workdir/pretrained_scoring",
#             # "dx upload config.yaml --destination DeepRVAT/workdir/pretrained_scoring/config.yaml",
#             "{params.dx_run}",
#             "touch {output}"
#         ])
