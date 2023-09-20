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
    dx_run_shell += f"--wait "
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
        reversed = pretrained_model_path / "reverse_finished.tmp",
        checkpoints = lambda wildcards: [
            pretrained_model_path / f'repeat_{repeat}/best/bag_{bag}.ckpt'
            for repeat in range(n_repeats) for bag in range(n_bags)
        ],
        dataset = '{phenotype}/deeprvat/association_dataset.pkl',
        data_config = '{phenotype}/deeprvat/hpopt_config.yaml',
        model_config = pretrained_model_path / 'config.yaml',
    output:
        '{phenotype}/deeprvat/burdens/chunk{chunk}.finished'
    threads: 8
    shell:
        ' && '.join([
            ('deeprvat_associate compute-burdens '
             + debug +
             ' --n-chunks '+ str(n_burden_chunks) + ' '
             '--chunk {wildcards.chunk} '
             '--dataset-file {input.dataset} '
             '{input.data_config} '
             '{input.model_config} '
             '{input.checkpoints} '
             '{wildcards.phenotype}/deeprvat/burdens'),
            'touch {output}'
        ])

rule all_association_dataset:
    input:
        expand('{phenotype}/deeprvat/association_dataset.pkl',
               phenotype=phenotypes)

rule association_dataset:
    input:
        config = '{phenotype}/deeprvat/hpopt_config.yaml'
    output:
        '{phenotype}/deeprvat/association_dataset.pkl'
    threads: 1
    params:
        dx_run = lambda wildcards, input, output: dx_run(
            command=(
                'deeprvat_associate make-dataset '
                + debug +
                str("/mnt/project/DeepRVAT" / dnanexus_destination / f'{input.config} ') +
                f'{output}'
            ),
            mkdirs=f"{wildcards.phenotype}/deeprvat",
            instance_type="mem3_ssd1_v2_x4",
            cost_limit=1,
        ),
    shell:
        " && ".join([
            "{params.dx_run}",
            "touch {output}"
        ])

rule all_config:
    input:
        config = expand('{phenotype}/deeprvat/hpopt_config.yaml',
                        phenotype=phenotypes),

rule config:
    input:
        config = 'config.yaml',
    output:
        config = '{phenotype}/deeprvat/hpopt_config.yaml',
    params:
        dx_run = lambda wildcards, input, output: dx_run(
            command=(
                'deeprvat_config update-config '
                f'--phenotype {wildcards.phenotype} '
                f'{input.config} '
                f'{output.config}'
            ),
            mkdirs=f"{wildcards.phenotype}/deeprvat",
            instance_type="mem1_ssd1_v2_x2",
            cost_limit=0.10,
        ),
    threads: 1
    shell:
        " && ".join([
            "{params.dx_run}",
            "touch {output}"
        ])
