from tqdm import tqdm
import logging
import os
import sys
import zarr
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import random
import math
from numcodecs import Blosc
from typing import Dict, List, Optional, Tuple
import click

compression_level = 1

AGG_FCT = {'mean': np.mean,
          'max': np.max}

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    pass


# python /home/e400p/home/deeprvat_public/deeprvat/deeprvat/deeprvat/average_burdens.py  average-burdens --n-chunks 20 --chunk 1 --n-repeats 6 --n-total-repeats 30  --n-combinations 2 --combination 1 --agg-fct mean  /home/e400p/ukbb/experiments/revision_2/multiple_repeats/Apolipoprotein_A/deeprvat/burdens/burdens.zarr  ~/home/analysis/deeprvat_score_analysis/test/avg_burdens.zarr

@cli.command()
@click.option("--n-chunks", type=int)
@click.option("--chunk", type=int)
@click.option("--initialize-zarr-only", is_flag=True)
@click.option("--n-repeats", type=int, default = 6)
@click.option("--n-total-repeats", type=int, default = 6)
@click.option("--n-combinations", type=int, default = 1)
@click.option("--combination", type=int, default = 0)
@click.option("--agg-fct", type=str, default = 'mean')
@click.argument("burden-file", type=click.Path(exists=True))
@click.argument("burden-out-file", type=click.Path())
def average_burdens(
    n_repeats: int,
    n_total_repeats: int,
    burden_file: str, 
    burden_out_file: str,
    initialize_zarr_only: bool, 
    agg_fct: Optional[str] = 'mean',
    n_combinations: Optional[int] = 1,
    combination: Optional[int] = 0,
    n_chunks: Optional[int] = None,
    chunk: Optional[int] = None,
):
    random.seed(combination) #set the random seed so that the repeats for a given combination are the same across chunks
    if n_repeats < n_total_repeats:
        all_repeat_combinations = list(combinations(range(n_total_repeats), n_repeats))
        this_repeats = random.sample(all_repeat_combinations, 1)[0]
        logger.info(f'Analyzing {n_repeats} repeats: {this_repeats} out {n_total_repeats} repeats')
    else:
        this_repeats = tuple([i for i in range(n_repeats)])
        logger.info(f'Aggregating across all repeats {this_repeats}')
    logger.info(f'Reading burdens to aggregate from {burden_file}')
    burdens = zarr.open(burden_file)
    n_total_samples = burdens.shape[0]
    if chunk is not None:
        if n_chunks is None:
            raise ValueError("n_chunks must be specified if chunk is not None")
        chunk_length = math.ceil(n_total_samples / n_chunks)
        chunk_start = chunk * chunk_length
        chunk_end = min(n_total_samples, chunk_start + chunk_length)
        samples = range(chunk_start, chunk_end)
        n_samples = len(samples)
        print(chunk_start, chunk_end)
    else:
        n_samples = n_total_samples
        chunk_start = 0
        chunk_end = n_samples
    
    logger.info(f'Computing result for chunk {chunk} out of {n_chunks} in range {chunk_start}, {chunk_end}')

    batch_size = 100
    logger.info(f'Batch size: {batch_size}')
    n_batches = (n_samples // batch_size +
                                        (n_samples % batch_size != 0))
    
    logger.info(f'Using aggregation function {agg_fct}')
    if initialize_zarr_only:
        logger.info('Only initializing zarr file')
        burdens_new = zarr.open(
                    burden_out_file,
                    mode='a',
                    shape=(burdens.shape[0], burdens.shape[1], n_combinations),
                    chunks=(1000, 1000),
                    dtype=np.float32,
                    compressor=Blosc(clevel=compression_level))
        logger.info('Finished')
        sys.exit()

    for i in tqdm(range(n_batches),
                                file=sys.stdout,
                                total=(n_samples // batch_size +
                                        (n_samples % batch_size != 0))):
        if i == 0:
            if not os.path.exists(burden_out_file): 
                logger.info('Generting new zarr file')
                burdens_new = zarr.open(
                    burden_out_file,
                    mode='a',
                    shape=(burdens.shape[0], burdens.shape[1], n_combinations),
                    chunks=(1000, 1000),
                    dtype=np.float32,
                    compressor=Blosc(clevel=compression_level))
            else: 
                logger.info('Only opening zarr file')
                burdens_new =  zarr.open(burden_out_file)
            
        start_idx = chunk_start + i * batch_size
        end_idx = min(start_idx + batch_size, chunk_end)
        print(start_idx, end_idx)
        this_burdens = np.take(burdens[start_idx:end_idx, :,:], this_repeats, axis = 2)
        this_burdens = AGG_FCT[agg_fct](this_burdens, axis = 2) 

        burdens_new[start_idx:end_idx, :, combination] = this_burdens
    
    logger.info(f'Writing aggregted burdens in range {chunk_start}, {chunk_end} to {burden_out_file}')
    


if __name__ == "__main__":
    cli()