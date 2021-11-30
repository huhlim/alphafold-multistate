
import json
import logging
import math
import random
import sys
import time
import zipfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Tuple, List, Union, Optional

import haiku
import importlib_metadata
import numpy as np
import pandas
from jax.lib import xla_bridge
from numpy import ndarray

try:
    import alphafold
except ModuleNotFoundError:
    raise RuntimeError(
        "\n\nalphafold is not installed. Please run `pip install colabfold[alphafold]`\n"
    )

from alphafold.common import protein
from alphafold.common.protein import Protein
from alphafold.data import (
    pipeline,
    msa_pairing,
    pipeline_multimer,
    templates,
    feature_processing,
)
from alphafold.data.tools import hhsearch
from alphafold.model import model
from colabfold.alphafold.models import load_models_and_params
from colabfold.alphafold.msa import make_fixed_size
from colabfold.citations import write_bibtex
from colabfold.colabfold import run_mmseqs2, chain_break, plot_paes, plot_plddts
from colabfold.plot import plot_msa
from colabfold.download import download_alphafold_params, default_data_dir
from colabfold.utils import (
    setup_logging,
    safe_filename,
    NO_GPU_FOUND,
    DEFAULT_API_SERVER,
    ACCEPT_DEFAULT_TERMS,
    get_commit,
)
from colabfold.batch import *

logger = logging.getLogger(__name__)

def mk_template(
    a3m_lines: str, template_path: str, query_sequence: str
) -> Dict[str, Any]:
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir="gpcr100/mmcif",
        max_template_date="2100-01-01",
        max_hits=20,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None,
    )

    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path="hhsearch", databases=[template_path]
    )

    hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
    hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(
        query_sequence=query_sequence, hits=hhsearch_hits
    )
    return dict(templates_result.features)

def get_msa_and_templates(
    jobname: str,
    query_sequences: Union[str, List[str]],
    result_dir: Path,
    msa_mode: str,
    use_templates: bool,
    activation_state: str,
    pair_mode: str,
    host_url: str = DEFAULT_API_SERVER,
) -> Tuple[
    Optional[List[str]], Optional[List[str]], List[str], List[int], List[Dict[str, Any]]
]:
    use_env = msa_mode == "MMseqs2 (UniRef+Environmental)"
    # remove duplicates before searching
    query_sequences = (
        [query_sequences] if isinstance(query_sequences, str) else query_sequences
    )
    query_seqs_unique = []
    for x in query_sequences:
        if x not in query_seqs_unique:
            query_seqs_unique.append(x)
    query_seqs_cardinality = [0] * len(query_seqs_unique)
    for seq in query_sequences:
        seq_idx = query_seqs_unique.index(seq)
        query_seqs_cardinality[seq_idx] += 1

    template_features = []
    if use_templates:
        a3m_lines_mmseqs2, template_paths = run_mmseqs2(
            query_seqs_unique,
            str(result_dir.joinpath(jobname)),
            use_env,
            use_templates=True,
            host_url=host_url,
        )
        
        for index in range(0, len(query_seqs_unique)):
            template_feature = mk_template(
                a3m_lines_mmseqs2[index],
                f"gpcr100/GPCR100.{activation_state}",
                query_seqs_unique[index],
            )
            template_features.append(template_feature)
    else:
        for index in range(0, len(query_seqs_unique)):
            template_feature = mk_mock_template(query_seqs_unique[index])
            template_features.append(template_feature)

    if len(query_sequences) == 1:
        pair_mode = "none"

    if pair_mode == "none" or pair_mode == "unpaired" or pair_mode == "unpaired+paired":
        if msa_mode == "single_sequence":
            a3m_lines = []
            num = 101
            for i, seq in enumerate(query_seqs_unique):
                a3m_lines.append(">" + str(num + i) + "\n" + seq)
        else:
            # find normal a3ms
            a3m_lines = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_env,
                use_pairing=False,
                host_url=host_url,
            )
    else:
        a3m_lines = None

    if pair_mode == "paired" or pair_mode == "unpaired+paired":
        # find paired a3m if not a homooligomers
        if len(query_seqs_unique) > 1:
            paired_a3m_lines = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_env,
                use_pairing=True,
                host_url=host_url,
            )
        else:
            # homooligomers
            num = 101
            paired_a3m_lines = []
            for i in range(0, query_seqs_cardinality[0]):
                paired_a3m_lines.append(
                    ">" + str(num + i) + "\n" + query_seqs_unique[0] + "\n"
                )
    else:
        paired_a3m_lines = None

    return (
        a3m_lines,
        paired_a3m_lines,
        query_seqs_unique,
        query_seqs_cardinality,
        template_features,
    )

def remove_msa_for_template_aligned_regions(feature_dict):
    mask = np.zeros(feature_dict['seq_length'][0], dtype=bool)
    for templ in feature_dict['template_sequence']:
        for i,aa in enumerate(templ.decode("utf-8")):
            if aa != '-':
                mask[i] = True
    #
    feature_dict['deletion_matrix_int'][:,mask] = 0
    feature_dict['msa'][:,mask] = 21
    return feature_dict

def run(
    queries: List[Tuple[str, Union[str, List[str]], Optional[List[str]]]],
    result_dir: Union[str, Path],
    num_models: int,
    num_recycles: int,
    model_order: List[int],
    is_complex: bool,
    model_type: str = "auto",
    msa_mode: str = "MMseqs2 (UniRef+Environmental)",
    use_templates: bool = False,
    activation_state: str = "Inactive",
    use_amber: bool = False,
    keep_existing_results: bool = True,
    rank_mode: str = "auto",
    pair_mode: str = "unpaired+paired",
    data_dir: Union[str, Path] = default_data_dir,
    host_url: str = DEFAULT_API_SERVER,
    stop_at_score: float = 100,
    recompile_padding: float = 1.1,
    recompile_all_models: bool = False,
    zip_results: bool = False,
):
    version = importlib_metadata.version("colabfold")
    commit = get_commit()
    print(commit)
    if commit:
        version += f" ({commit})"

    logger.info(f"Running colabfold {version}")

    data_dir = Path(data_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)
    model_type = set_model_type(is_complex, model_type)
    if model_type == "AlphaFold2-multimer":
        model_extension = "_multimer"
    elif model_type == "AlphaFold2-ptm":
        model_extension = "_ptm"
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    # Record the parameters of this run
    config = {
        "num_queries": len(queries),
        "use_templates": use_templates,
        "activation_state": activation_state,
        "use_amber": use_amber,
        "msa_mode": msa_mode,
        "model_type": model_type,
        "num_models": num_models,
        "num_recycles": num_recycles,
        "model_order": model_order,
        "keep_existing_results": keep_existing_results,
        "rank_mode": rank_mode,
        "pair_mode": pair_mode,
        "host_url": host_url,
        "stop_at_score": stop_at_score,
        "recompile_padding": recompile_padding,
        "recompile_all_models": recompile_all_models,
        "commit": get_commit(),
        "version": importlib_metadata.version("colabfold"),
    }
    result_dir.joinpath("config.json").write_text(json.dumps(config, indent=4))
    use_env = msa_mode == "MMseqs2 (UniRef+Environmental)"
    use_msa = (
        msa_mode == "MMseqs2 (UniRef only)"
        or msa_mode == "MMseqs2 (UniRef+Environmental)"
    )

    bibtex_file = write_bibtex(
        model_type, use_msa, use_env, use_templates, use_amber, result_dir
    )

    model_runner_and_params = load_models_and_params(
        num_models,
        use_templates,
        num_recycles,
        model_order,
        model_extension,
        data_dir,
        recompile_all_models,
    )

    crop_len = 0
    for job_number, (raw_jobname, query_sequence, a3m_lines) in enumerate(queries):
        jobname = safe_filename(raw_jobname)
        # In the colab version and with --zip we know we're done when a zip file has been written
        result_zip = result_dir.joinpath(jobname).with_suffix(".result.zip")
        if keep_existing_results and result_zip.is_file():
            logger.info(f"Skipping {jobname} (result.zip)")
            continue
        # In the local version we use a marker file
        is_done_marker = result_dir.joinpath(jobname + ".done.txt")
        if keep_existing_results and is_done_marker.is_file():
            logger.info(f"Skipping {jobname} (already done)")
            continue

        query_sequence_len = (
            len(query_sequence)
            if isinstance(query_sequence, str)
            else sum(len(s) for s in query_sequence)
        )
        logger.info(
            f"Query {job_number + 1}/{len(queries)}: {jobname} (length {query_sequence_len})"
        )

        try:
            if a3m_lines is not None:
                (
                    unpaired_msa,
                    paired_msa,
                    query_seqs_unique,
                    query_seqs_cardinality,
                    template_features,
                ) = unserialize_msa(a3m_lines, query_sequence)
            else:
                (
                    unpaired_msa,
                    paired_msa,
                    query_seqs_unique,
                    query_seqs_cardinality,
                    template_features,
                ) = get_msa_and_templates(
                    jobname,
                    query_sequence,
                    result_dir,
                    msa_mode,
                    use_templates,
                    activation_state,
                    pair_mode,
                    host_url,
                )
            msa = msa_to_str(
                unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality
            )
            result_dir.joinpath(jobname + ".a3m").write_text(msa)
        except Exception as e:
            logger.exception(f"Could not get MSA/templates for {jobname}: {e}")
            continue
        try:
            input_features = generate_input_feature(
                query_seqs_unique,
                query_seqs_cardinality,
                unpaired_msa,
                paired_msa,
                template_features,
                is_complex,
                model_type,
            )
            input_features = remove_msa_for_template_aligned_regions(input_features)
        except Exception as e:
            logger.exception(f"Could not generate input features {jobname}: {e}")
            continue
        try:
            query_sequence_len_array = [
                len(query_seqs_unique[i])
                for i, cardinality in enumerate(query_seqs_cardinality)
                for _ in range(0, cardinality)
            ]

            if sum(query_sequence_len_array) > crop_len:
                crop_len = math.ceil(sum(query_sequence_len_array) * recompile_padding)

            outs, model_rank = predict_structure(
                jobname,
                result_dir,
                input_features,
                is_complex,
                use_templates,
                sequences_lengths=query_sequence_len_array,
                crop_len=crop_len,
                model_type=model_type,
                model_runner_and_params=model_runner_and_params,
                do_relax=use_amber,
                rank_by=rank_mode,
                stop_at_score=stop_at_score,
            )
        except RuntimeError as e:
            # This normally happens on OOM. TODO: Filter for the specific OOM error message
            logger.error(f"Could not predict {jobname}. Not Enough GPU memory? {e}")
            continue

        msa_plot = plot_msa(
            input_features["msa"],
            input_features["msa"][0],
            query_sequence_len_array,
            query_sequence_len,
        )
        msa_plot.savefig(str(result_dir.joinpath(jobname + "_coverage.png")))
        msa_plot.close()
        paes_plot = plot_paes(
            [outs[k]["pae"] for k in model_rank], Ls=query_sequence_len_array, dpi=200
        )
        paes_plot.savefig(str(result_dir.joinpath(jobname + "_PAE.png")))
        paes_plot.close()
        plddt_plot = plot_plddts(
            [outs[k]["plddt"] for k in model_rank], Ls=query_sequence_len_array, dpi=200
        )
        plddt_plot.savefig(str(result_dir.joinpath(jobname + "_plddt.png")))
        plddt_plot.close()

        if zip_results:
            result_files = (
                [
                    bibtex_file,
                    result_dir.joinpath("config.json"),
                    result_dir.joinpath(jobname + ".a3m"),
                ]
                + sorted(result_dir.glob(jobname + "*.png"))
                + sorted(result_dir.glob(f"{jobname}_unrelaxed_*.pdb"))
                + sorted(result_dir.glob(f"{jobname}_relaxed_*.pdb"))
            )

            with zipfile.ZipFile(result_zip, "w") as result_zip:
                for file in result_files:
                    result_zip.write(file, arcname=file.name)
            # Delete only after the zip was successful, and also not the bibtex and config because we need those again
            for file in result_files[2:]:
                file.unlink()
        else:
            is_done_marker.touch()

    logger.info("Done")
