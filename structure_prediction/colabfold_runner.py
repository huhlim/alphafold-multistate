from colabfold.batch import *

logger = logging.getLogger(__name__)


def mk_template(
    a3m_lines: str, template_path: str, mmcif_path: str, query_sequence: str
) -> Dict[str, Any]:
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=mmcif_path,
        max_template_date="2100-01-01",
        max_hits=20,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None,
    )

    hhsearch_pdb70_runner = hhsearch.HHSearch(binary_path="hhsearch", databases=[template_path])

    hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
    hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(
        query_sequence=query_sequence, hits=hhsearch_hits
    )
    return dict(templates_result.features)


def get_msa_and_templates(
    jobname: str,
    protein_family: str,
    query_sequences: Union[str, List[str]],
    result_dir: Path,
    msa_mode: str,
    use_templates: bool,
    custom_template_path: str,
    conformational_state: str,
    pair_mode: str,
    host_url: str = DEFAULT_API_SERVER,
) -> Tuple[Optional[List[str]], Optional[List[str]], List[str], List[int], List[Dict[str, Any]]]:
    from colabfold.colabfold import run_mmseqs2

    use_env = msa_mode == "mmseqs2_uniref_env"
    if isinstance(query_sequences, str):
        query_sequences = [query_sequences]

    # remove duplicates before searching
    query_seqs_unique = []
    for x in query_sequences:
        if x not in query_seqs_unique:
            query_seqs_unique.append(x)

    # determine how many times is each sequence is used
    query_seqs_cardinality = [0] * len(query_seqs_unique)
    for seq in query_sequences:
        seq_idx = query_seqs_unique.index(seq)
        query_seqs_cardinality[seq_idx] += 1

    template_features = []
    if use_templates:
        a3m_lines_mmseqs2 = run_mmseqs2(
            query_seqs_unique,
            str(result_dir.joinpath(jobname)),
            use_env,
            use_templates=False,
            host_url=host_url,
        )

        if protein_family == "GPCR":
            mmcif_path = "gpcr100/mmcif"
            template_path = f"gpcr100/GPCR100.{conformational_state}"
        elif protein_family == "Kinase":
            mmcif_path = "kinase100/cif"
            template_path = f"kinase100/kinase100.{conformational_state}"

        if custom_template_path is not None:
            raise NotImplementedError
        for index in range(0, len(query_seqs_unique)):
            template_feature = mk_template(
                a3m_lines_mmseqs2[index],
                template_path,
                mmcif_path,
                query_seqs_unique[index],
            )
            logger.info(
                f"Sequence {index} found templates: {template_feature['template_domain_names']}"
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

    if msa_mode != "single_sequence" and (pair_mode == "paired" or pair_mode == "unpaired+paired"):
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
                paired_a3m_lines.append(">" + str(num + i) + "\n" + query_seqs_unique[0] + "\n")
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
    mask = np.zeros(feature_dict["seq_length"][0], dtype=bool)
    for templ in feature_dict["template_sequence"]:
        for i, aa in enumerate(templ.decode("utf-8")):
            if aa != "-":
                mask[i] = True
    #
    feature_dict["deletion_matrix_int"][:, mask] = 0
    feature_dict["msa"][:, mask] = 21
    return feature_dict


def run(
    queries: List[Tuple[str, Union[str, List[str]], Optional[List[str]]]],
    result_dir: Union[str, Path],
    num_models: int,
    is_complex: bool,
    num_recycles: Optional[int] = None,
    recycle_early_stop_tolerance: Optional[float] = None,
    model_order: List[int] = [1, 2],
    num_ensemble: int = 1,
    model_type: str = "auto",
    msa_mode: str = "mmseqs2_uniref_env",
    protein_family: str = "GPCR",
    conformational_state: str = "Inactive",
    use_templates: bool = False,
    custom_template_path: str = None,
    num_relax: int = 0,
    keep_existing_results: bool = True,
    rank_by: str = "auto",
    pair_mode: str = "unpaired_paired",
    data_dir: Union[str, Path] = default_data_dir,
    host_url: str = DEFAULT_API_SERVER,
    random_seed: int = 0,
    num_seeds: int = 1,
    recompile_padding: Union[int, float] = 10,
    zip_results: bool = False,
    prediction_callback: Callable[[Any, Any, Any, Any, Any], Any] = None,
    save_single_representations: bool = False,
    save_pair_representations: bool = False,
    save_all: bool = False,
    save_recycles: bool = False,
    use_dropout: bool = False,
    use_gpu_relax: bool = False,
    stop_at_score: float = 100,
    dpi: int = 200,
    max_seq: Optional[int] = None,
    max_extra_seq: Optional[int] = None,
    use_cluster_profile: bool = True,
    feature_dict_callback: Callable[[Any], Any] = None,
    **kwargs,
):
    # check what device is available
    try:
        # check if TPU is available
        import jax.tools.colab_tpu

        jax.tools.colab_tpu.setup_tpu()
        logger.info("Running on TPU")
        DEVICE = "tpu"
        use_gpu_relax = False
    except:
        if jax.local_devices()[0].platform == "cpu":
            logger.info("WARNING: no GPU detected, will be using CPU")
            DEVICE = "cpu"
            use_gpu_relax = False
        else:
            import tensorflow as tf

            logger.info("Running on GPU")
            DEVICE = "gpu"
            # disable GPU on tensorflow
            tf.config.set_visible_devices([], "GPU")

    from alphafold.notebooks.notebook_utils import get_pae_json
    from colabfold.alphafold.models import load_models_and_params
    from colabfold.colabfold import plot_paes, plot_plddts
    from colabfold.plot import plot_msa_v2

    data_dir = Path(data_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)
    model_type = set_model_type(is_complex, model_type)

    # backward-compatibility with old options
    old_names = {
        "MMseqs2 (UniRef+Environmental)": "mmseqs2_uniref_env",
        "MMseqs2 (UniRef only)": "mmseqs2_uniref",
        "unpaired+paired": "unpaired_paired",
    }
    msa_mode = old_names.get(msa_mode, msa_mode)
    pair_mode = old_names.get(pair_mode, pair_mode)
    feature_dict_callback = kwargs.pop("input_features_callback", feature_dict_callback)
    use_dropout = kwargs.pop("training", use_dropout)
    use_fuse = kwargs.pop("use_fuse", True)
    use_bfloat16 = kwargs.pop("use_bfloat16", True)
    max_msa = kwargs.pop("max_msa", None)
    if max_msa is not None:
        max_seq, max_extra_seq = [int(x) for x in max_msa.split(":")]

    if kwargs.pop("use_amber", False) and num_relax == 0:
        num_relax = num_models * num_seeds

    if len(kwargs) > 0:
        print(f"WARNING: the following options are not being used: {kwargs}")

    # decide how to rank outputs
    if rank_by == "auto":
        rank_by = "multimer" if is_complex else "plddt"
    if "ptm" not in model_type and "multimer" not in model_type:
        rank_by = "plddt"

    # get max length
    max_len = 0
    max_num = 0
    for _, query_sequence, _ in queries:
        N = 1 if isinstance(query_sequence, str) else len(query_sequence)
        L = len("".join(query_sequence))
        if L > max_len:
            max_len = L
        if N > max_num:
            max_num = N

    # get max sequences
    # 512 5120 = alphafold_ptm (models 1,3,4)
    # 512 1024 = alphafold_ptm (models 2,5)
    # 508 2048 = alphafold-multimer_v3 (models 1,2,3)
    # 508 1152 = alphafold-multimer_v3 (models 4,5)
    # 252 1152 = alphafold-multimer_v[1,2]

    set_if = lambda x, y: y if x is None else x
    if model_type in ["alphafold2_multimer_v1", "alphafold2_multimer_v2"]:
        (max_seq, max_extra_seq) = (set_if(max_seq, 252), set_if(max_extra_seq, 1152))
    elif model_type == "alphafold2_multimer_v3":
        (max_seq, max_extra_seq) = (set_if(max_seq, 508), set_if(max_extra_seq, 2048))
    else:
        (max_seq, max_extra_seq) = (set_if(max_seq, 512), set_if(max_extra_seq, 5120))

    if msa_mode == "single_sequence":
        num_seqs = 1
        if is_complex and "multimer" not in model_type:
            num_seqs += max_num
        if use_templates:
            num_seqs += 4
        max_seq = min(num_seqs, max_seq)
        max_extra_seq = max(min(num_seqs - max_seq, max_extra_seq), 1)

    # sort model order
    model_order.sort()

    # Record the parameters of this run
    config = {
        "num_queries": len(queries),
        "use_templates": use_templates,
        "protein_family": protein_family,
        "conformational_state": conformational_state,
        "num_relax": num_relax,
        "msa_mode": msa_mode,
        "model_type": model_type,
        "num_models": num_models,
        "num_recycles": num_recycles,
        "recycle_early_stop_tolerance": recycle_early_stop_tolerance,
        "num_ensemble": num_ensemble,
        "model_order": model_order,
        "keep_existing_results": keep_existing_results,
        "rank_by": rank_by,
        "max_seq": max_seq,
        "max_extra_seq": max_extra_seq,
        "pair_mode": pair_mode,
        "host_url": host_url,
        "stop_at_score": stop_at_score,
        "random_seed": random_seed,
        "num_seeds": num_seeds,
        "recompile_padding": recompile_padding,
        "commit": get_commit(),
        "use_dropout": use_dropout,
        "use_cluster_profile": use_cluster_profile,
        "use_fuse": use_fuse,
        "use_bfloat16": use_bfloat16,
        "version": importlib_metadata.version("colabfold"),
    }
    config_out_file = result_dir.joinpath("config.json")
    config_out_file.write_text(json.dumps(config, indent=4))
    use_env = "env" in msa_mode
    use_msa = "mmseqs2" in msa_mode
    use_amber = num_relax > 0

    bibtex_file = write_bibtex(model_type, use_msa, use_env, use_templates, use_amber, result_dir)
    if custom_template_path is not None:
        mk_hhsearch_db(custom_template_path)

    # get max length (for padding purposes)
    max_len = 0
    for _, query_sequence, _ in queries:
        L = len("".join(query_sequence))
        if L > max_len:
            max_len = L

    pad_len = 0
    ranks, metrics = [], []
    first_job = True
    for job_number, (raw_jobname, query_sequence, a3m_lines) in enumerate(queries):
        jobname = safe_filename(raw_jobname)

        #######################################
        # check if job has already finished
        #######################################
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

        total_len = len("".join(query_sequence))
        logger.info(f"Query {job_number + 1}/{len(queries)}: {jobname} (length {total_len})")

        ###########################################
        # generate MSA (a3m_lines) and templates
        ###########################################
        try:
            if use_templates or a3m_lines is None:
                (
                    unpaired_msa,
                    paired_msa,
                    query_seqs_unique,
                    query_seqs_cardinality,
                    template_features,
                ) = get_msa_and_templates(
                    jobname,
                    protein_family,
                    query_sequence,
                    result_dir,
                    msa_mode,
                    use_templates,
                    custom_template_path,
                    conformational_state,
                    pair_mode,
                    host_url,
                )
            if a3m_lines is not None:
                (
                    unpaired_msa,
                    paired_msa,
                    query_seqs_unique,
                    query_seqs_cardinality,
                    template_features_,
                ) = unserialize_msa(a3m_lines, query_sequence)
                if not use_templates:
                    template_features = template_features_

            # save a3m
            msa = msa_to_str(unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality)
            result_dir.joinpath(f"{jobname}.a3m").write_text(msa)

        except Exception as e:
            logger.exception(f"Could not get MSA/templates for {jobname}: {e}")
            continue

        #######################
        # generate features
        #######################
        try:
            (feature_dict, domain_names) = generate_input_feature(
                query_seqs_unique,
                query_seqs_cardinality,
                unpaired_msa,
                paired_msa,
                template_features,
                is_complex,
                model_type,
                max_seq=max_seq,
            )
            remove_msa_for_template_aligned_regions(feature_dict)

            # to allow display of MSA info during colab/chimera run (thanks tomgoddard)
            if feature_dict_callback is not None:
                feature_dict_callback(feature_dict)

        except Exception as e:
            logger.exception(f"Could not generate input features {jobname}: {e}")
            continue

        ######################
        # predict structures
        ######################
        try:
            # get list of lengths
            query_sequence_len_array = sum(
                [[len(x)] * y for x, y in zip(query_seqs_unique, query_seqs_cardinality)], []
            )

            # decide how much to pad (to avoid recompiling)
            if total_len > pad_len:
                if isinstance(recompile_padding, float):
                    pad_len = math.ceil(total_len * recompile_padding)
                else:
                    pad_len = total_len + recompile_padding
                pad_len = min(pad_len, max_len)
                logger.info(f"Padding length to {pad_len}")

            # prep model and params
            if first_job:
                # if one job input adjust max settings
                if len(queries) == 1 or msa_mode == "single_sequence":
                    # get number of sequences
                    if "msa_mask" in feature_dict:
                        num_seqs = int(sum(feature_dict["msa_mask"].max(-1) == 1))
                    else:
                        num_seqs = int(len(feature_dict["msa"]))

                    if use_templates:
                        num_seqs += 4

                    # adjust max settings
                    max_seq = min(num_seqs, max_seq)
                    max_extra_seq = max(min(num_seqs - max_seq, max_extra_seq), 1)
                    logger.info(f"Setting max_seq={max_seq}, max_extra_seq={max_extra_seq}")

                model_runner_and_params = load_models_and_params(
                    num_models=num_models,
                    use_templates=use_templates,
                    num_recycles=num_recycles,
                    num_ensemble=num_ensemble,
                    model_order=model_order,
                    model_type=model_type,
                    data_dir=data_dir,
                    stop_at_score=stop_at_score,
                    rank_by=rank_by,
                    use_dropout=use_dropout,
                    max_seq=max_seq,
                    max_extra_seq=max_extra_seq,
                    use_cluster_profile=use_cluster_profile,
                    recycle_early_stop_tolerance=recycle_early_stop_tolerance,
                    use_fuse=use_fuse,
                    use_bfloat16=use_bfloat16,
                    save_all=save_all,
                )
                first_job = False

            results = predict_structure(
                prefix=jobname,
                result_dir=result_dir,
                feature_dict=feature_dict,
                is_complex=is_complex,
                use_templates=use_templates,
                sequences_lengths=query_sequence_len_array,
                pad_len=pad_len,
                model_type=model_type,
                model_runner_and_params=model_runner_and_params,
                num_relax=num_relax,
                rank_by=rank_by,
                stop_at_score=stop_at_score,
                prediction_callback=prediction_callback,
                use_gpu_relax=use_gpu_relax,
                random_seed=random_seed,
                num_seeds=num_seeds,
                save_all=save_all,
                save_single_representations=save_single_representations,
                save_pair_representations=save_pair_representations,
                save_recycles=save_recycles,
            )
            result_files = results["result_files"]
            ranks.append(results["rank"])
            metrics.append(results["metric"])

        except RuntimeError as e:
            # This normally happens on OOM. TODO: Filter for the specific OOM error message
            logger.error(f"Could not predict {jobname}. Not Enough GPU memory? {e}")
            continue

        ###############
        # save plots
        ###############

        # make msa plot
        msa_plot = plot_msa_v2(feature_dict, dpi=dpi)
        coverage_png = result_dir.joinpath(f"{jobname}_coverage.png")
        msa_plot.savefig(str(coverage_png), bbox_inches="tight")
        msa_plot.close()
        result_files.append(coverage_png)

        # load the scores
        scores = []
        for r in results["rank"][:5]:
            scores_file = result_dir.joinpath(f"{jobname}_scores_{r}.json")
            with scores_file.open("r") as handle:
                scores.append(json.load(handle))

        # write alphafold-db format (pAE)
        af_pae_file = result_dir.joinpath(f"{jobname}_predicted_aligned_error_v1.json")
        af_pae_file.write_text(
            json.dumps(
                {
                    "predicted_aligned_error": scores[0]["pae"],
                    "max_predicted_aligned_error": scores[0]["max_pae"],
                }
            )
        )
        result_files.append(af_pae_file)

        # make pAE plots
        paes_plot = plot_paes(
            [np.asarray(x["pae"]) for x in scores], Ls=query_sequence_len_array, dpi=dpi
        )
        pae_png = result_dir.joinpath(f"{jobname}_pae.png")
        paes_plot.savefig(str(pae_png), bbox_inches="tight")
        paes_plot.close()
        result_files.append(pae_png)

        # make pLDDT plot
        plddt_plot = plot_plddts(
            [np.asarray(x["plddt"]) for x in scores], Ls=query_sequence_len_array, dpi=dpi
        )
        plddt_png = result_dir.joinpath(f"{jobname}_plddt.png")
        plddt_plot.savefig(str(plddt_png), bbox_inches="tight")
        plddt_plot.close()
        result_files.append(plddt_png)

        if use_templates:
            templates_file = result_dir.joinpath(f"{jobname}_template_domain_names.json")
            templates_file.write_text(json.dumps(domain_names))
            result_files.append(templates_file)

        result_files.append(result_dir.joinpath(jobname + ".a3m"))
        result_files += [bibtex_file, config_out_file]

        if zip_results:
            with zipfile.ZipFile(result_zip, "w") as result_zip:
                for file in result_files:
                    result_zip.write(file, arcname=file.name)

            # Delete only after the zip was successful, and also not the bibtex and config because we need those again
            for file in result_files[:-2]:
                file.unlink()
        else:
            is_done_marker.touch()

    logger.info("Done")
    return {"rank": ranks, "metric": metrics}
