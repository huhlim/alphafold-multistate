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

    use_env = msa_mode == "MMseqs2 (UniRef+Environmental)"
    # remove duplicates before searching
    query_sequences = [query_sequences] if isinstance(query_sequences, str) else query_sequences
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
        a3m_lines_mmseqs2 = run_mmseqs2(
            query_seqs_unique,
            str(result_dir.joinpath(jobname)),
            use_env,
            # use_templates=True,
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
    num_recycles: int,
    model_order: List[int],
    is_complex: bool,
    model_type: str = "auto",
    msa_mode: str = "MMseqs2 (UniRef+Environmental)",
    protein_family: str = "GPCR",
    conformational_state: str = "Inactive",
    use_templates: bool = False,
    custom_template_path: str = None,
    use_amber: bool = False,
    keep_existing_results: bool = True,
    rank_by: str = "auto",
    pair_mode: str = "unpaired+paired",
    data_dir: Union[str, Path] = default_data_dir,
    host_url: str = DEFAULT_API_SERVER,
    stop_at_score: float = 100,
    recompile_padding: float = 1.1,
    recompile_all_models: bool = False,
    zip_results: bool = False,
    prediction_callback: Callable[[Any, Any, Any, Any, Any], Any] = None,
    save_single_representations: bool = False,
    save_pair_representations: bool = False,
    training: bool = False,
    use_gpu_relax: bool = False,
    stop_at_score_below: float = 0,
    dpi: int = 200,
    max_msa: str = None,
):
    from alphafold.notebooks.notebook_utils import get_pae_json
    from colabfold.alphafold.models import load_models_and_params
    from colabfold.colabfold import plot_paes, plot_plddts
    from colabfold.plot import plot_msa

    data_dir = Path(data_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)

    model_type = "alphafold2_ptm"
    model_suffix = "_ptm"

    if rank_by == "auto":
        # score complexes by ptmscore and sequences by plddt
        rank_by = "plddt" if not is_complex else "ptmscore"
        rank_by = (
            "multimer" if is_complex and model_type.startswith("AlphaFold2-multimer") else rank_by
        )

    # Record the parameters of this run
    config = {
        "num_queries": len(queries),
        "use_templates": use_templates,
        "protein_family": protein_family,
        "conformational_state": conformational_state,
        "use_amber": use_amber,
        "msa_mode": msa_mode,
        "model_type": model_type,
        "num_models": num_models,
        "num_recycles": num_recycles,
        "model_order": model_order,
        "keep_existing_results": keep_existing_results,
        "rank_by": rank_by,
        "pair_mode": pair_mode,
        "host_url": host_url,
        "stop_at_score": stop_at_score,
        "stop_at_score_below": stop_at_score_below,
        "recompile_padding": recompile_padding,
        "recompile_all_models": recompile_all_models,
        "commit": get_commit(),
        "is_training": training,
        "version": importlib_metadata.version("colabfold"),
    }
    config_out_file = result_dir.joinpath("config.json")
    config_out_file.write_text(json.dumps(config, indent=4))
    use_env = msa_mode == "MMseqs2 (UniRef+Environmental)"
    use_msa = msa_mode == "MMseqs2 (UniRef only)" or msa_mode == "MMseqs2 (UniRef+Environmental)"

    bibtex_file = write_bibtex(model_type, use_msa, use_env, use_templates, use_amber, result_dir)

    save_representations = save_single_representations or save_pair_representations

    model_runner_and_params = load_models_and_params(
        #     num_models,
        #     use_templates,
        #     num_recycles,
        #     1,
        #     model_order,
        #     model_extension,
        #     data_dir,
        #     recompile_all_models,
        #     stop_at_score=stop_at_score,
        #     rank_by=rank_by,
        #     return_representations=save_representations,
        #     training=training,
        #     max_msa=max_msa,
        # )
        num_models=num_models,
        use_templates=use_templates,
        num_recycles=num_recycles,
        num_ensemble=1,
        model_order=model_order,
        model_suffix=model_suffix,
        data_dir=data_dir,
        stop_at_score=stop_at_score,
        rank_by=rank_by,
        use_dropout=training,
    )
    if custom_template_path is not None:
        mk_hhsearch_db(custom_template_path)

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
                    protein_family,
                    query_sequence,
                    result_dir,
                    msa_mode,
                    use_templates,
                    custom_template_path,
                    conformational_state,  # TODO
                    pair_mode,
                    host_url,
                )
            msa = msa_to_str(unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality)
            result_dir.joinpath(jobname + ".a3m").write_text(msa)
        except Exception as e:
            logger.exception(f"Could not get MSA/templates for {jobname}: {e}")
            continue
        try:
            input_features, domain_names = generate_input_feature(
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
                rank_by=rank_by,
                stop_at_score=stop_at_score,
                stop_at_score_below=stop_at_score_below,
                prediction_callback=prediction_callback,
                use_gpu_relax=use_gpu_relax,
            )
        except RuntimeError as e:
            # This normally happens on OOM. TODO: Filter for the specific OOM error message
            logger.error(f"Could not predict {jobname}. Not Enough GPU memory? {e}")
            continue

        # Write representations if needed

        representation_files = []

        if save_representations:
            for i, key in enumerate(model_rank):
                out = outs[key]
                model_id = i + 1
                model_name = out["model_name"]
                representations = out["representations"]

                if save_single_representations:
                    single_representation = np.asarray(representations["single"])
                    single_filename = result_dir.joinpath(
                        f"{jobname}_single_repr_{model_id}_{model_name}"
                    )
                    np.save(single_filename, single_representation)

                if save_pair_representations:
                    pair_representation = np.asarray(representations["pair"])
                    pair_filename = result_dir.joinpath(
                        f"{jobname}_pair_repr_{model_id}_{model_name}"
                    )
                    np.save(pair_filename, pair_representation)

        # Write alphafold-db format (PAE)
        alphafold_pae_file = result_dir.joinpath(jobname + "_predicted_aligned_error_v1.json")
        alphafold_pae_file.write_text(get_pae_json(outs[0]["pae"], outs[0]["max_pae"]))
        num_alignment = (
            int(input_features["num_alignments"])
            if model_type.startswith("AlphaFold2-multimer")
            else input_features["num_alignments"][0]
        )
        msa_plot = plot_msa(
            input_features["msa"][0:num_alignment],
            input_features["msa"][0],
            query_sequence_len_array,
            query_sequence_len,
        )
        coverage_png = result_dir.joinpath(jobname + "_coverage.png")
        msa_plot.savefig(str(coverage_png))
        msa_plot.close()
        paes_plot = plot_paes(
            [outs[k]["pae"] for k in model_rank], Ls=query_sequence_len_array, dpi=200
        )
        pae_png = result_dir.joinpath(jobname + "_PAE.png")
        paes_plot.savefig(str(pae_png))
        paes_plot.close()
        plddt_plot = plot_plddts(
            [outs[k]["plddt"] for k in model_rank], Ls=query_sequence_len_array, dpi=200
        )
        plddt_png = result_dir.joinpath(jobname + "_plddt.png")
        plddt_plot.savefig(str(plddt_png))
        plddt_plot.close()
        result_files = [
            bibtex_file,
            config_out_file,
            alphafold_pae_file,
            result_dir.joinpath(jobname + ".a3m"),
            pae_png,
            coverage_png,
            plddt_png,
            *representation_files,
        ]
        if use_templates:
            templates_file = result_dir.joinpath(jobname + "_template_domain_names.json")
            templates_file.write_text(json.dumps(domain_names))
            result_files.append(templates_file)
        for i, key in enumerate(model_rank):
            result_files.append(
                result_dir.joinpath(
                    f"{jobname}_unrelaxed_rank_{i + 1}_{outs[key]['model_name']}.pdb"
                )
            )
            result_files.append(
                result_dir.joinpath(
                    f"{jobname}_unrelaxed_rank_{i + 1}_{outs[key]['model_name']}_scores.json"
                )
            )
            if use_amber:
                result_files.append(
                    result_dir.joinpath(
                        f"{jobname}_relaxed_rank_{i + 1}_{outs[key]['model_name']}.pdb"
                    )
                )

        if zip_results:
            with zipfile.ZipFile(result_zip, "w") as result_zip:
                for file in result_files:
                    result_zip.write(file, arcname=file.name)
            # Delete only after the zip was successful, and also not the bibtex and config because we need those again
            for file in result_files[2:]:
                file.unlink()
        else:
            is_done_marker.touch()

    logger.info("Done")
