#!/usr/bin/env python

import os
import sys
import pathlib
import argparse

# to prioritize the original AF2
sys.path = sys.path[1:] + sys.path[:1]

import colabfold_runner
from libconfig_af import (
    data_dir,
    gpcr100_active_db_path,
    gpcr100_inactive_db_path,
    mmcif_active_db_path,
    mmcif_inactive_db_path,
)


def read_sequence(fa_fn):
    seq = []
    with open(fa_fn) as fp:
        for line in fp:
            if not line.startswith(">"):
                seq.append(line.strip())
    return "".join(seq)


def main():
    arg = argparse.ArgumentParser(prog="run_multi-state_modeling using ColabFold")
    arg.add_argument(dest="fa_fn", help="input FASTA file")
    arg.add_argument(
        "-o", "--output_dir", dest="output_dir", default="./", help="output_dir (default=./)"
    )
    arg.add_argument(
        "-s",
        "--state",
        dest="state",
        default="inactive",
        choices=["active", "inactive"],
        help="modeling state (default=inactive)",
    )
    #
    if len(sys.argv) == 1:
        arg.print_help()
        return
    #
    arg = arg.parse_args()
    arg.fa_fn = pathlib.Path(arg.fa_fn)
    #
    jobname = arg.fa_fn.stem
    queries = [(jobname, read_sequence(arg.fa_fn), None)]
    #
    result_dir = pathlib.Path(arg.output_dir) / jobname
    if not result_dir.exists():
        result_dir.mkdir(parents=True)
    #
    conformational_state = {"active": "Active", "inactive": "Inactive"}[arg.state]
    #
    gpcr100_dir = pathlib.Path("gpcr100")
    if not gpcr100_dir.exists():
        gpcr100_dir.mkdir()
    mmcif_dir = gpcr100_dir / "mmcif"
    if not mmcif_dir.exists():
        mmcif_dir.mkdir()

    if conformational_state == "Active":
        db_home = pathlib.Path(gpcr100_active_db_path)
        cif_home = pathlib.Path(mmcif_active_db_path)
    else:
        db_home = pathlib.Path(gpcr100_inactive_db_path)
        cif_home = pathlib.Path(mmcif_inactive_db_path)
    db_home = db_home.parent
    #
    for name in ["hhm", "a3m", "cs219"]:
        for suffix in ["ffindex", "ffdata"]:
            src_fn = db_home / f"GPCR100.{conformational_state}_{name}.{suffix}"
            dst_fn = gpcr100_dir / f"GPCR100.{conformational_state}_{name}.{suffix}"
            if not dst_fn.exists():
                os.symlink(src_fn, dst_fn)
    for src_fn in cif_home.glob("*.cif"):
        dst_fn = mmcif_dir / src_fn.name
        if not dst_fn.exists():
            os.symlink(src_fn, dst_fn)

    results = colabfold_runner.run(
        queries=queries,
        result_dir=result_dir,
        protein_family="GPCR",
        conformational_state=conformational_state,
        use_templates=True,
        custom_template_path=None,
        use_amber=False,
        msa_mode="MMseqs2 (UniRef+Environmental)",
        model_type="auto",
        num_models=5,
        num_recycles=3,
        model_order=[1, 2],
        is_complex=False,
        data_dir=pathlib.Path(data_dir),
        keep_existing_results=False,
        rank_by="auto",
        stop_at_score=float(100),
    )


if __name__ == "__main__":
    main()
