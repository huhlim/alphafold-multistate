#!/bin/bash

EXEC=/home/huhlim/apps/AlphaFold/run.py
output_dir=af.seq_only

uniprot_id=$1

fa_fn=fa/$uniprot_id.fa
mkdir -p $output_dir

if [[ -e $output_dir/$uniprot_id/ranked_0.pdb ]]; then
    exit -1
fi

$EXEC --fasta_path=$fa_fn \
      --output_dir=$output_dir \
      --use_templates=False \
      --use_msa=False \
      &> $output_dir/$uniprot_id.log

