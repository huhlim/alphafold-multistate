#!/bin/bash

EXEC=/home/huhlim/apps/AlphaFold/run.py
output_dir=af

uniprot_id=$1

fa_fn=fa/$uniprot_id.fa
mkdir -p $output_dir

if [[ -e $output_dir/$uniprot_id/ranked_4.pdb ]]; then
    exit -1
fi

$EXEC --fasta_path=$fa_fn \
      --output_dir=$output_dir \
      --max_sequence_identity=70.0 \
      &> $output_dir/$uniprot_id.log

