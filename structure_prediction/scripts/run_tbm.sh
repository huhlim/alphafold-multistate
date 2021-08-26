#!/bin/bash

EXEC=/home/huhlim/apps/AlphaFold/run_tbm.py
output_dir=tbm

uniprot_id=$1

fa_fn=fa/$uniprot_id.fa
mkdir -p $output_dir

if [[ -e $output_dir/$uniprot_id/$uniprot_id.model.pdb ]]; then
    exit -1
fi
mkdir -p $output_dir/$uniprot_id
if [[ ! -e $output_dir/$uniprot_id/msas ]]; then
    ln -sf ../../af/$uniprot_id/msas $output_dir/$uniprot_id/msas
fi

$EXEC --fasta_path=$fa_fn \
      --output_dir=$output_dir \
      --max_sequence_identity=70.0 \
      &> $output_dir/$uniprot_id.log

