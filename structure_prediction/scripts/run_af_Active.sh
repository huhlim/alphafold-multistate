#!/bin/bash

EXEC=/home/huhlim/apps/AlphaFold/run.py
output_dir=af_Active

uniprot_id=$1

fa_fn=fa/$uniprot_id.fa
mkdir -p $output_dir

if [[ -e $output_dir/$uniprot_id/ranked_0.pdb ]]; then
    exit -1
fi
mkdir -p $output_dir/$uniprot_id
if [[ ! -e $output_dir/$uniprot_id/msas ]]; then
    ln -sf ../../af/$uniprot_id/msas $output_dir/$uniprot_id/msas
fi

database=/home/huhlim/work/af/multi_state/GPCRdb/GPCR100.Active

$EXEC --fasta_path=$fa_fn \
      --output_dir=$output_dir \
      --max_sequence_identity=70.0 \
      --pdb70_database_path=$database \
      &> $output_dir/$uniprot_id.log

