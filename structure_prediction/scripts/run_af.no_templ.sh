#!/bin/bash

EXEC=/home/huhlim/apps/AlphaFold/run.py
output_dir=af.no_templ

uniprot_id=$1

fa_fn=fa/$uniprot_id.fa
mkdir -p $output_dir

if [[ -e $output_dir/$uniprot_id/ranked_4.pdb ]]; then
    exit -1
fi
mkdir -p $output_dir/$uniprot_id
if [[ ! -e $output_dir/$uniprot_id/msas ]]; then
    ln -sf ../../af/$uniprot_id/msas $output_dir/$uniprot_id/msas
fi

LOCK=$output_dir/$uniprot_id/LOCK
if [[ -e $LOCK ]]; then
    exit -1
fi
touch $LOCK

$EXEC --fasta_path=$fa_fn \
      --output_dir=$output_dir \
      --use_templates=false \
      &> $output_dir/$uniprot_id.log

rm $LOCK
