#!/usr/bin/env python

import os
import sys
import argparse
import subprocess as sp

import libconfig_af

def set_presets(arg):
    preset = arg.preset
    if preset.startswith("original"):
        arg.modeling_tool = 'alphafold'
        arg.use_templates = True
        arg.use_msa = True

    elif preset.startswith("study"):
        arg.modeling_tool = 'alphafold'
        arg.use_templates = True
        arg.use_msa = True
        arg.remove_msa = True

    elif preset == 'no_templ':
        arg.modeling_tool = 'alphafold'
        arg.use_templates = False
        arg.use_msa = True

    elif preset == 'seqonly':
        arg.modeling_tool = 'alphafold'
        arg.use_templates = False
        arg.use_msa = False

    elif preset == 'tbm':
        arg.modeling_tool = 'modeller'

def main():
    arg = argparse.ArgumentParser(prog='run_multi-state_modeling')
    arg.add_argument(dest='fa_fn', help='input FASTA file')
    arg.add_argument('-p', '--preset', dest='preset', default=None, \
            choices=['original', 'study', 'no_templ', 'seqonly', 'tbm'], \
            help='preset for run (default=none)')
    arg.add_argument('-o', '--output_dir', dest='output_dir', default='./', \
            help='output_dir (default=./)')
    arg.add_argument('-m', '--modeling', dest='modeling_tool', default='alphafold', \
            choices=['alphafold', 'modeller'], \
            help='structure building tool (default=alphafold)')
    arg.add_argument('-s', '--state', dest='state', default='none', \
            choices=['none', 'active', 'inactive'], \
            help='modeling state (default=none)')
    arg.add_argument('--seq_id', dest='seq_id_cutoff', default=-1.0, type=float, \
            help='sequence identity cutoff (default=none)')
    arg.add_argument('--no_msa', dest='use_msa', default=True, action='store_false', 
            help='whether to use MSA input features (default=True)')
    arg.add_argument('--no_templates', dest='use_templates', default=True, action='store_false', \
            help='whether to use Template-based input features (default=True)')
    arg.add_argument('--remove_msa', dest='remove_msa', default=False, action='store_true', \
            help='whether to remove MSA input features for template aligned region (default=False)')
    if len(sys.argv) == 1:
        arg.print_help()
        return
    #
    arg = arg.parse_args()
    set_presets(arg)

    if arg.modeling_tool == 'alphafold':
        EXEC = libconfig_af.exec_run_alphafold
    elif arg.modeling_tool == 'modeller':
        EXEC = libconfig_af.exec_run_tbm

    if arg.state == 'none':
        structure_db_path = libconfig_af.pdb70_database_path
    elif arg.state == 'active':
        structure_db_path = libconfig_af.gpcr100_active_db_path
    elif arg.state == 'inactive':
        structure_db_path = libconfig_af.gpcr100_inactive_db_path

    if arg.seq_id_cutoff > 100.:
        sys.exit("ERROR: sequence identity cutoff should be <= 100%\n")

    if arg.modeling_tool == 'modeller' and (not arg.use_templates):
        sys.exit("ERROR: not a compatible option, --modeling=%s --no_msa\n"%arg.modeling_tool)

    cmd = [EXEC]
    cmd.append('--fasta_path=%s'%arg.fa_fn)
    cmd.append('--output_dir=%s'%arg.output_dir)
    if arg.modeling_tool == 'alphafold':
        if arg.use_templates:
            cmd.append('--use_templates=true')
        else:
            cmd.append('--use_templates=false')
        if arg.use_msa:
            cmd.append('--use_msa=true')
        else:
            cmd.append('--use_msa=False')
        if arg.remove_msa:
            cmd.append("--remove_msa_for_template_aligned=true")
    if arg.use_templates:
        cmd.append("--pdb70_database_path=%s"%structure_db_path)
    if arg.seq_id_cutoff > 0.:
        cmd.append("--max_sequence_identity=%.1f"%arg.seq_id_cutoff)
    #print (cmd)
    sp.call(cmd)

if __name__ == '__main__':
    main()

