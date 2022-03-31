#!/usr/bin/env python

import json
import os
import pathlib
import pickle
import random
import sys
import time
from typing import Dict

from absl import app
from absl import flags
from absl import logging
import numpy as np

from alphafold.common import protein, residue_constants
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.data.tools import hhsearch

import libmodeller as modeller

import libconfig_af

flags.DEFINE_string('fasta_path', None, 'Paths to FASTA files, each containing '
                  'one sequence. Paths should be separated by commas. '
                  'All FASTA paths must have a unique basename as the '
                  'basename is used to name the output directories for '
                  'each prediction.')
flags.DEFINE_string('output_dir', os.getcwd(), 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_list('model_names', libconfig_af.model_names, 'Names of models to use.')

flags.DEFINE_string('data_dir', libconfig_af.data_dir, 'Path to directory of supporting data.')
flags.DEFINE_integer("n_templates", 1, "number of templates")
flags.DEFINE_integer("n_model", 8, "number of models to try")
flags.DEFINE_string('jackhmmer_binary_path', libconfig_af.jackhmmer_binary_path,
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', libconfig_af.hhblits_binary_path, 
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', libconfig_af.hhsearch_binary_path,
                    'Path to the HHsearch executable.')
flags.DEFINE_string('kalign_binary_path', libconfig_af.kalign_binary_path,
                    'Path to the Kalign executable.')

flags.DEFINE_string('uniref90_database_path', libconfig_af.uniref90_database_path, 
        'Path to the Uniref90 database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', libconfig_af.mgnify_database_path, 
        'Path to the MGnify database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', libconfig_af.bfd_database_path, 
        'Path to the BFD database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', libconfig_af.small_bfd_database_path,
        'Path to the small version of BFD used with the "reduced_dbs" preset.')
flags.DEFINE_string('uniclust30_database_path', libconfig_af.uniclust30_database_path, 
        'Path to the Uniclust30 database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', libconfig_af.uniprot_database_path, 
        'Path to the Uniprot database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', libconfig_af.pdb70_database_path, 
        'Path to the PDB70 database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', libconfig_af.pdb_seqres_database_path, 
        'Path to the PDB seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', libconfig_af.template_mmcif_dir,
                    'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', libconfig_af.max_template_date, 
                    'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', libconfig_af.obsolete_pdbs_path,
                    'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_float("max_sequence_identity", -1., "Maximum sequence identity for template prefilter")
flags.DEFINE_string("msa_path", None, "User input MSA")
flags.DEFINE_string("custom_templates", None, "User input templates")
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
FLAGS = flags.FLAGS

def write_template_pdb(seq, atom_positions, atom_mask):
    restypes = residue_constants.restypes + ['X']
    aatype = []
    for aa in seq:
        if aa == '-':
            aatype.append(restypes.index("X"))
        else:
            aatype.append(restypes.index(aa))
    aatype = np.array(aatype, dtype=np.int32)

    prot = protein.Protein(
            aatype=aatype, \
            atom_positions=atom_positions, \
            atom_mask=atom_mask, \
            residue_index=np.arange(atom_positions.shape[0]).astype(int)+1, \
            chain_index=np.zeros_like(aatype, dtype=int), \
            b_factors=np.zeros_like(atom_mask))

    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], 'UNK')
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors

    pdb_lines.append('MODEL     1')
    atom_index = 1
    chain_index = -1
    residue_index_prev = residue_index[0]-100
    #
    res_beg = [None, None]
    res_end = [None, None]
    # Add all atom sites.
    for i in range(aatype.shape[0]):
      if residue_index[i] - residue_index_prev > 1:   # chain break
        chain_index += 1
        res_num = 1
        if chain_index > 0:
          pdb_lines.append("TER")
      elif residue_index[i] != residue_index_prev:
        res_num += 1
      residue_index_prev = residue_index[i]
      if aatype[i] > residue_constants.restype_num:
          continue
      chain_id = protein.PDB_CHAIN_IDS[chain_index]
      res_name_3 = res_1to3(aatype[i])
      #
      for atom_name, pos, mask, b_factor in zip(
          atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
        if mask < 0.5:
          continue

        record_type = 'ATOM'
        name = atom_name if len(atom_name) == 4 else f' {atom_name}'
        alt_loc = ''
        insertion_code = ''
        occupancy = 1.00
        element = atom_name[0]  # Protein supports only C, N, O, S, this works.
        charge = ''
        # PDB is a columnar format, every space matters here!
        atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                     f'{res_name_3:>3} {chain_id:>1}'
                     f'{res_num:>4}{insertion_code:>1}   '
                     f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                     f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                     f'{element:>2}{charge:>2}')
        pdb_lines.append(atom_line)
        atom_index += 1
        #
        if res_beg[0] is None:
            res_beg[0] = chain_id
            res_beg[1] = res_num
        res_end[0] = chain_id
        res_end[1] = res_num

    # Close the chain.
    pdb_lines.append("TER")
    pdb_lines.append('ENDMDL')

    pdb_lines.append('END')
    pdb_lines.append('')
    return '\n'.join(pdb_lines), res_beg, res_end

def predict_structure(
    fasta_path: str,
    fasta_name: str,
    msa_path: str,
    output_dir_base: str,
    data_pipeline: pipeline.DataPipeline,
    n_model: int, 
    random_seed: int):
  """Predicts structure using AlphaFold for the given sequence."""
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Get features.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  if os.path.exists(features_output_path):
      with open(features_output_path, 'rb') as f:
        feature_dict = pickle.load(f)
  else:
      feature_dict = data_pipeline.process(
          input_fasta_path=fasta_path,
          input_msa_path=msa_path,
          input_pdb_path=None,
          msa_output_dir=msa_output_dir)

      # Write out features as a pickled dictionary.
      with open(features_output_path, 'wb') as f:
        pickle.dump(feature_dict, f, protocol=4)
  #
  pir_fn = os.path.join(output_dir, '%s.pir'%fasta_name)
  with open(pir_fn, 'wt') as fout:
      fout.write(">P1;%s\n"%fasta_name)
      fout.write("sequence:%s:1:A:%d:A::::\n"%(fasta_name, feature_dict['seq_length'][0]))
      fout.write('%s*\n'%feature_dict['sequence'][0].decode())
      #
      n_templ = len(feature_dict['template_domain_names'])
      for i in range(n_templ):
          name = feature_dict['template_domain_names'][i].decode()
          atom_positions = feature_dict['template_all_atom_positions'][i]
          atom_mask = feature_dict['template_all_atom_masks'][i]
          residue_present = (atom_mask.sum(axis=-1) > 0)
          seq = []
          for present,aa in zip(residue_present, feature_dict['template_sequence'][i].decode()):
              if present:
                  seq.append(aa)
              else:
                  seq.append("-")
          seq = ''.join(seq)
          #
          templ, res_beg, res_end = write_template_pdb(seq, atom_positions, atom_mask)
          templ_pdb_fn = os.path.join(output_dir, '%s.pdb'%name)
          with open(templ_pdb_fn, 'wt') as f:
              f.write(templ)
          #
          fout.write(">P1;%s\n"%(name))
          fout.write("structureX:%s:%d:%s:%d:%s::::\n"%(name, \
                  res_beg[1],res_beg[0], res_end[1],res_end[0]))
          fout.write("%s*\n"%seq)
    #
  os.chdir(output_dir)
  #
  n_proc = min(n_model, libconfig_af.N_PROC)
  pir_fn = pir_fn.split("/")[-1]
  try:
      modeller.build_model(pir_fn, '.', n_model, n_proc)
  except:
      pir_fn = modeller.update_pir(pir_fn)
      modeller.build_model(pir_fn, '.', n_model, n_proc)
      os.remove(pir_fn)
  #
  for i in range(n_proc):
      fn = "run_tbm.slave%d"%i
      if os.path.exists(fn):
          os.remove(fn)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.custom_templates is not None:
    FLAGS.template_mmcif_dir = FLAGS.custom_templates
    FLAGS.pdb70_database_path = "%s/pdb70"%FLAGS.custom_templates

  # Check for duplicate FASTA file names.
  fasta_name = pathlib.Path(FLAGS.fasta_path).stem

  template_searcher = hhsearch.HHSearch(
      binary_path=FLAGS.hhsearch_binary_path,
      databases=[FLAGS.pdb70_database_path])
  template_featurizer = templates.HhsearchHitFeaturizer(
      mmcif_dir=FLAGS.template_mmcif_dir,
      max_template_date=FLAGS.max_template_date,
      max_hits=FLAGS.n_templates,
      kalign_binary_path=FLAGS.kalign_binary_path,
      release_dates_path=None,
      obsolete_pdbs_path=FLAGS.obsolete_pdbs_path,
      max_sequence_identity=FLAGS.max_sequence_identity)

  # Input Conformation
  conformation_info_extractor = None
  data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniclust30_database_path=FLAGS.uniclust30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      template_conformation=conformation_info_extractor,
      use_small_bfd=False, 
      use_msa=False,
      use_precomputed_msas=True,
      is_multimer=False,
      n_cpu=libconfig_af.N_PROC)

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize)
  logging.info('Using random seed %d for the data pipeline', random_seed)

  predict_structure(
    fasta_path=FLAGS.fasta_path,
    fasta_name=fasta_name,
    msa_path=FLAGS.msa_path,
    output_dir_base=FLAGS.output_dir,
    data_pipeline=data_pipeline,
    n_model=FLAGS.n_model, 
    random_seed=random_seed)

if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_path',
  ])

  app.run(main)
