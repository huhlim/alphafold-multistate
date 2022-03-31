# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the input features for the AlphaFold model."""

import os
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from alphafold.common import residue_constants
from alphafold.data import msa_identifiers
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.data.tools import jackhmmer
import numpy as np

# Internal import (7716).

FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]


def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features


def make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  species_ids = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa.sequences):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(msa.deletion_matrix[sequence_index])
      identifiers = msa_identifiers.get_identifiers(
          msa.descriptions[sequence_index])
      species_ids.append(identifiers.species_id.encode('utf-8'))

  num_res = len(msas[0].sequences[0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
  return features

def run_msa_tool(msa_runner, input_fasta_path: str, msa_out_path: str,
                 msa_format: str, use_precomputed_msas: bool,
                 max_sto_sequences: Optional[int] = None
                 ) -> Mapping[str, Any]:
  """Runs an MSA tool, checking if output already exists first."""
  if not use_precomputed_msas or not os.path.exists(msa_out_path):
    if msa_format == 'sto' and max_sto_sequences is not None:
      result = msa_runner.query(input_fasta_path, max_sto_sequences)[0]  # pytype: disable=wrong-arg-count
    else:
      result = msa_runner.query(input_fasta_path)[0]
    with open(msa_out_path, 'w') as f:
      f.write(result[msa_format])
  else:
    logging.warning('Reading MSA from file %s', msa_out_path)
    if msa_format == 'sto' and max_sto_sequences is not None:
      precomputed_msa = parsers.truncate_stockholm_msa(
          msa_out_path, max_sto_sequences)
      result = {'sto': precomputed_msa}
    else:
      with open(msa_out_path, 'r') as f:
        result = {msa_format: f.read()}
  return result

def append_sequence_info(sequence_features, input_seqs):
    for_pdb_record = {}
    for_pdb_record['residue_number'] = []
    for_pdb_record['chain_index'] = []
    for chain_index, input_seq in enumerate(input_seqs):
        for residue_index in range(len(input_seq)):
            for_pdb_record['residue_number'].append(residue_index+1)
            for_pdb_record['chain_index'].append(chain_index)
    for key in for_pdb_record:
        for_pdb_record[key] = np.array(for_pdb_record[key])
    sequence_features['for_pdb_record'] = for_pdb_record

def split_chain(num_res_per_chain, sequence_features):
  PARAM_CHAIN_BREAK = 200
  #
  L_prev = 0
  for L in num_res_per_chain[:-1]:
    sequence_features['residue_index'][L_prev+L:] += PARAM_CHAIN_BREAK
    L_prev += L

class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               uniclust30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               template_searcher: TemplateSearcher,
               template_featurizer: templates.TemplateHitFeaturizer,
               template_conformation: templates.ConformationInfoExactractor,
               use_small_bfd: bool,
               use_msa: bool = True,
               n_cpu: int = 8,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000,
               use_precomputed_msas: bool = False,
               is_multimer: bool = False,
               ):
    """Initializes the data pipeline."""
    self._use_small_bfd = use_small_bfd
    self._use_msa = use_msa
    self._use_template = (template_featurizer is not None)
    self._is_multimer = is_multimer
    #
    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path, n_cpu=n_cpu)
    if use_small_bfd:
      self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=small_bfd_database_path, n_cpu=n_cpu)
    else:
      self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniclust30_database_path], n_cpu=n_cpu)
    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path, n_cpu=n_cpu)
    self.template_searcher = template_searcher
    self.template_featurizer = template_featurizer
    self.template_conformation = template_conformation
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits
    self.use_precomputed_msas = use_precomputed_msas

  def process(self, 
          input_fasta_path: str, 
          input_msa_path: str,
          input_pdb_path: str,
          msa_output_dir: str) -> FeatureDict:

    # sequence-based features
    input_sequence, input_seqs, sequence_features = \
            self._process_input_fasta(input_fasta_path)

    # MSA-based features
    if self._use_msa:
      if input_msa_path:
        msa_features = \
            self._process_input_msa(input_fasta_path, input_msa_path, msa_output_dir)
        msa_for_templates = None
      else:
        msa_features, msa_for_templates = \
            self._process_search_msa(input_fasta_path, msa_output_dir)
    else:
      msa_features = self._process_null_msa(input_sequence, msa_output_dir)
      msa_for_templates = None

    # template structure-based features
    if self._use_template:
      if msa_for_templates is None:
        msa_for_templates = \
          self._process_search_msa_for_templates(input_fasta_path, msa_output_dir)
      templates_features = \
          self._process_search_templates(input_sequence, msa_for_templates, msa_output_dir)
    else:
      templates_features = self._process_null_templates(input_sequence)

    # combine template- and conformation-based features
    if self.template_conformation is not None and input_pdb_path is not None:
        conf_features = self.template_conformation.extract(input_sequence, input_pdb_path)
        templates_features = templates.combine_template_features(templates_features, conf_features)
    
    return {**sequence_features, **msa_features, **templates_features}

  def _process_input_fasta(self, input_fasta_path):
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      if not self._is_multimer:
        raise ValueError(
            f'More than one input sequence found in {input_fasta_path}.')
      else:
        input_sequence = ''.join(input_seqs)
        input_description = ''.join(input_descs)
        num_res = len(input_sequence)
        num_res_per_chain = [len(seq) for seq in input_seqs]
    else:
      input_sequence = input_seqs[0]
      input_description = input_descs[0]
      num_res = len(input_sequence)

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)
    append_sequence_info(sequence_features, input_seqs)

    if self._is_multimer:
      split_chain(num_res_per_chain, sequence_features)

    return input_sequence, input_seqs, sequence_features
  
  def _process_search_msa(self, input_fasta_path, msa_output_dir):
    uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    jackhmmer_uniref90_result = run_msa_tool(
        msa_runner=self.jackhmmer_uniref90_runner,
        input_fasta_path=input_fasta_path,
        msa_out_path=uniref90_out_path,
        msa_format='sto',
        use_precomputed_msas=self.use_precomputed_msas,
        max_sto_sequences=self.uniref_max_hits)
    mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
    jackhmmer_mgnify_result = run_msa_tool(
        msa_runner=self.jackhmmer_mgnify_runner,
        input_fasta_path=input_fasta_path,
        msa_out_path=mgnify_out_path,
        msa_format='sto',
        use_precomputed_msas=self.use_precomputed_msas,
        max_sto_sequences=self.mgnify_max_hits)

    msa_for_templates = jackhmmer_uniref90_result['sto']
    msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
    msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
        msa_for_templates)

    uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
    mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])

    if self._use_small_bfd:
      bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.sto')
      jackhmmer_small_bfd_result = run_msa_tool(
          msa_runner=self.jackhmmer_small_bfd_runner,
          input_fasta_path=input_fasta_path,
          msa_out_path=bfd_out_path,
          msa_format='sto',
          use_precomputed_msas=self.use_precomputed_msas)
      bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])
    else:
      bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
      hhblits_bfd_uniclust_result = run_msa_tool(
          msa_runner=self.hhblits_bfd_uniclust_runner,
          input_fasta_path=input_fasta_path,
          msa_out_path=bfd_out_path,
          msa_format='a3m',
          use_precomputed_msas=self.use_precomputed_msas)
      bfd_msa = parsers.parse_a3m(hhblits_bfd_uniclust_result['a3m'])

    msa_features = make_msa_features((uniref90_msa, bfd_msa, mgnify_msa))

    logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
    logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
    logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    return msa_features, msa_for_templates

  def _process_search_msa_for_templates(self, input_fasta_path, msa_output_dir, chain_id=None):
    if chain_id is None:
      uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    else:
      input_fasta_chain_path = os.path.join(msa_output_dir, "input_seq_%d.fa"%chain_id)
      with open(input_fasta_chain_path, 'wt') as fout:
        fout.write(">Query_%d"%chain_id)
        fout.write(input_fasta_path)
      input_fasta_path = input_fasta_chain_path

      uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits_%d.sto'%chain_id)
    jackhmmer_uniref90_result = run_msa_tool(
        self.jackhmmer_uniref90_runner, input_fasta_path, uniref90_out_path,
        'sto', self.use_precomputed_msas)

    msa_for_templates = jackhmmer_uniref90_result['sto']
    #msa_for_templates = parsers.truncate_stockholm_msa(
    #    msa_for_templates, max_sequences=self.uniref_max_hits)
    msa_for_templates = parsers.deduplicate_stockholm_msa(
        msa_for_templates)
    msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
        msa_for_templates)
    return msa_for_templates

  def _process_input_msa(self, input_fasta_path, input_msa_path, msa_output_dir):
    with open(input_msa_path) as fp:
      a3m = fp.read()
    out_path = os.path.join(msa_output_dir, 'input_msa.a3m')
    with open(out_path, 'wt') as fout:
      fout.write(a3m)
    out_msa = parsers.parse_a3m(a3m)
    msa_features = make_msa_features((out_msa,))
    #
    logging.info('Input MSA size: %d sequences.', len(out_msa))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    return msa_features

  def _process_null_msa(self, input_sequence, msa_output_dir):
    out_path = os.path.join(msa_output_dir, 'input_seq.a3m')
    with open(out_path, 'wt') as fout:
      fout.write(">Query\n")
      fout.write(input_sequence)
    with open(out_path) as fp:
      a3m = fp.read()
    out_msa = parsers.parse_a3m(a3m)
    msa_features = make_msa_features((out_msa,))
    #
    logging.info('Input MSA size: %d sequences.', len(out_msa))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    return msa_features

  def _process_search_templates(self, input_sequence, msa_for_templates, msa_output_dir):
    # search templates
    if self.template_searcher.input_format == 'sto':    # HMMer
      pdb_templates_result = self.template_searcher.query(msa_for_templates)
    elif self.template_searcher.input_format == 'a3m':  # HHsearch
      uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
      pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
    else:
      raise ValueError('Unrecognized template input format: '
                       f'{self.template_searcher.input_format}')

    pdb_hits_out_path = os.path.join(
        msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}')
    with open(pdb_hits_out_path, 'w') as f:
      f.write(pdb_templates_result)

    # post-processing
    pdb_template_hits = self.template_searcher.get_template_hits(
        output_string=pdb_templates_result, input_sequence=input_sequence)
    templates_result = self.template_featurizer.get_templates(
        query_sequence=input_sequence,
        hits=pdb_template_hits)
    logging.info('Total number of templates (NB: this can include bad '
                 'templates and is later filtered to top 4): %d.',
                 templates_result.features['template_domain_names'].shape[0])
    return templates_result.features

  def _process_null_templates(self, input_sequence):
    return templates.make_null_template_features(input_sequence)
