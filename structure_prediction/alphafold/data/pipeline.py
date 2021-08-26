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
from typing import Mapping, Optional, Sequence
from absl import logging
from alphafold.common import residue_constants
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import jackhmmer
import numpy as np

# Internal import (7716).

FeatureDict = Mapping[str, np.ndarray]


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


def make_msa_features(
    msas: Sequence[Sequence[str]],
    deletion_matrices: Sequence[parsers.DeletionMatrix]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

  num_res = len(msas[0][0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  return features

def split_chain(num_res_per_chain, sequence_features):
  PARAM_CHAIN_BREAK = 100
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
               hhsearch_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               uniclust30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               pdb70_database_path: str,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_small_bfd: bool,
               use_msa: bool,
               is_oligomer: bool,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000):
    """Constructs a feature dict for a given FASTA file."""
    self._use_small_bfd = use_small_bfd
    self._use_msa = use_msa
    self._is_oligomer = is_oligomer
    #
    if self._use_msa:
        self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=uniref90_database_path)
        if use_small_bfd:
          self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
              binary_path=jackhmmer_binary_path,
              database_path=small_bfd_database_path)
        else:
          self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
              binary_path=hhblits_binary_path,
              databases=[bfd_database_path, uniclust30_database_path])
        self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=mgnify_database_path)
        #
        self.hhsearch_pdb70_runner = hhsearch.HHSearch(
            binary_path=hhsearch_binary_path,
            databases=[pdb70_database_path])

    elif template_featurizer is not None:
        self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=uniref90_database_path)
        #
        self.hhsearch_pdb70_runner = hhsearch.HHSearch(
            binary_path=hhsearch_binary_path,
            databases=[pdb70_database_path])
    self.template_featurizer = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits

  def process(self, input_fasta_path: str, input_msa_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      if not self._is_oligomer:
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
      num_res_per_chain = [num_res]

    if self._use_msa and (input_msa_path is None):
      # uniref90
      uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
      if os.path.exists(uniref90_out_path):
        with open(uniref90_out_path) as f:
          jackhmmer_uniref90_result_sto = f.read()
      else:
        jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(
            input_fasta_path)[0]
        jackhmmer_uniref90_result_sto = jackhmmer_uniref90_result['sto']
        with open(uniref90_out_path, 'w') as f:
          f.write(jackhmmer_uniref90_result_sto)

      uniref90_msa, uniref90_deletion_matrix, _ = parsers.parse_stockholm(
          jackhmmer_uniref90_result_sto)
      uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
          jackhmmer_uniref90_result_sto, max_sequences=self.uniref_max_hits)

      # mgnify
      mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
      if os.path.exists(mgnify_out_path):
        with open(mgnify_out_path) as f:
          jackhmmer_mgnify_result_sto = f.read()
      else:
        jackhmmer_mgnify_result = self.jackhmmer_mgnify_runner.query(
            input_fasta_path)[0]
        jackhmmer_mgnify_result_sto = jackhmmer_mgnify_result['sto']
        with open(mgnify_out_path, 'w') as f:
          f.write(jackhmmer_mgnify_result_sto)

      mgnify_msa, mgnify_deletion_matrix, _ = parsers.parse_stockholm(
          jackhmmer_mgnify_result_sto)
      mgnify_msa = mgnify_msa[:self.mgnify_max_hits]
      mgnify_deletion_matrix = mgnify_deletion_matrix[:self.mgnify_max_hits]

      # pdb70
      hhsearch_result = self.hhsearch_pdb70_runner.query(uniref90_msa_as_a3m)
      hhsearch_hits = parsers.parse_hhr(hhsearch_result)

      # bfd
      if self._use_small_bfd:
        bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.a3m')
        if os.path.exists(bfd_out_path):
          with open(bfd_out_path) as f:
            jackhmmer_small_bfd_result_sto = f.read()
        else:
            jackhmmer_small_bfd_result = self.jackhmmer_small_bfd_runner.query(
                input_fasta_path)[0]
            jackhmmer_small_bfd_result_sto = jackhmmer_small_bfd_result['sto']
            with open(bfd_out_path, 'w') as f:
              f.write(jackhmmer_small_bfd_result_sto)
        bfd_msa, bfd_deletion_matrix, _ = parsers.parse_stockholm(
            jackhmmer_small_bfd_result_sto)
      else:
        bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
        if os.path.exists(bfd_out_path):
            with open(bfd_out_path) as f:
                hhblits_bfd_uniclust_result_a3m = f.read()
        else:
            hhblits_bfd_uniclust_result = self.hhblits_bfd_uniclust_runner.query(
                input_fasta_path)
            hhblits_bfd_uniclust_result_a3m = hhblits_bfd_uniclust_result['a3m']
            with open(bfd_out_path, 'w') as f:
              f.write(hhblits_bfd_uniclust_result_a3m)
        bfd_msa, bfd_deletion_matrix = parsers.parse_a3m(
            hhblits_bfd_uniclust_result_a3m)

    elif self.template_featurizer is not None:
      if input_msa_path is None:
        # uniref90
        uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
        if os.path.exists(uniref90_out_path):
          with open(uniref90_out_path) as f:
            jackhmmer_uniref90_result_sto = f.read()
        else:
          jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(
              input_fasta_path)[0]
          jackhmmer_uniref90_result_sto = jackhmmer_uniref90_result['sto']
          with open(uniref90_out_path, 'w') as f:
            f.write(jackhmmer_uniref90_result_sto)
        #
        uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
            jackhmmer_uniref90_result_sto, max_sequences=self.uniref_max_hits)
        input_msa_a3m = uniref90_msa_as_a3m
      else:
        with open(input_msa_path) as f:
          input_msa_a3m = f.read()
      #
      hhsearch_result = self.hhsearch_pdb70_runner.query(input_msa_a3m)
      hhsearch_hits = parsers.parse_hhr(hhsearch_result)

    if self.template_featurizer is not None:
      templates_result = self.template_featurizer.get_templates(
          query_sequence=input_sequence,
          query_pdb_code=None,
          query_release_date=None,
          hits=hhsearch_hits)
      templates_features = templates_result.features
    else:
      num_templates_ = 0
      #
      templates_features = {}
      templates_features['template_aatype'] = \
              np.zeros([num_templates_, num_res, 22], np.float32)
      templates_features['template_all_atom_masks'] = \
              np.zeros([num_templates_, num_res, 37], np.float32)
      templates_features['template_all_atom_positions'] = \
              np.zeros([num_templates_, num_res, 37, 3], np.float32)
      templates_features['template_domain_names'] = \
              np.zeros([num_templates_], np.float32)
      templates_features['template_sum_probs'] = \
              np.zeros([num_templates_], np.float32)


    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    if self._use_msa:
      if input_msa_path is None:
        msa_features = make_msa_features(
            msas=(uniref90_msa, bfd_msa, mgnify_msa),
            deletion_matrices=(uniref90_deletion_matrix,
                               bfd_deletion_matrix,
                               mgnify_deletion_matrix))
      else:
        with open(input_msa_path) as f:
          a3m = f.read()
        input_msa, input_deletion_matrix = parsers.parse_a3m(a3m)
        msa_features = make_msa_features(msas=(input_msa,), \
                deletion_matrices=(input_deletion_matrix,))
    else:
      with open(input_fasta_path) as f:
        a3m = f.read()
      seq_msa, seq_deletion_matrix = parsers.parse_a3m(a3m)
      msa_features = make_msa_features(msas=(seq_msa,), \
              deletion_matrices=(seq_deletion_matrix,))

    if self._is_oligomer and len(num_res_per_chain) > 1:
      split_chain(num_res_per_chain, sequence_features)

    if self._use_msa:
      if input_msa_path is None:
        logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
        logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
        logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
      else:
        logging.info('Input MSA size: %d sequences.', len(input_msa))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    logging.info('Total number of templates (NB: this can include bad '
                 'templates and is later filtered to top 4): %d.',
                 templates_features['template_domain_names'].shape[0])
    return {**sequence_features, **msa_features, **templates_features}
