#!/usr/bin/env python

import numpy as np
from absl import logging

def apply_template_mask(feature_dict, residues):
    mask = np.zeros(feature_dict['template_aatype'].shape[1], dtype=bool)
    for residue_range in residues:
        if '-' in residue_range:
            x = residue_range.split("-")
            for r in range(int(x[0]), int(x[1])+1):
                mask[r-1] = True
        else:
            mask[int(residue_range)] = True
    mask = ~mask
    logging.info("Applying template mask %s", residues)
    #
    feature_dict['template_aatype'][:,mask] = 0.
    feature_dict['template_aatype'][:,mask,21] = 0.
    if 'template_all_atom_masks' in feature_dict:
        feature_dict['template_all_atom_masks'][:,mask] = 0.
    elif 'template_all_atom_mask' in feature_dict:
        feature_dict['template_all_atom_mask'][:,mask] = 0.
    feature_dict['template_all_atom_positions'][:,mask] = 0.
    return feature_dict

def remove_msa_for_template_aligned_regions(feature_dict):
    if 'template_all_atom_masks' in feature_dict:
        mask = feature_dict['template_all_atom_masks']
    elif 'template_all_atom_mask' in feature_dict:
        mask = feature_dict['template_all_atom_mask']
    mask = (mask.sum(axis=(0,2)) > 0)
    #
    logging.info("Removing MSA features for %d residues.", len(np.where(mask)[0]))
    #
    # need to check further for multimer_mode
    if 'deletion_matrix_int' in feature_dict:
        feature_dict['deletion_matrix_int'][:,mask] = 0
    else:
        feature_dict['deletion_matrix'][:,mask] = 0
    feature_dict['msa'][:,mask] = 21
    return feature_dict

def retrieve_custom_features(processed_feature_dict, feature_dict):
    for name in ['for_pdb_record']:
        if name in feature_dict:
            processed_feature_dict[name] = feature_dict[name]

