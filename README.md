# Multi-state modeling of protein structures using AlphaFold

## Building state-annotated HHsuite databases
All the required scripts and examples are in [build_state_annotated_databases](https://github.com/huhlim/alphafold-multistate/tree/main/build_state_annotated_databases)
1. **Getting activation state annotations for available experimental GPCR structures**  
The list of GPCR structures with activation state annotations: [GPCRdb](https://gpcrdb.org/structure/), [Activation state definition](https://docs.gpcrdb.org/structures.html#structure-descriptors)
2. **Preparing input files for building state-annotated HHsuite databases**  
The script takes a list of PDB IDs for a state, either active, inactive, or intermediate states. For example, [GPCR.Active](https://github.com/huhlim/alphafold-multistate/blob/main/build_state_annotated_databases/GPCR.Active), [GPCR.Inactive](https://github.com/huhlim/alphafold-multistate/blob/main/build_state_annotated_databases/GPCR.Inactive), and [GPCR.Intermediate](https://github.com/huhlim/alphafold-multistate/blob/main/build_state_annotated_databases/GPCR.Intermediate) are lists of active, inactive, and intermediate state GPCRs for this study, respectively. In addition, to select the _preferred_ chain among multiple chains of a PDB file, a list of PDB IDs with the _preferred_ chains is required. [Example](https://github.com/huhlim/alphafold-multistate/blob/main/build_state_annotated_databases/GPCR.chains)
3. **Running the script**  
The script is based on [the official guideline for building customized HHsuite databases](https://github.com/soedinglab/hh-suite/wiki#building-customized-databases).
To run the script, [HHsuite](https://github.com/soedinglab/hh-suite) and [UniClust30 database](http://gwdu111.gwdg.de/~compbiol/uniclust/2020_06/) are required. Also, one needs to modify [build_db.sh](https://github.com/huhlim/alphafold-multistate/blob/cc76e4cc08c121993a03599c62ae29b0cb38c106/build_state_annotated_databases/build_db.sh#L6) to adjust the path of the UniClust30 database.   
Example command:  
```
./build_db.sh GPCR.${state}
```
4. **Expected outputs**
The output of the scripts will be a set of HHsuite database files for a GPCR state.  
```
GPCR100.${state}_a3m.ff{data,index}
GPCR100.${state}_hhm.ff{data,index}
GPCR100.${state}_cs219.ff{data,index}
```
5. **Pre-built GPCR databases**  
State-annotated GPCR databases can be obtained from our repositories on [Zenodo](https://zenodo.org/record/5745217) or [Google Drive](https://drive.google.com/drive/folders/1JYp-6LkElUgpiWIB8GroSI_z9rlVmr5o?usp=sharing).

## GPCR structure prediction using AlphaFold

The structure prediction scripts rely on [AlphaFold](https://github.com/deepmind/alphafold). We slightly modified it to conduct ablation studies and to model GPCR structures in a specific activation state. Follow the setup procedure and download genetic databases and model parameters for AlphaFold. In contrast to the original AlphaFold, our scripts are based on a non-Docker version and run on top of an Anaconda environment for AlphaFold. To create an environment for running AlphaFold, one may refer to [an issue page](https://github.com/deepmind/alphafold/issues/24) of the AlphaFold repository or execute commands described in [our script](https://github.com/huhlim/alphafold-multistate/blob/main/structure_prediction/conda_create.sh). 

0. Prerequisite
- [AlphaFold package](https://github.com/deepmind/alphafold)
- Anaconda environment for AlphaFold
- Activation state annotated GPCR100 databases

1. Update [libconfig_alphafold.py](https://github.com/huhlim/alphafold-multistate/blob/main/structure_prediction/libconfig_alphafold.py)
One needs to update
- Paths for executables: jackhmmer, hhblits, hhsearch, kalign
- Paths for genetic databases: DOWNLOAD_DIR, {uniref90, mgnify, bfd, small_bfd, uniclust30, pdb70}_database_path, template_mmcif_dir, obsolete_pdbs_path
- Paths for activation state annotated GPCR100 databases: gpcr100_active_db_path, gpcr100_inactive_db_path

2. GPCR structure predictions
We assumed an activated Anaconda environment that has all required libraries/packages for running AlphaFold. 
- Modeling GPCRs in a specific activation state (this study)
```bash
./structure_prediction/run.py ${FASTA_FILE} --preset study --state active    # for modeling in active state
./structure_prediction/run.py ${FASTA_FILE} --preset study --state inactive  # for modeling in inactive state
```
In addition to this script, you may use a ColabFold-based script that is utilized for our [Colab notebook](https://colab.research.google.com/github/huhlim/alphafold-multistate/blob/main/AlphaFold_multistate.ipynb)
```bash
./structure_prediction/run_colabfold.py ${FASTA_FILE} --state active   # for modeling in active state
./structure_prediction/run_colabfold.py ${FASTA_FILE} --state inactive # for modeling in inactive state
```
Note that, this script is optimized for our Colab notebook environment and has not extensively tested. Also, running this script creates a directory, `gpcr100`, which contains symbolic links to the required database files.

- The original AlphaFold protocol
```bash
./structure_prediction/run.py ${FASTA_FILE} --preset original
```
- Other protocols for the ablation study as described in the paper
```bash
# running the original AlphaFold protocol but using activation state-annotated GPCR databases
./structure_prediction/run.py ${FASTA_FILE} --preset original --state active     # for modeling in active state
./structure_prediction/run.py ${FASTA_FILE} --preset original --state inactive   # for modeling in inactive state

# running AlphaFold using sequence and MSA-based features, without structure templates-based features
./structure_prediction/run.py ${FASTA_FILE} --preset no_templ

# running AlphaFold using sequence-based features only, without MSA and structure templates-based features
./structure_prediction/run.py ${FASTA_FILE} --preset seqonly

# running MODELLER
./structure_prediction/run.py [FASTA file] --preset tbm
```

- Sampling of intermediate conformations
```bash
./structure_prediction/interpolate.py --fasta_path=${FASTA_FILE} \
                                      --pdb_init=${INACTIVE_MODEL},${ACTIVE_MODEL} \
                                      --unk_pdb=True \
                                      --interpolate_region=${TM_RESIDUES}
```
Both active and inactive state models need to be generated first before providing them to the script. The option "interpolate_region" is optional, but it may improve structure comparison between states. An example input is as follows: "19-51,56-87,92-127,136-167,183-223,376-413,418-443".

## Running the protocol on Colab
A slightly modified protocol using ColabFold pipeline is implemented on [Colab](https://colab.research.google.com/github/huhlim/alphafold-multistate/blob/main/AlphaFold_multistate.ipynb). The main difference is the MSA generation step; the ColabFold-based protocol utilizes MMseqs2 for homologous sequence searches. 

## GPCR models in the active and inactive states
We have modeled non-olfactory human GPCRs in the active and inactive states using our multi-state modeling protocol. The models are available via [Zenodo](https://zenodo.org/record/5745217) or [Google Drive](https://drive.google.com/drive/folders/1JYp-6LkElUgpiWIB8GroSI_z9rlVmr5o?usp=sharing).

## References
[1] Heo, L. and Feig, M., Multi-State Modeling of G-protein Coupled Receptors at Experimental Accuracy, _Proteins_ (**2022**), doi:10.1002/prot.26382. [Link](https://onlinelibrary.wiley.com/doi/10.1002/prot.26382)  
[2] Jumper, J. _et al._, Highly accurate protein structure prediction with AlphaFold, _Nature_ (**2021**), 596, 583-589. [Link](https://www.nature.com/articles/s41586-021-03819-2)  
[3] Mirdita, M. _et al._, ColabFold - Making protein folding accessible to all, _Nature Methods_ (**2022**), 19, 679-682. [Link](https://www.nature.com/articles/s41592-022-01488-1)
