# Multi-state modeling of protein structures using AlphaFold

## Building state-annotated HHsuite databases
All the required scripts and examples are in [build_state_annotated_databases](https://github.com/huhlim/alphafold-multistate/tree/main/build_state_annotated_databases)
1. **Getting activation state annotations for available experimental GPCR structures**  
The list of GPCR structures with activation state annotations: [GPCRdb](https://gpcrdb.org/structure/), [Activation state definition](https://docs.gpcrdb.org/structures.html#structure-descriptors)
2. **Preparing input files for building state-annotated HHsuite databases**  
The script takes a list of PDB IDs for a state, either active, inactive, or intermediate states. For example, [GPCR.Active](https://github.com/huhlim/alphafold-multistate/blob/main/build_state_annotated_databases/GPCR.Active), [GPCR.Inactive](https://github.com/huhlim/alphafold-multistate/blob/main/build_state_annotated_databases/GPCR.Inactive), and [GPCR.Intermediate](https://github.com/huhlim/alphafold-multistate/blob/main/build_state_annotated_databases/GPCR.Intermediate) are lists of active, inactive, and intermediate state GPCRs for this study, respectively. In addition, to select the _preferred_ chain among multiple chains of a PDB file, a list of PDB IDs with the _preferred_ chains is required. [Example](https://github.com/huhlim/alphafold-multistate/blob/main/build_state_annotated_databases/GPCR.chains)
3. **Running the script**  
The script is based on [the official guideline for building customized HHsuite databases](https://github.com/soedinglab/hh-suite/wiki#building-customized-databases).
To run the script, [HHsuite](https://github.com/soedinglab/hh-suite) and [UniClust30 database](http://gwdu111.gwdg.de/~compbiol/uniclust/2020_06/) are required. Also, you need to modify [build_db.sh](https://github.com/huhlim/alphafold-multistate/blob/cc76e4cc08c121993a03599c62ae29b0cb38c106/build_state_annotated_databases/build_db.sh#L6) for the path of the UniClust30 database.   
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
You can obtain state-annotated GPCR databases at [here](https://zenodo.org/record/5156185)

## GPCR structure prediction using AlphaFold

The structure prediction scripts rely on [AlphaFold](https://github.com/deepmind/alphafold). We slightly modified to conduct ablation studies and model GPCR structures in a specific activation state. You should follow the setup procedure and download genetic databases and model parameters for AlphaFold. In contrast to the original AlphaFold, our scripts are based on non-Docker version and running on top of an Anaconda environment for AlphaFold. To create an environment for running AlphaFold, you may refer to [an issue page](https://github.com/deepmind/alphafold/issues/24) of the AlphaFold repository or execute commands described in [our script](https://github.com/huhlim/alphafold-multistate/blob/main/structure_prediction/conda_create.sh). 

0. Prerequisite
- [AlphaFold package](https://github.com/deepmind/alphafold)
- Anaconda environment for AlphaFold
- Activation state annotated GPCR100 databases

1. Update [libconfig_alphafold.py](https://github.com/huhlim/alphafold-multistate/blob/main/structure_prediction/libconfig_alphafold.py)
You need to update
- Paths for executables: jackhmmer, hhblits, hhsearch, kalign
- Paths for genetic databases: DOWNLOAD_DIR, {uniref90, mgnify, bfd, small_bfd, uniclust30, pdb70}_database_path, template_mmcif_dir, obsolete_pdbs_path
- Paths for activation state annotated GPCR100 databases: gpcr100_active_db_path, gpcr100_inactive_db_path

2. GPCR structure predictions
We assumed that you activated an Anaconda environment that has all required library/package for running AlphaFold. 
- Modeling GPCRs in a specific activation state (this study)
```bash
./structure_prediction/run.py ${FASTA_FILE} --preset study --state active    # for modeling in active state
./structure_prediction/run.py ${FASTA_FILE} --preset study --state inactive  # for modeling in inactive state
```
- The original AlphaFold protocol
```bash
./structure_prediction/run.py ${FASTA_FILE} --preset original
```
- Other protocols for the ablation study described in the paper
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

