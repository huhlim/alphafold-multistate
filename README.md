# Multiple-state modeling of protein structures using AlphaFold

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
You can obtain state-annotated GPCR databases at [here](https://zenodo.org/deposit/5156185)
