#!/bin/bash

PDB_LIST=$1

N_PROC=32
UNICLUST30=""
if [[ $UNICLUST30 == "" ]]; then
    echo "NEED TO SET UNICLUST30 database path"
    echo "http://gwdu111.gwdg.de/~compbiol/uniclust/2020_06/"
    exit -1
fi

suffix=$(echo $PDB_LIST | awk -F. '{print $NF}')

# download mmCIF files
mkdir -p cif.$suffix
cd cif.$suffix
for pdb_id in $(cat $PDB_LIST);
do
    pdb_id=$(echo $pdb_id | awk '{print tolower($0)}')
    wget -q https://files.rcsb.org/download/$pdb_id.cif
done
cd ..

# write FASTA files
cif2fasta.py -i cif.$suffix -o GPCR.$suffix.fas -c $N_PROC -p pdb_filter.$suffix.dat
python select_GPCR_only.py GPCR.chains pdb_filter.$suffix.dat GPCR.$suffix.fas

# cluster
mkdir -p cluster.$suffix
mmseqs createdb GPCR.$suffix.fas cluster.$suffix/GPCR.$suffix
mmseqs cluster cluster.$suffix/GPCR.$suffix cluster.$suffix/GPCR100_clu.$suffix /scratch/clustering -c 1.0 --min-seq-id 1.0
mmseqs createtsv cluster.$suffix/GPCR.$suffix cluster.$suffix/GPCR.$suffix cluster.$suffix/GPCR100_clu.$suffix cluster.$suffix/GPCR100_clu.$suffix.tsv

pdbfilter.py GPCR.$suffix.fas cluster.$suffix/GPCR100_clu.$suffix.tsv pdb_filter.$suffix.dat GPCR100.$suffix.fas

db=GPCR100.$suffix
ffindex_from_fasta -s $db.fas.ff{data,index} $db.fas

# generate MSA files
mkdir -p msa.$suffix
python split_fasta.py $db.fas msa.$suffix
cd msa.$suffix
../run_hhblits.sh   $UNICLUST30
cd ..
mkdir -p fa.$suffix
mv msa.$suffix/*.fa fa.$suffix/

# build HHsuite database
cd msa.$suffix
ffindex_build -s ../${db}_msa.ff{data,index} .
cd ..

ffindex_apply ${db}_msa.ff{data,index} \
    -i ${db}_a3m.ffindex -d ${db}_a3m.ffdata -- addss.pl -v 0 stdin stdout
rm ${db}_msa.ff{data,index}

ffindex_apply ${db}_a3m.ff{data,index} \
    -i ${db}_hhm.ffindex -d ${db}_hhm.ffdata -- hhmake -i stdin -o stdout -v 0

cstranslate -f -x 0.3 -c 4 -I a3m -i ${db}_a3m -o ${db}_cs219

sort -k 3 -n -r ${db}_cs219.ffindex | cut -f 1 > sort.$suffix
ffindex_order sort.$suffix ${db}_hhm.ff{data,index} ${db}_hhm_ordered.ff{data,index}
mv ${db}_hhm_ordered.ffindex ${db}_hhm.ffindex
mv ${db}_hhm_ordered.ffdata ${db}_hhm.ffdata
ffindex_order sort.$suffix ${db}_a3m.ff{data,index} ${db}_a3m_ordered.ff{data,index}
mv ${db}_a3m_ordered.ffindex ${db}_a3m.ffindex
mv ${db}_a3m_ordered.ffdata ${db}_a3m.ffdata
rm sort.$suffix

# compress outputs
tar czf ${db}.tgz ${db}_a3m.ff{data,index} ${db}_hhm.ff{data,index} ${db}_cs219.ff{data,index}

