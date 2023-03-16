#!/bin/bash

CUDA=11.5

ENVNAME=af

conda create --name $ENVNAME -y -c conda-forge -c nvidia -c bioconda\
    python=3.8 \
    openmm=7.5.1 \
    cudatoolkit=$CUDA \
    cudnn=8.2 \
    libcusolver  \
    mdtraj \
    pdbfixer \
    hhsuite \
    kalign3 \
    hmmer \
    pip 

source activate $ENVNAME

pip install --upgrade pip 
pip3 install -r requirements.txt 
pip3 uninstall -yq jax jaxlib
pip3 install "jax[cuda]==0.3.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

openmm_patch=$(readlink -f openmm.patch)
cd $CONDA_PREFIX/lib/python3.8/site-packages
patch -p0 < $openmm_patch
cd -

if [[ ! -e alphafold/common/stereo_chemical_props.txt ]]; then
    wget -q -P alphafold/common/ \
        https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
fi

echo "####################################################"
echo "# update binary_paths in libconfig_af.py as follows"
echo "####################################################"
for binary in jackhmmer hhblits hhsearch kalign hmmsearch hmmbuild;
do
    binary_path=$(which $binary)
    echo "${binary}_binary_path = \"$binary_path\""
done

# IF you want to use ColabFold interface, uncomment the following and install ColabFold
# pip install --no-warn-conflicts -q "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold@v1.5.2"

