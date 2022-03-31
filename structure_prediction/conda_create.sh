#!/bin/bash

CUDA=11.0

ENVNAME=af

conda create --name $ENVNAME -y -c conda-forge -c nvidia \
    python=3.8.5 \
    openmm=7.5.1 \
    mdtraj \
    cudatoolkit=$CUDA.3 \
    pdbfixer \
    pip \
    absl-py=0.13.0  \
    cudnn=8.0.4 \
    libcusolver 

source activate $ENVNAME

pip install --upgrade pip 
pip3 install -r requirements.txt 
pip3 install --upgrade "jax[cuda${CUDA/./}]" jaxlib==0.1.69+cuda${CUDA/./} -f \
      https://storage.googleapis.com/jax-releases/jax_releases.html

openmm_patch=$(readlink -f docker/openmm.patch)
cd $CONDA_PREFIX/lib/python3.8/site-packages
patch -p0 < $openmm_patch
cd -

if [[ ! -e alphafold/common/stereo_chemical_props.txt ]]; then
    wget -q -P alphafold/common/ \
        https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
fi

