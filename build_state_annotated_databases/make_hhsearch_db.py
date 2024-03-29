#!/usr/bin/env python

import os
import sys

import io
import pathlib
from Bio.PDB import MMCIFParser

restype_3to1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

# script is from ColabFold
def make_hhsearch_db(template_dir: pathlib.Path, template_s):
    # clear the dir
    for fn in template_dir.glob("pdb70_*"):
        os.remove(fn)
    #
    a3m_ffdata = open(template_dir / "pdb70_a3m.ffdata", "w")
    a3m_ffindex = open(template_dir / "pdb70_a3m.ffindex", "w")
    cs219_ffdata = open(template_dir / "pdb70_cs219.ffdata", "w")
    cs219_ffindex = open(template_dir / "pdb70_cs219.ffindex", "w")
    #
    index = 1000000
    index_offset = 0
    cif_fn_s = template_dir.glob("*.cif")
    for cif_fn in cif_fn_s:
        name = cif_fn.stem.lower()
        print(name)
        parser = MMCIFParser(QUIET=True)
        with open(cif_fn) as fp:
            cif = io.StringIO(fp.read())
            structure = parser.get_structure("none", cif)
        model = list(structure.get_models())[0]
        #
        for chain in model:
            seq = []
            for residue in chain:
                # modified
                if residue.id[0] not in ["H_MSE", " "]:
                    continue
                if residue.id[2] != " ":
                    continue
                seq.append(restype_3to1.get(residue.resname, "X"))
            seq = "".join(seq)
            if len(seq) < 10:
                continue
            #
            id = f"{name}_{chain.id}"
            if template_s is not None and id not in template_s:
                continue
            #
            a3m_str = f">{id}\n{seq}\n\0"
            a3m_str_len = len(a3m_str)
            a3m_ffdata.write(a3m_str)
            cs219_ffdata.write("\n\0")
            a3m_ffindex.write(f"{index}\t{index_offset}\t{a3m_str_len}\n")
            cs219_ffindex.write(f"{index}\t{index_offset}\t{len(seq)}\n")
            index += 1
            index_offset += a3m_str_len


def read_templates(fn):
    template_s = []
    with open(fn) as fp:
        for line in fp:
            if not line.startswith("#"):
                pdb_id, chain_id = line.strip().split("_")
                template_s.append(f"{pdb_id.lower()}_{chain_id}")
    return template_s


def main():
    if len(sys.argv) == 1:
        sys.stderr.write(f"usage: {__file__} [TEMPLATE DIRECTORY] (TEMPLATE LIST: optional)\n")
        return
    #
    template_dir = pathlib.Path(sys.argv[1])
    if len(sys.argv) > 2:
        template_s = read_templates(sys.argv[2])
    else:
        template_s = None
    #
    make_hhsearch_db(template_dir, template_s)


if __name__ == "__main__":
    main()
