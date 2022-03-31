#!/usr/bin/env python

from Bio import pairwise2
from Bio.Align import substitution_matrices
from typing import Sequence

class BioAlign:
    def __init__(self, matrix_name="BLOSUM62"):
        self.matrix = substitution_matrices.load(matrix_name)
    def align(self, sequences: Sequence[str]) -> str:
        aligned = pairwise2.align.globaldx(sequences[0], sequences[1], self.matrix)[0]
        a3m = []
        a3m.append(">seqA")
        a3m.append(aligned.seqA)
        a3m.append(">seqB")
        a3m.append(aligned.seqB)
        return '\n'.join(a3m)

if __name__ == '__main__':
    main()

