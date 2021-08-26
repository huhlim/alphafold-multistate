#!/usr/bin/env python

import sys

def main():
    gpcr_s = []
    with open(sys.argv[1]) as fp:
        for line in fp:
            gpcr_s.append(line.strip())
    #
    pdb_s = []
    with open(sys.argv[2]) as fp:
        for line in fp:
            if line.startswith("#"):
                pdb_s.append(line)
            else:
                x = line.strip().split()
                if x[0] in gpcr_s:
                    pdb_s.append(line)

    with open(sys.argv[2], 'wt') as fout:
        fout.writelines(pdb_s)
    #
    fas_s = []
    with open(sys.argv[3]) as fp:
        read = False
        for line in fp:
            if line.startswith(">"):
                x = line.strip().split()[0][1:]
                if x in gpcr_s:
                    read = True
                    fas_s.append(line)
                else:
                    read = False
            elif read:
                fas_s.append(line)

    with open(sys.argv[3], 'wt') as fout:
        fout.writelines(fas_s)


if __name__ == '__main__':
    main()

