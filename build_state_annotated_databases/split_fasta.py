#!/usr/bin/env python

import sys

def main():
    fas_fn = sys.argv[1]
    out_dir = sys.argv[2]
    #
    fa_s = {}
    with open(fas_fn) as fp:
        for line in fp:
            if line.startswith(">"):
                name = line.strip().split()[0][1:]
                fa_s[name] = []
            fa_s[name].append(line)
    for name in fa_s:
        with open("%s/%s.fa"%(out_dir, name), 'wt') as fout:
            fout.writelines(fa_s[name])

if __name__ == '__main__':
    main()

