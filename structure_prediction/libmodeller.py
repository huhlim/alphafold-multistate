#!/usr/bin/env python

import os
import sys
import importlib
import subprocess

from modeller import *
from modeller.automodel import *

def parse_pir(pir_fn):
    query = []
    templates = []
    with open(pir_fn, 'r') as fp:
        for line in fp:
            if line.startswith(">P1;"):
                name = line.strip().split(";")[1]
            elif line.startswith("sequence"):
                query.append(name)
            elif line.startswith("structure"):
                templates.append(name)
    if len(query) != 1:
        if len(query) == 0:
            sys.stderr.write("ERROR: there is no QUERY sequence in the PIR file\n")
        else:
            sys.stderr.write("ERROR: there are multiple (%d) QUERY sequences in the PIR file\n"%len(query))
        sys.exit()
    return query[0], tuple(templates)

def update_pir(pir_fn0):
    wrt = []
    with open(pir_fn0) as fp:
        for line in fp:
            if not line.startswith("structure:"):
                wrt.append(line)
                continue
            x = line.strip().split(":")
            x[2] = '' ; x[4] = ''
            wrt.append(":".join(x) + '\n')
    with open("%s.fix"%pir_fn0, 'wt') as fout:
        fout.writelines(wrt)
    return '%s.fix'%pir_fn0

def build_model(pir_fn, pdb_home, n_model, n_proc, ini=None, rsr=None, use_soap=False):
    log.none()
    env = environ()
    env.io.atom_files_directory = pdb_home
    #
    query, templates = parse_pir(pir_fn)
    #
    if n_model > 1 and use_soap:
        soap_protein_od = importlib.import_module("modeller.soap_protein_od")
        model = automodel(env, \
                          alnfile = pir_fn, \
                          knowns = templates, \
                          sequence = query,\
                          inifile = ini,\
                          csrfile = rsr,\
                          assess_methods=soap_protein_od.Scorer())
    else:
        model = automodel(env, \
                          alnfile = pir_fn, \
                          knowns = templates, \
                          sequence = query,\
                          inifile = ini,\
                          csrfile = rsr,\
                          )
    model.starting_model = 1
    model.ending_model = n_model

    if n_proc != 1:
        parallel = importlib.import_module("modeller.parallel")
        proc = parallel.job()
        for i in range(n_proc):
            proc.append(parallel.local_slave())
        model.use_parallel_job(proc)

    model.make()

    if n_model > 1:
        score = []
        for out in model.outputs:
            if use_soap:
                score.append(out['SOAP-Protein-OD score'])
            else:
                score.append(out['molpdf'])
        i_min = score.index(min(score))
        cmd = 'ln -sf %s.B9999%04d.pdb %s.model.pdb'%(query,i_min+1, query)
    else:
        cmd = 'ln -sf %s.B9999%04d.pdb %s.model.pdb'%(query,1, query)
    sys.stdout.write('%s\n'%cmd)
    subprocess.call(cmd, shell=True)
    #
    for i in range(n_model):
        fn = '%s.V9999%04d'%(query, i+1)
        if os.path.exists(fn):
            os.remove(fn)
        fn = '%s.D%08d'%(query, i+1)
        if os.path.exists(fn):
            os.remove(fn)
    #
    if n_proc != 1:
        for i in range(n_proc):
            fn = 'run_modeller.slave%d'%i
            if os.path.exists(fn):
                os.remove(fn)

def main():
    arg = argparse.ArgumentParser(prog='run_modeller')
    arg.add_argument(dest='pir_fn', metavar='ALIGNMENT', \
            help='Sequence alignment file in PIR format')
    arg.add_argument('-p', '--pdb', dest='pdb_home', metavar='PDB_HOME', nargs='*', default=['.'], \
            help='A directory containing PDB files (default: ./)')
    arg.add_argument('-n', '--n_model', dest='n_model', metavar='N_MODEL', default=1, type=int,\
            help='Number of models to be generated (default: 1)')
    arg.add_argument('-c', '--cpu', dest='n_proc', metavar='PROC', default=1, type=int, \
            help='Number of processors to be used (default: 1)')
    arg.add_argument('-i', '--initial', dest='ini', metavar='INI', \
            help='Initial model for structure building')
    arg.add_argument('-r', '--restraint', dest='rsr', metavar='RSR', \
            help='Restraint file for structure building')
    arg.add_argument('--soap', dest='use_soap', action='store_true', default=False)
    arg.add_argument('--fix', dest='use_fix', action='store_true', default=False)
    if len(sys.argv) == 1:
        return arg.print_help()
    arg = arg.parse_args()
    #
    if not arg.use_fix:
        model = build_model(arg.pir_fn, arg.pdb_home, arg.n_model, arg.n_proc, ini=arg.ini, rsr=arg.rsr, use_soap=arg.use_soap)
    else:
        try:
            model = build_model(arg.pir_fn, arg.pdb_home, arg.n_model, arg.n_proc, ini=arg.ini, rsr=arg.rsr, use_soap=arg.use_soap)
        except:
            pir_fn = update_pir(arg.pir_fn)
            model = build_model(pir_fn, arg.pdb_home, arg.n_model, arg.n_proc, ini=arg.ini, rsr=arg.rsr, use_soap=arg.use_soap)
            os.remove(pir_fn)

if __name__ == '__main__':
    main()
