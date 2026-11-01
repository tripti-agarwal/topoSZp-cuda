#!/usr/bin/env python3
"""Run several TopoSZp+ experiments on the SDRBench fields, one after another.

    python3 run_experiments.py --data DATA --out RESULTS [--exp main,scaling,baselines,bitrate,pl,vis] [options]

Each experiment is its own script and can also be run alone, e.g. as a
separate batch job (submit_separate.sh does that):

    exp_main.py       TopoSZp+ OpenMP + CUDA, every field x bound      -> main.csv
    exp_scaling.py    OpenMP thread sweep                              -> scaling.csv
    exp_baselines.py  TopoSZp+ vs SZp, SZ3, SZ2, ZFP (TTHRESH)         -> baselines.csv
    exp_bitrate.py    bit rate vs FN / FP / FT                         -> bitrate.csv
    exp_pl.py         check against TTK's piecewise-linear definition  -> pl.csv
    exp_vis.py        fields for the visualization figure              -> vis/

This wrapper accepts the options of all of them (--help lists them). Every
experiment resumes: runs already in RESULTS are skipped. Then run
summarize.py RESULTS for the tables and plots.
"""
import argparse, importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, common_args, context, log, split

EXPERIMENTS = ['main', 'scaling', 'baselines', 'bitrate', 'pl', 'vis']

def main():
    mods = {e: importlib.import_module('exp_' + e) for e in EXPERIMENTS}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 conflict_handler='resolve')   # --compressors is an option of two experiments
    common_args(ap)
    ap.add_argument('--exp', default=','.join(EXPERIMENTS), help='experiments to run, in this order')
    for e in EXPERIMENTS:
        mods[e].add_args(ap)
    a = ap.parse_args()
    exps = split(a.exp)
    for e in exps:
        if e not in mods:
            sys.exit('unknown experiment %s (choose from %s)' % (e, ','.join(EXPERIMENTS)))
    c, by = context(a, need_gpu='main' in exps,
                    need_compressors=a.compressors if {'baselines', 'bitrate'} & set(exps) else None)
    for e in exps:
        mods[e].run_exp(c, by, a)
    log(c, 'done. Results in %s; tables: python3 %s %s' % (c.out, os.path.join(HERE, 'summarize.py'), c.out))

if __name__ == '__main__':
    main()
