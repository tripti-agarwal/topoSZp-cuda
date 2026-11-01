#!/usr/bin/env python3
"""Experiment "baselines": TopoSZp+ vs plain SZp, SZ3, SZ2, ZFP (and TTHRESH if built).

    python3 exp_baselines.py --data DATA --out RESULTS [--datasets ...] [--baseline-bounds ...] [--tag t]

Every field x every bound with every compressor; FN / FP / FT counted with the
same checker (topo_check). Output: baselines[.<tag>].csv. Feeds the average
false cases table.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def add_args(ap):
    ap.add_argument('--compressors', default='toposzp,szp,sz3,sz,zfp,tthresh', help='missing tools are skipped')
    ap.add_argument('--baseline-bounds', default='abs:1e-3,abs:1e-4,abs:1e-5,rel:1e-3,rel:1e-4,rel:1e-5')
    ap.add_argument('--baseline-fields', type=int, default=0, help='fields per dataset (0 = all)')

def run_exp(c, by, a):
    c.cmp_fields = a.baseline_fields
    log(c, '==== experiment: baselines (bounds %s)' % a.baseline_bounds)
    compare(c, by, 'baselines', split(a.baseline_bounds))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap)
    add_args(ap)
    a = ap.parse_args()
    c, by = context(a, need_compressors=a.compressors)
    run_exp(c, by, a)
    log(c, 'done. Results in %s; tables: python3 %s %s' % (c.out, os.path.join(HERE, 'summarize.py'), c.out))

if __name__ == '__main__':
    main()
