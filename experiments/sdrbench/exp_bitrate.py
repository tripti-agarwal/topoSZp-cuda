#!/usr/bin/env python3
"""Experiment "bitrate": bit rate vs FN / FP / FT for TopoSZp+ and the baselines.

    python3 exp_bitrate.py --data DATA --out RESULTS [--bitrate-datasets CESM-ATM-2D] [--tag t]

The compressors of exp_baselines.py over --bitrate-bounds on --bitrate-datasets.
Output: bitrate[.<tag>].csv; summarize.py draws plots/bitrate_<dataset>.pdf.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def add_args(ap):
    ap.add_argument('--compressors', default='toposzp,szp,sz3,sz,zfp,tthresh', help='missing tools are skipped')
    ap.add_argument('--bitrate-datasets', default='CESM-ATM-2D')
    ap.add_argument('--bitrate-bounds', default='abs:1e-2,abs:5e-3,abs:1e-3,abs:5e-4,abs:1e-4,abs:5e-5,abs:1e-5,abs:5e-6,abs:1e-6')
    ap.add_argument('--bitrate-fields', type=int, default=0, help='fields per dataset (0 = all)')

def run_exp(c, by, a):
    c.cmp_fields = a.bitrate_fields
    log(c, '==== experiment: bitrate (datasets %s)' % a.bitrate_datasets)
    compare(c, by, 'bitrate', split(a.bitrate_bounds), split(a.bitrate_datasets))

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
