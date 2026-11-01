#!/usr/bin/env python3
"""Experiment "scaling": OpenMP thread sweep of TopoSZp+.

    python3 exp_scaling.py --data DATA --out RESULTS [--scaling-threads 1,2,4,8,16,32] [--tag t]

Full compression and decompression (critical points, ranks, encoding / decoding,
restoration) on --scaling-fields fields per dataset at --scaling-bound, plus
plain SZp times, and whether the output is identical for every thread count.
Output: scaling[.<tag>].csv. Feeds the parallel scalability table.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def add_args(ap):
    ncores = usable_cpus()[1]
    ths = [t for t in [1, 2, 4, 8, 16, 32, 64, 128] if t < ncores] + [ncores]
    ap.add_argument('--scaling-threads', default=','.join(map(str, ths)),
                    help='default: powers of two up to the physical cores of this job (%d)' % ncores)
    ap.add_argument('--scaling-bound', default='rel:1e-3')
    ap.add_argument('--scaling-fields', type=int, default=5, help='fields per dataset (0 = all)')

def exp_scaling(c, by):
    t = Table(c, 'scaling')
    for ds, fields in by.items():
        for f in fields[:c.scaling_fields or None]:
            mode, eb, a, st = resolve(c, f, c.scaling_bound)
            ref = None
            for th in c.scaling_threads:
                if t.has(dataset=ds, field=f['field'], mode=mode, eb=eb, threads=th):
                    continue
                row = {'dataset': ds, 'field': f['field'], 'dims': 'x'.join(map(str, f['dims'])), 'n': f['n'],
                       'mode': mode, 'eb': eb, 'abs_eb': a, 'threads': th}
                if st:
                    t.add(dict(row, status=st)); continue
                lg = 'scaling/%s/%s_T%d' % (ds, f['field'], th)
                r = toposzp_cpu(c, f, a, th, lg, keep_output=True)
                if r['status'] != 'ok':
                    t.add(dict(row, status=r['status'], note=r['note'])); continue
                h = md5(r['output'])
                if ref is None:
                    ref = h
                tp = topo(c, f, r['output'], a, lg) if th == c.scaling_threads[0] else None
                z = plain_szp(c, f, a, th, lg)
                cleanup(*r['_cleanup'])
                row.update({'status': 'ok', 'comp_s': r['comp_s'], 'decomp_s': r['decomp_s'],
                            'szp_comp_s': z.get('comp_s'), 'szp_decomp_s': z.get('decomp_s'),
                            'err_after': tp['after']['err'] if tp else '', 'identical_to_first': int(h == ref)})
                t.add(row)
                log(c, 'scaling %s/%s T=%d: comp %.4f s decomp %.4f s (SZp %s/%s) identical=%d' % (
                    ds, f['field'], th, r['comp_s'], r['decomp_s'], fmt(z.get('comp_s'), '%.4f'), fmt(z.get('decomp_s'), '%.4f'), h == ref))

def run_exp(c, by, a):
    c.scaling_threads = [int(x) for x in split(a.scaling_threads)]
    c.scaling_bound, c.scaling_fields = a.scaling_bound, a.scaling_fields
    log(c, '==== experiment: scaling (threads %s, bound %s)' % (a.scaling_threads, a.scaling_bound))
    exp_scaling(c, by)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap)
    add_args(ap)
    a = ap.parse_args()
    c, by = context(a, )
    run_exp(c, by, a)
    log(c, 'done. Results in %s; tables: python3 %s %s' % (c.out, os.path.join(HERE, 'summarize.py'), c.out))

if __name__ == '__main__':
    main()
