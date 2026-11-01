#!/usr/bin/env python3
"""Experiment "pl": TopoSZp+ output under the piecewise-linear (TTK) definition.

    python3 exp_pl.py --data DATA --out RESULTS [--pl-fields 3] [--pl-bound rel:1e-3] [--tag t]

pl_topo_check on --pl-fields fields per dataset: PL extrema and saddles kept,
PL FP / FT, saddles by persistence, and how many axis-rule saddles are PL saddles.
Output: pl[.<tag>].csv.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def add_args(ap):
    ap.add_argument('--pl-bound', default='rel:1e-3')
    ap.add_argument('--pl-fields', type=int, default=3, help='fields per dataset (0 = all)')

def exp_pl(c, by):
    t = Table(c, 'pl')
    for ds, fields in by.items():
        for f in fields[:c.pl_fields or None]:
            mode, eb, a, st = resolve(c, f, c.pl_bound)
            if t.has(dataset=ds, field=f['field'], mode=mode, eb=eb):
                continue
            row = {'dataset': ds, 'field': f['field'], 'dims': 'x'.join(map(str, f['dims'])), 'mode': mode, 'eb': eb, 'abs_eb': a}
            if st:
                t.add(dict(row, status=st)); continue
            lg = 'pl/%s/%s_%s_%s' % (ds, f['field'], mode, eb)
            r = toposzp_cpu(c, f, a, c.threads, lg, keep_output=True)
            if r['status'] != 'ok':
                t.add(dict(row, status=r['status'], note=r['note'])); continue
            rc, out, _ = run(c, [c.tool('pl_topo_check'), f['path'], r['output'], repr(a)] + dimargs(f), lg,
                             {'OMP_NUM_THREADS': c.nproc}, timeout=c.timeout * 4)
            cleanup(*r['_cleanup'])
            m = re.search(r'^PLTOPO (.+)$', out, re.M)
            tot = re.search(r'^total\s+(\d+) \|\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\S+\|\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', out, re.M)
            if not m or not tot:
                t.add(dict(row, status='check_failed', note=first_error(out))); continue
            kv = dict(x.rsplit('=', 1) for x in m.group(1).split() if '=' in x)
            g = [int(x) for x in tot.groups()]
            row.update({'status': 'ok', 'pl_extrema': kv['extrema'], 'pl_extrema_kept_final_pct': kv['extrema_kept_final'].rstrip('%'),
                        'pl_saddles': kv['saddles'], 'pl_saddles_kept_raw_pct': kv['kept_raw'].rstrip('%'),
                        'pl_saddles_kept_final_pct': kv['kept_final'].rstrip('%'),
                        'pl_saddles_kept_final_1cell_pct': kv['kept_final_1cell'].rstrip('%'),
                        'pl_saddles_pers_ge_2eb_kept_final_pct': kv['kept_final_pers>=2eb'].rstrip('%'),
                        'axis_saddles': kv['axis_saddles'], 'axis_saddles_that_are_pl_pct': kv['axis_saddles_that_are_PL'].rstrip('%'),
                        'total_orig': g[0], 'kept_raw': g[1], 'FN_raw': g[2], 'FT_raw': g[3], 'FP_raw': g[4],
                        'kept_final': g[5], 'FN_final': g[6], 'FT_final': g[7], 'FP_final': g[8], 'err_final': kv['err/eb']})
            t.add(row)
            log(c, 'pl %s/%s %s: PL extrema kept %s%%, PL saddles kept %s%% -> %s%%, FP %d FT %d' % (
                ds, f['field'], c.pl_bound, row['pl_extrema_kept_final_pct'], row['pl_saddles_kept_raw_pct'],
                row['pl_saddles_kept_final_pct'], g[8], g[7]))

def run_exp(c, by, a):
    c.pl_bound, c.pl_fields = a.pl_bound, a.pl_fields
    log(c, '==== experiment: pl (bound %s)' % a.pl_bound)
    exp_pl(c, by)

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
