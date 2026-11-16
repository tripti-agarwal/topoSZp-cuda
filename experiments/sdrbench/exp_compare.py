#!/usr/bin/env python3
"""Experiment "compare": TopoSZp+ next to topology-preserving compressors
(LOPC, and TopoSZ / TopoA when described in external_compressors.json) and
error-bounded ones, with every quality measure on the same fields.

    python3 exp_compare.py --data DATA --out RESULTS [--compare-fields 3] [--compare-bounds rel:1e-3,rel:1e-4]
                           [--compare-compressors toposzp,szp,sz3,zfp,lopc,lopc-gpu] [--external FILE] [--tag t]

Per compressor, field and bound:
  - size, bit rate, compression / decompression time;
  - critical points by the axis rule with strict comparisons (topo_check): kept, FN, FP, FT;
  - the same with ties broken by index (quality_check, tb_*), the convention of LOPC and TTK;
  - PSNR and maximum error;
  - piecewise-linear critical points (pl_topo_check): extrema and saddles kept,
    saddles with persistence >= 2 eps kept, FN / FT / FP.
All compressors get the same absolute bound. Compressors that preserve a
simplified contour tree (TopoSZ, TopoA) have a persistence parameter as well;
set it in their command templates. Slow compressors are limited by --timeout
(status "compress_failed", note TIMEOUT).
Output: compare[.<tag>].csv.
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def add_args(ap):
    ap.add_argument('--compare-compressors', default='toposzp,szp,sz3,zfp,lopc,lopc-gpu',
                    help='missing tools are skipped; compressors of external_compressors.json are added')
    ap.add_argument('--compare-bounds', default='rel:1e-3,rel:1e-4')
    ap.add_argument('--compare-fields', type=int, default=3, help='fields per dataset (0 = all)')
    ap.add_argument('--external', default=None, help='JSON with command templates of further compressors '
                                                     '(default: <bin>/external_compressors.json)')
    ap.add_argument('--no-pl', action='store_true', help='skip the piecewise-linear check (slow on large 3D fields)')

def exp_compare(c, by):
    t = Table(c, 'compare')
    for ds, fields in by.items():
        for f in fields[:c.compare_fields or None]:
            for bound in c.compare_bounds:
                mode, eb, a, st = resolve(c, f, bound)
                for comp in c.compressors:
                    if t.has(compressor=comp, dataset=ds, field=f['field'], mode=mode, eb=eb):
                        continue
                    row = {'compressor': comp, 'dataset': ds, 'field': f['field'], 'dims': 'x'.join(map(str, f['dims'])),
                           'n': f['n'], 'mode': mode, 'eb': eb, 'abs_eb': a}
                    if st and comp in ('toposzp', 'szp'):
                        t.add(dict(row, status=st)); continue
                    lg = 'compare/%s/%s/%s_%s_%s' % (comp, ds, f['field'], mode, eb)
                    r = baseline(c, comp, f, a, lg)
                    if r['status'] != 'ok':
                        t.add(dict(row, status=r['status'], note=r.get('note', ''))); continue
                    tp = topo(c, f, r['output'], a, lg)
                    rc, out, _ = run(c, [c.tool('quality_check'), f['path'], r['output']] + dimargs(f), lg,
                                     {'OMP_NUM_THREADS': c.ncores})
                    q = re.search(r'^QUALITY (.+)$', out, re.M)
                    if not tp or not q:
                        cleanup(*r['_cleanup']); t.add(dict(row, status='check_failed', note=first_error(out))); continue
                    qv = dict(x.split('=') for x in q.group(1).split())
                    row.update({'status': 'ok', 'bytes': r['bytes'], 'ratio': 4.0 * f['n'] / r['bytes'],
                                'bitrate': 8.0 * r['bytes'] / f['n'], 'comp_s': r['comp_s'], 'decomp_s': r['decomp_s'],
                                'psnr': qv['psnr']})
                    row.update(tp['after'])
                    row.update({k: qv[k] for k in ('tb_orig', 'tb_kept', 'tb_FN', 'tb_FP', 'tb_FT')})
                    if not c.no_pl:
                        rc, out, _ = run(c, [c.tool('pl_topo_check'), f['path'], r['output'], repr(a)] + dimargs(f), lg,
                                         {'OMP_NUM_THREADS': c.ncores}, timeout=c.timeout * 4)
                        m = re.search(r'^PLTOPO (.+)$', out, re.M)
                        tot = re.search(r'^total\s+(\d+) \|\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\S+\|\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', out, re.M)
                        if m and tot:
                            kv = dict(x.rsplit('=', 1) for x in m.group(1).split() if '=' in x)
                            g = [int(x) for x in tot.groups()]
                            row.update({'pl_extrema': kv['extrema'], 'pl_extrema_kept_pct': kv['extrema_kept_final'].rstrip('%'),
                                        'pl_saddles': kv['saddles'], 'pl_saddles_kept_pct': kv['kept_final'].rstrip('%'),
                                        'pl_saddles_pers_ge_2eb_kept_pct': kv['kept_final_pers>=2eb'].rstrip('%'),
                                        'pl_FN': g[6], 'pl_FT': g[7], 'pl_FP': g[8]})
                        else:
                            row['note'] = 'PL check failed: ' + first_error(out)
                    cleanup(*r['_cleanup'])
                    t.add(row)
                    log(c, 'compare %s %s/%s %s: %.2f bits  PSNR %s  strict FN %d FP %d FT %d  tie-broken FN %s FP %s FT %s  '
                           'PL saddles kept %s%% (pers>=2eps %s%%) PL FP %s  comp %.3f s decomp %.3f s' % (
                            comp, ds, f['field'], bound, row['bitrate'], row['psnr'], row['FN'], row['FP'], row['FT'],
                            row['tb_FN'], row['tb_FP'], row['tb_FT'], row.get('pl_saddles_kept_pct', '-'),
                            row.get('pl_saddles_pers_ge_2eb_kept_pct', '-'), row.get('pl_FP', '-'), r['comp_s'], r['decomp_s']))

def run_exp(c, by, a):
    c.compare_bounds, c.compare_fields, c.no_pl = split(a.compare_bounds), a.compare_fields, a.no_pl
    log(c, '==== experiment: compare (%s; bounds %s; %s fields per dataset)' % (
        ', '.join(c.compressors), a.compare_bounds, a.compare_fields or 'all'))
    exp_compare(c, by)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap)
    add_args(ap)
    a = ap.parse_args()
    c, by = context(a, need_compressors=a.compare_compressors)
    run_exp(c, by, a)
    log(c, 'done. Results in %s' % c.out)

if __name__ == '__main__':
    main()
