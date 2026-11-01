#!/usr/bin/env python3
"""Experiment "main": TopoSZp+ on every field x every bound, OpenMP and CUDA.

    python3 exp_main.py --data DATA --out RESULTS [--datasets CESM-ATM-3D] [--bounds ...] [--tag t]

Per run: compression and decompression time (CPU, and GPU if available),
compressed size of TopoSZp+ and of plain SZp, critical points of plain SZp
quantization (before) and of the TopoSZp+ output (after): kept maxima / minima /
saddles, FP, FN, FT, max error / eps, and whether CPU and GPU outputs are
bit-identical. Output: main[.<tag>].csv. Feeds the dataset table, the topology
preservation table, the 3D / GPU section and the CPU vs GPU timing table.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def add_args(ap):
    ap.add_argument('--bounds', default='abs:1e-3,abs:1e-4,abs:1e-5,rel:1e-3,rel:1e-4,rel:1e-5',
                    help='mode:value list; rel = fraction of each field\'s value range')
    ap.add_argument('--gpu', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--gpu-reps', type=int, default=1,
                    help='timed repetitions of the CUDA program (it also runs an OpenMP comparison internally)')

def exp_main(c, by):
    t = Table(c, 'main')
    for ds, fields in by.items():
        for f in fields:
            for bound in c.bounds:
                mode, eb, a, st = resolve(c, f, bound)
                if t.has(dataset=ds, field=f['field'], mode=mode, eb=eb):
                    continue
                s = stats(c, f)
                row = {'dataset': ds, 'field': f['field'], 'dims': 'x'.join(map(str, f['dims'])), 'n': f['n'],
                       'range': s['range'], 'absmax': s['absmax'], 'mode': mode, 'eb': eb, 'abs_eb': a,
                       'threads': c.threads}
                if st:
                    t.add(dict(row, status=st, note='max|v|/eb >= 2^31')); log(c, 'main %s/%s %s: %s' % (ds, f['field'], bound, st)); continue
                lg = 'main/%s/%s_%s_%s' % (ds, f['field'], mode, eb)
                r = toposzp_cpu(c, f, a, c.threads, lg, keep_output=True)
                if r['status'] != 'ok':
                    t.add(dict(row, status=r['status'], note=r['note'])); log(c, 'main %s/%s %s: %s (%s)' % (ds, f['field'], bound, r['status'], r['note'])); continue
                row.update({'cpu_comp_s': r['comp_s'], 'cpu_decomp_s': r['decomp_s'], 'toposzp_bytes': r['bytes'],
                            'toposzp_ratio': '%.4f' % (4.0 * f['n'] / r['bytes'])})
                tp = topo(c, f, r['output'], a, lg)
                if not tp:
                    cleanup(*r['_cleanup']); t.add(dict(row, status='check_failed')); continue
                for stage in ('before', 'after'):
                    row.update({k + '_' + stage: v for k, v in tp[stage].items()})
                z = plain_szp(c, f, a, c.threads, lg)
                if z['status'] == 'ok':
                    row.update({'szp_bytes': z['bytes'], 'szp_ratio': '%.4f' % (4.0 * f['n'] / z['bytes']),
                                'szp_comp_s': z['comp_s'], 'szp_decomp_s': z['decomp_s']})
                if c.gpu:
                    g = toposzp_gpu(c, f, a, lg)
                    if g['status'] == 'ok':
                        row.update({'gpu_comp_s': g['comp_s'], 'gpu_decomp_s': g['decomp_s']})
                        same = filecmp.cmp(r['output'], g['output'], shallow=False)
                        row['cpu_gpu_identical'] = int(same)
                        if not same:
                            gt = topo(c, f, g['output'], a, lg)
                            if gt:
                                row.update({'gpu_FP': gt['after']['FP'], 'gpu_FN': gt['after']['FN'],
                                            'gpu_FT': gt['after']['FT'], 'gpu_err': gt['after']['err']})
                        cleanup(g['output'])
                    else:
                        row['note'] = 'GPU: ' + g['note']
                cleanup(*r['_cleanup'])
                row['status'] = 'ok'
                t.add(row)
                a_ = tp['after']
                log(c, 'main %s/%s %s: ratio %s  ext %d/%d  sad %d->%d/%d  FP %d FT %d  err %.3f eps  cpu %.3f/%.3f s%s' % (
                    ds, f['field'], bound, row['toposzp_ratio'], a_['max_kept'] + a_['min_kept'], a_['max_orig'] + a_['min_orig'],
                    tp['before']['sad_kept'], a_['sad_kept'], a_['sad_orig'], a_['FP'], a_['FT'], a_['err'],
                    r['comp_s'], r['decomp_s'],
                    ('  gpu %.3f/%.3f s identical=%s' % (row['gpu_comp_s'], row['gpu_decomp_s'], row.get('cpu_gpu_identical'))
                     if 'gpu_comp_s' in row else '')))

def run_exp(c, by, a):
    c.bounds = split(a.bounds)
    log(c, '==== experiment: main (bounds %s)' % ','.join(c.bounds))
    exp_main(c, by)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap)
    add_args(ap)
    a = ap.parse_args()
    c, by = context(a, need_gpu=True)
    run_exp(c, by, a)
    log(c, 'done. Results in %s; tables: python3 %s %s' % (c.out, os.path.join(HERE, 'summarize.py'), c.out))

if __name__ == '__main__':
    main()
