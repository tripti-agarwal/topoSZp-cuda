#!/usr/bin/env python3
"""Experiment "ablation": what each restoration step restores and costs, and
where the GPU time goes.

    python3 exp_ablation.py --data DATA --out RESULTS [--ablation-fields 5] [--ablation-bound rel:1e-3] [--tag t]

Runs the TopoSZp+ tools with SZP_TIMING=1 on --ablation-fields fields per
dataset. Per field it records
  - saddles restored by the saddle stencil, the RBF step and the neighbourhood move,
  - the time of every restoration step on the CPU (OpenMP) and on the GPU,
  - the steps of the one-call GPU compression (range check, upload, detection,
    ranking, rank stream, encoding) and decompression (decoding, ranks, restoration),
  - the GPU time of the one-call path next to the step-by-step path.
With timing on, the GPU programs synchronize the device after every step, so
their totals can be slightly above those of the main experiment.
Output: ablation[.<tag>].csv.
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def add_args(ap):
    ap.add_argument('--ablation-bound', default='rel:1e-3')
    ap.add_argument('--ablation-fields', type=int, default=5, help='fields per dataset (0 = all)')

def timing_line(out, what):
    """Last 'SZP_TIMING <what> k=v ...' line as a dict of floats ({} if absent)."""
    m = re.findall(r'^SZP_TIMING %s\s+(.*)$' % re.escape(what), out, re.M)
    return {k: float(v) for k, v in (x.split('=') for x in m[-1].split() if '=' in x)} if m else {}

def exp_ablation(c, by):
    t = Table(c, 'ablation')
    env = {'OMP_NUM_THREADS': c.threads, 'SZP_TIMING': 1}
    for ds, fields in by.items():
        for f in fields[:c.ablation_fields or None]:
            mode, eb, a, st = resolve(c, f, c.ablation_bound)
            if t.has(dataset=ds, field=f['field'], mode=mode, eb=eb):
                continue
            row = {'dataset': ds, 'field': f['field'], 'dims': 'x'.join(map(str, f['dims'])), 'n': f['n'],
                   'mode': mode, 'eb': eb, 'abs_eb': a, 'threads': c.threads}
            if st:
                t.add(dict(row, status=st)); continue
            lg = 'ablation/%s/%s' % (ds, f['field'])
            # ---- CPU: compress, then decompress with step timing
            p = link(c, f, 'abl')
            rc, out, _ = run(c, [c.tool('toposzp_comp'), p, c.bs, repr(a)] + dimargs(f), lg, env)
            if rc != 0 or not os.path.exists(p + '.SZp'):
                cleanup(p, p + '.SZp'); t.add(dict(row, status='compress_failed', note=first_error(out))); continue
            rc, out, _ = run(c, [c.tool('toposzp_decomp'), p + '.SZp', f['n'], c.bs, repr(a)] + dimargs(f), lg, env)
            cleanup(p, p + '.SZp', p + '.SZp.out')
            m = re.search(r'topology restoration: extrema lost (\d+) -> \d+ of (\d+); saddles lost (\d+) -> (\d+) of (\d+) '
                          r'\(stencil (\d+), RBF (\d+), neighbourhood move (\d+)\)', out)
            ct = timing_line(out, 'restore_cpu')
            if rc != 0 or not m or not ct:
                t.add(dict(row, status='decompress_failed', note=first_error(out) or 'no timing output (rebuild the tools)')); continue
            g = [int(x) for x in m.groups()]
            row.update({'status': 'ok', 'extrema_lost': g[0], 'extrema': g[1], 'saddles_lost': g[2], 'saddles_lost_after': g[3],
                        'saddles': g[4], 'by_stencil': g[5], 'by_rbf': g[6], 'by_move': g[7],
                        'cpu_decomp_s': num(r'decompression time=([0-9.eE+-]+)', out)})
            row.update({'cpu_%s_s' % k: v for k, v in ct.items()})
            # ---- GPU: the test program runs the step-by-step path, then the one-call path
            if c.gpu:
                exe = c.tool('test_cuda_topology_3d' if len(f['dims']) == 3 else 'test_cuda_topology')
                rc, out, _ = run(c, [exe, f['path']] + dimargs(f) + [repr(a), c.bs], lg, env)
                kv = lambda tag: dict(x.split('=') for x in (re.search(r'^%s\s+(.*)$' % tag, out, re.M) or [None, ''])[1].split() if '=' in x)
                one, step = kv('TIMING'), kv('TIMING_STEPWISE')
                gc, gd, gr = timing_line(out, 'compress_gpu'), timing_line(out, 'decompress_gpu'), timing_line(out, 'restore_gpu')
                if rc != 0 or not one or not gc or not gd or re.search(r'^ONECALL.*DIFFERENT', out, re.M):
                    row['note'] = 'GPU: ' + (first_error(out) or 'no one-call timing')
                else:
                    row.update({'gpu_comp_s': one.get('comp_s'), 'gpu_decomp_s': one.get('decomp_s'),
                                'gpu_stepwise_comp_s': step.get('comp_s'), 'gpu_stepwise_decomp_s': step.get('decomp_s')})
                    row.update({'gc_%s_s' % k: v for k, v in gc.items()})
                    row.update({'gd_%s_s' % k: v for k, v in gd.items()})
                    row.update({'gr_%s_s' % k: v for k, v in gr.items() if k != 'on_device'})
            t.add(row)
            lost = max(1, row['saddles_lost'])
            log(c, 'ablation %s/%s: saddles lost %d; restored by stencil %.1f%% RBF %.1f%% move %.1f%%; CPU restore steps '
                   '(s) extrema %.3f stencil %.3f rbf %.3f move %.3f%s' % (
                    ds, f['field'], row['saddles_lost'], 100.0 * row['by_stencil'] / lost, 100.0 * row['by_rbf'] / lost,
                    100.0 * row['by_move'] / lost, ct.get('extrema', 0), ct.get('stencil', 0), ct.get('rbf', 0), ct.get('move', 0),
                    '; GPU comp %s s (step-by-step %s s)' % (row.get('gpu_comp_s'), row.get('gpu_stepwise_comp_s')) if row.get('gpu_comp_s') else ''))

def run_exp(c, by, a):
    c.ablation_bound, c.ablation_fields = a.ablation_bound, a.ablation_fields
    log(c, '==== experiment: ablation (bound %s, %s fields per dataset)' % (a.ablation_bound, a.ablation_fields or 'all'))
    exp_ablation(c, by)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap)
    ap.add_argument('--gpu', choices=['auto', 'on', 'off'], default='auto')
    add_args(ap)
    a = ap.parse_args()
    c, by = context(a, need_gpu=True)
    run_exp(c, by, a)
    log(c, 'done. Results in %s' % c.out)

if __name__ == '__main__':
    main()
