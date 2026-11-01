#!/usr/bin/env python3
"""Turn the CSVs of the experiment scripts (exp_*.py / run_experiments.py) into the paper's tables and plots.

    python3 summarize.py RESULTS_DIR [--scaling-mode rel --scaling-eb 1e-3]

Writes RESULTS_DIR/summary.md (all tables, readable), RESULTS_DIR/tables.tex
(LaTeX tabular bodies, booktabs) and RESULTS_DIR/plots/ (bit rate vs FN / FP /
FT / total, if matplotlib is installed). Averages are over the fields of a
dataset that ran successfully; runs skipped because of the format limit or
failures are counted separately and never averaged in.
"""
import argparse, csv, glob, math, os, sys
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_results   # merges <name>.csv and the per-job <name>.<tag>.csv files

def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None

def pct(k, o):
    k, o = f(k), f(o)
    return None if k is None or not o else 100.0 * k / o

def n(x, p=1):
    if x is None:
        return '--'
    if isinstance(x, float) and abs(x) >= 1000 and p <= 1:
        return '{:,.0f}'.format(x)
    return ('%.' + str(p) + 'f') % x

def x(v):   # ratio with a times sign
    return '--' if v is None else '%.2f×' % v

def bound(r):
    return '%s %s' % (r['mode'], r['eb'])

def groups(rows, *keys):
    g = OrderedDict()
    for r in rows:
        g.setdefault(tuple(r[k] for k in keys), []).append(r)
    return g

class Out:
    def __init__(self):
        self.md, self.tex = [], []
    def table(self, title, head, rows, note=''):
        self.md += ['## ' + title, '', '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
        self.md += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows] + ([''] + [note] if note else []) + ['']
        esc = lambda s: str(s).replace('%', '\\%').replace('_', '\\_').replace('->', '$\\to$').replace('×', '$\\times$')
        self.tex += ['%% ' + title, '\\begin{tabular}{l' + 'r' * (len(head) - 1) + '}', '\\toprule',
                     ' & '.join('\\textbf{%s}' % esc(h) for h in head) + ' \\\\', '\\midrule']
        self.tex += [' & '.join(esc(c) for c in r) + ' \\\\' for r in rows] + ['\\bottomrule', '\\end{tabular}', '']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--table-bound', default='abs 1e-3', help='bound used for the dataset / CPU-GPU tables')
    a = ap.parse_args()
    R = a.results
    out = Out()
    mfiles = sorted(glob.glob(os.path.join(R, 'machine*.txt')))
    machine = open(mfiles[0]).read() if mfiles else ''
    out.md += ['# TopoSZp+ experiment summary', '', '```', machine.strip(), '```', '']

    main_rows = read_results(R, 'main')
    ok = [r for r in main_rows if r['status'] == 'ok']

    # ------------------------------------------------------------------ datasets
    if main_rows:
        rows = []
        for (ds,), rs in groups(main_rows, 'dataset').items():
            flds = OrderedDict((r['field'], r) for r in rs)
            nvals = int(next(iter(flds.values()))['n'])
            dims = next(iter(flds.values()))['dims'].replace('x', ' × ')
            tb = [r for r in rs if r['status'] == 'ok' and bound(r) == a.table_bound] or \
                 [r for r in rs if r['status'] == 'ok']   # bound not run: all bounds
            rows.append([ds, len(flds), dims, n(len(flds) * nvals * 4 / 1e9, 2) + ' GB',
                         n(mean([f(r['err_after']) for r in tb]), 3), n(max([f(r['err_after']) for r in tb] or [None]) if tb else None, 3)])
        have_tb = any(r['status'] == 'ok' and bound(r) == a.table_bound for r in main_rows)
        out.table('Datasets', ['Dataset', 'Fields', 'Dimensions', 'Size',
                               'Mean max error / eps (%s)' % (a.table_bound if have_tb else 'all bounds'),
                               'Largest max error / eps'], rows,
                  'Max error of TopoSZp+ in units of the error bound eps (the guarantee is < 2).')

    # ------------------------------------------------------------------ preservation (main)
    if main_rows:
        rows = []
        for (ds, mode, eb), rs in groups(main_rows, 'dataset', 'mode', 'eb').items():
            g = [r for r in rs if r['status'] == 'ok']
            skipped = sum(r['status'] == 'format_limit' for r in rs)
            failed = len(rs) - len(g) - skipped
            ext = lambda r, s: pct(f(r['max_kept_' + s]) + f(r['min_kept_' + s]), f(r['max_orig_' + s]) + f(r['min_orig_' + s])) if g else None
            allcp = lambda r, s: pct(f(r['max_kept_' + s]) + f(r['min_kept_' + s]) + f(r['sad_kept_' + s]),
                                     f(r['max_orig_' + s]) + f(r['min_orig_' + s]) + f(r['sad_orig_' + s]))
            rows.append([ds, '%s %s' % (mode, eb), '%d%s' % (len(g), (' (+%d skipped)' % skipped if skipped else '') + (' (%d failed)' % failed if failed else '')),
                         x(mean([f(r['toposzp_ratio']) for r in g])), x(mean([f(r['szp_ratio']) for r in g])),
                         '%s -> %s' % (n(mean([allcp(r, 'before') for r in g])), n(mean([allcp(r, 'after') for r in g]))),
                         '%s -> %s' % (n(mean([ext(r, 'before') for r in g])), n(mean([ext(r, 'after') for r in g]))),
                         '%s -> %s' % (n(mean([pct(r['sad_kept_before'], r['sad_orig_before']) for r in g])),
                                      n(mean([pct(r['sad_kept_after'], r['sad_orig_after']) for r in g]))),
                         '%s -> %s' % (n(mean([f(r['FN_before']) for r in g]), 0), n(mean([f(r['FN_after']) for r in g]), 0)),
                         n(sum(f(r['FP_after']) for r in g), 0), n(sum(f(r['FT_after']) for r in g), 0),
                         n(max([f(r['err_after']) for r in g] or [None]) if g else None, 3)])
        out.table('Topology preservation: plain SZp (before) -> TopoSZp+ (after)',
                  ['Dataset', 'Bound', 'Fields', 'Ratio TopoSZp+', 'Ratio SZp', 'All CPs kept %', 'Extrema kept %',
                   'Saddles kept %', 'FN per field', 'FP', 'FT', 'Max err / eps'], rows,
                  'Percentages and FN are means over fields; FP and FT are totals over fields. "skipped": max|value|/eps exceeds '
                  'the 32-bit bin index of the format (use a relative bound for these fields).')

    # ------------------------------------------------------------------ CPU vs GPU
    if ok:
        rows = []
        for (ds, mode, eb), rs in groups(ok, 'dataset', 'mode', 'eb').items():
            gb = mean([4.0 * f(r['n']) / 1e9 for r in rs])
            tc, td = mean([f(r['cpu_comp_s']) for r in rs]), mean([f(r['cpu_decomp_s']) for r in rs])
            gc, gd = mean([f(r['gpu_comp_s']) for r in rs]), mean([f(r['gpu_decomp_s']) for r in rs])
            ident = [r['cpu_gpu_identical'] for r in rs if r['cpu_gpu_identical'] != '']
            rows.append([ds, '%s %s' % (mode, eb), len(rs), n(tc, 4), n(gb / tc if tc else None, 2), n(td, 4), n(gb / td if td else None, 2),
                         n(gc, 4), n(gb / gc if gc else None, 2), n(gd, 4), n(gb / gd if gd else None, 2),
                         '%d/%d' % (sum(x == '1' for x in ident), len(ident)) if ident else '--'])
        gnotes = sorted({r['note'] for r in ok if r.get('note', '').startswith('GPU:')})
        out.table('Compression and decompression time: OpenMP (%s threads) and CUDA' % (ok[0]['threads']),
                  ['Dataset', 'Bound', 'Fields', 'CPU comp s', 'GB/s', 'CPU decomp s', 'GB/s', 'GPU comp s', 'GB/s',
                   'GPU decomp s', 'GB/s', 'CPU = GPU output'], rows,
                  'Means per field. Compression = critical points + ranks + encoding; decompression = decoding + topology '
                  'restoration. GPU times are those reported by the CUDA test programs.'
                  + ('  GPU runs failed: %s' % '; '.join(gnotes[:5]) if gnotes else '')
                  + ('  No GPU times: the CUDA programs were not run (no GPU or --gpu off).'
                     if not gnotes and not any(r.get('gpu_comp_s') for r in ok) else ''))

    # ------------------------------------------------------------------ scaling
    sc = [r for r in read_results(R, 'scaling') if r['status'] == 'ok']
    if sc:
        ths = sorted({int(r['threads']) for r in sc})
        rows = []
        for (ds,), rs in groups(sc, 'dataset').items():
            byt = defaultdict(list)
            for r in rs:
                byt[int(r['threads'])].append(r)
            c1 = mean([f(r['comp_s']) for r in byt[ths[0]]])
            cells = []
            for t in ths:
                cells.append(n(mean([f(r['comp_s']) for r in byt[t]]), 4))
            best = min(ths, key=lambda t: mean([f(r['comp_s']) for r in byt[t]]) or 1e30)
            sp = c1 / mean([f(r['comp_s']) for r in byt[best]]) if c1 else None
            dsp = mean([f(r['decomp_s']) for r in byt[ths[0]]]) / mean([f(r['decomp_s']) for r in byt[ths[-1]]]) if byt[ths[-1]] else None
            ident = all(r['identical_to_first'] == '1' for r in rs)
            rows.append([ds, len({r['field'] for r in rs})] + cells + ['%s× (%d t)' % (n(sp, 1), best), n(dsp, 1) + '×', 'yes' if ident else 'NO'])
        out.table('OpenMP scalability: mean compression time per field (s), bound %s' % bound(sc[0]),
                  ['Dataset', 'Fields'] + ['%d t' % t for t in ths] + ['Best speedup', 'Decomp speedup %d t' % ths[-1], 'Identical output'],
                  rows, 'Full pipeline (critical points, ranks, encoding). Identical output = the decompressed field is the same for every thread count.')

    # ------------------------------------------------------------------ baselines
    for name, title in (('baselines', 'Compressor comparison'), ('bitrate', 'Bit-rate sweep')):
        br = read_results(R, name)
        if not br:
            continue
        g = [r for r in br if r['status'] == 'ok']
        rows = []
        order = ['toposzp', 'szp', 'sz3', 'sz', 'zfp', 'tthresh']
        label = {'toposzp': 'TopoSZp+', 'szp': 'SZp', 'sz3': 'SZ3', 'sz': 'SZ2', 'zfp': 'ZFP', 'tthresh': 'TTHRESH*'}
        dropped = []
        for (ds, mode, eb), rs in groups(g, 'dataset', 'mode', 'eb').items():
            # Average every compressor over the same fields: those all of them finished.
            bycomp = groups(rs, 'compressor')
            common = set.intersection(*[{r['field'] for r in v} for v in bycomp.values()])
            dropped += sorted({'%s/%s %s %s' % (ds, r['field'], mode, eb) for r in rs if r['field'] not in common})
            for comp in [c for c in order if (c,) in bycomp]:
                cr = [r for r in bycomp[(comp,)] if r['field'] in common]
                if not cr:
                    continue
                tot = [f(r['FN']) + f(r['FP']) + f(r['FT']) for r in cr]
                rows.append([ds, '%s %s' % (mode, eb), label[comp], len(cr), n(mean([f(r['bitrate']) for r in cr]), 2),
                             x(mean([f(r['ratio']) for r in cr])), n(mean([f(r['FN']) for r in cr]), 0),
                             n(mean([f(r['FP']) for r in cr]), 0), n(mean([f(r['FT']) for r in cr]), 0), n(mean(tot), 0),
                             n(max(f(r['err']) for r in cr), 3)])
        fails = [r for r in br if r['status'] != 'ok']
        out.table(title, ['Dataset', 'Bound', 'Compressor', 'Fields', 'Bits/value', 'Ratio', 'FN', 'FP', 'FT', 'Total', 'Max err / eps'],
                  rows, 'Means over fields; FN/FP/FT counted with the same checker (topo_check) for every compressor. '
                        '*TTHRESH -e is a relative (Frobenius) target, not a pointwise bound.'
                        + ('  %d runs did not finish (see %s.csv, status column).' % (len(fails), name) if fails else '')
                        + ('  Left out because not every compressor finished them: %s.' % ', '.join(dropped) if dropped else ''))
        if name == 'bitrate':
            plot_bitrate(R, g, label, order)

    # ------------------------------------------------------------------ PL check
    pl = [r for r in read_results(R, 'pl') if r['status'] == 'ok']
    if pl:
        rows = []
        def pc(r, k, count):   # a percentage of zero items (or the checker's -1) is not a number
            v = f(r[k])
            return None if v is None or v < 0 or not f(r.get(count)) else v
        for (ds, mode, eb), rs in groups(pl, 'dataset', 'mode', 'eb').items():
            rows.append([ds, '%s %s' % (mode, eb), len(rs),
                         n(sum(f(r['pl_extrema']) or 0 for r in rs), 0), n(mean([pc(r, 'pl_extrema_kept_final_pct', 'pl_extrema') for r in rs])),
                         n(sum(f(r['pl_saddles']) or 0 for r in rs), 0),
                         '%s -> %s' % (n(mean([pc(r, 'pl_saddles_kept_raw_pct', 'pl_saddles') for r in rs])),
                                      n(mean([pc(r, 'pl_saddles_kept_final_pct', 'pl_saddles') for r in rs]))),
                         n(mean([pc(r, 'pl_saddles_pers_ge_2eb_kept_final_pct', 'pl_saddles') for r in rs])),
                         n(mean([f(r['FP_raw']) for r in rs]), 0) + ' -> ' + n(mean([f(r['FP_final']) for r in rs]), 0),
                         n(mean([f(r['FT_raw']) for r in rs]), 0) + ' -> ' + n(mean([f(r['FT_final']) for r in rs]), 0),
                         n(sum(f(r['axis_saddles']) or 0 for r in rs), 0),
                         n(mean([pc(r, 'axis_saddles_that_are_pl_pct', 'axis_saddles') for r in rs]))])
        out.table('Cross-check with the piecewise-linear (TTK) definition',
                  ['Dataset', 'Bound', 'Fields', 'PL extrema', 'PL extrema kept %', 'PL saddles', 'PL saddles kept % (SZp -> TopoSZp+)',
                   'PL saddles with persistence >= 2 eps kept %', 'PL FP (SZp -> TopoSZp+)', 'PL FT (SZp -> TopoSZp+)', 'Axis saddles',
                   'Axis saddles that are PL saddles %'], rows,
                  'Freudenthal triangulation (6 neighbors in 2D, 14 in 3D), ties broken by index, as TTK. TopoSZp+ guarantees '
                  'refer to the axis rule; this table shows how the result looks under the PL definition.')

    open(os.path.join(R, 'summary.md'), 'w').write('\n'.join(out.md) + '\n')
    open(os.path.join(R, 'tables.tex'), 'w').write('\n'.join(out.tex) + '\n')
    print('\n'.join(out.md))
    print('\nWrote %s, %s' % (os.path.join(R, 'summary.md'), os.path.join(R, 'tables.tex')))

def plot_bitrate(R, rows, label, order):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[(r['dataset'], r['compressor'])][(r['mode'], r['eb'])].append(r)
    d = os.path.join(R, 'plots'); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'bitrate_curves.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['dataset', 'compressor', 'mode', 'eb', 'fields', 'bitrate', 'FN', 'FP', 'FT', 'total'])
        for (ds, comp), byb in agg.items():
            for (mode, eb), rs in byb.items():
                fn, fp, ft = (mean([f(r[k]) for r in rs]) for k in ('FN', 'FP', 'FT'))
                w.writerow([ds, label.get(comp, comp), mode, eb, len(rs), '%.4f' % mean([f(r['bitrate']) for r in rs]),
                            '%.2f' % fn, '%.2f' % fp, '%.2f' % ft, '%.2f' % (fn + fp + ft)])
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib not available: plot data written to %s/bitrate_curves.csv' % d)
        return
    for ds in sorted({k[0] for k in agg}):
        fig, axs = plt.subplots(1, 4, figsize=(13, 3.0))
        for ax, (key, title) in zip(axs, (('FN', 'FN'), ('FP', 'FP'), ('FT', 'FT'), ('total', 'Total'))):
            for comp in order:
                if (ds, comp) not in agg:
                    continue
                pts = []
                for (mode, eb), rs in agg[(ds, comp)].items():
                    vals = {k: mean([f(r[k]) for r in rs]) for k in ('FN', 'FP', 'FT')}
                    vals['total'] = vals['FN'] + vals['FP'] + vals['FT']
                    pts.append((mean([f(r['bitrate']) for r in rs]), vals[key]))
                pts.sort()
                ax.plot([p[0] for p in pts], [max(p[1], 0.5) for p in pts], marker='o', ms=3, label=label.get(comp, comp))
            ax.set_yscale('log'); ax.set_xlabel('bits per value'); ax.set_title(title); ax.grid(alpha=0.3)
        axs[0].set_ylabel('mean count per field (0 drawn at 0.5)'); axs[-1].legend(fontsize=7)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(d, 'bitrate_%s.%s' % (ds, ext)), dpi=150)
        plt.close(fig)
    print('plots in %s' % d)

if __name__ == '__main__':
    main()
