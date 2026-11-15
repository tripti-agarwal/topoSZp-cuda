#!/usr/bin/env python3
"""Cut small blocks out of real SDRBench fields, for developing the algorithm
on a machine without the full data.

    python3 make_crops.py DATA_DIR OUT_DIR [--fields 3] [--size 64,160,160] [--size2d 600,600]

For every dataset in DATA_DIR/manifest.csv, takes the first --fields fields
(plus a few named ones) and writes a centred block of at most --size points
per axis (2D: the last two numbers) as float32, with a manifest.csv in the
same format, so the experiment scripts run on the crops unchanged.
"""
import argparse, csv, os
import numpy as np

EXTRA = {'CESM-ATM-3D': ['T_', 'Q_', 'U_'], 'Hurricane': ['Uf48', 'TCf48', 'Pf48'],
         'NYX': ['dark_matter_density', 'temperature', 'velocity_x'], 'SCALE-LETKF': ['T-', 'U-', 'QV-'],
         'CESM-ATM-2D': ['CLDHGH', 'TS_', 'PSL']}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data'); ap.add_argument('out')
    ap.add_argument('--fields', type=int, default=3)
    ap.add_argument('--size', default='64,160,160', help='3D block')
    ap.add_argument('--size2d', default='600,600', help='2D block')
    a = ap.parse_args()
    size = [int(x) for x in a.size.split(',')]
    size2 = [int(x) for x in a.size2d.split(',')]
    rows = list(csv.DictReader(open(os.path.join(a.data, 'manifest.csv'))))
    by = {}
    for r in rows:
        by.setdefault(r['dataset'], []).append(r)
    out_rows = []
    for ds, rs in by.items():
        pick = rs[:a.fields] + [r for r in rs[a.fields:] if any(r['field'].startswith(p) for p in EXTRA.get(ds, []))]
        os.makedirs(os.path.join(a.out, ds), exist_ok=True)
        for r in pick:
            dims = [int(r[k]) for k in ('d1', 'd2', 'd3') if r.get(k) and int(r[k]) > 0]
            want = size if len(dims) == 3 else size2
            m = np.memmap(r['path'], dtype='<f4', mode='r', shape=tuple(dims))
            sl = tuple(slice((n - min(n, w)) // 2, (n - min(n, w)) // 2 + min(n, w)) for n, w in zip(dims, want))
            c = np.ascontiguousarray(m[sl])
            p = os.path.join(a.out, ds, r['field'] + '.crop.f32')
            c.tofile(p)
            d = list(c.shape) + [0] * (3 - c.ndim)
            out_rows.append({'dataset': ds, 'field': r['field'], 'path': os.path.abspath(p),
                             'd1': d[0], 'd2': d[1], 'd3': d[2]})
            print('%-12s %-28s %s -> %s  (%.1f MB)' % (ds, r['field'], 'x'.join(map(str, dims)),
                                                       'x'.join(map(str, c.shape)), c.nbytes / 1e6))
    with open(os.path.join(a.out, 'manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, ['dataset', 'field', 'path', 'd1', 'd2', 'd3']); w.writeheader(); w.writerows(out_rows)
    print('wrote %d crops to %s' % (len(out_rows), a.out))

if __name__ == '__main__':
    main()
