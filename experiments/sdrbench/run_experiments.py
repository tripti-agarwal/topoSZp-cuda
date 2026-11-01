#!/usr/bin/env python3
"""Run the TopoSZp+ experiments of the paper on SDRBench data.

    python3 run_experiments.py --data DATA_DIR [--out RESULTS_DIR] [--exp main,scaling,...]

DATA_DIR must contain manifest.csv (written by download_sdrbench.sh). Tools come
from --bin (default: ./bin, built by setup.sh). Every experiment appends one row
per run to RESULTS_DIR/<experiment>.csv; rerunning with the same --out skips the
runs already done, so an interrupted job can be resumed.

Experiments (--exp, comma-separated; default: all):
  main       every field x every bound (--bounds): TopoSZp+ on the CPU (OpenMP,
             --threads) and the GPU (CUDA) if available. Records compression and
             decompression time, compressed size of TopoSZp+ and of plain SZp,
             critical points of the original, of plain SZp quantization (before)
             and of the TopoSZp+ output (after): kept maxima / minima / saddles,
             FP, FN, FT, max error / eps, and whether CPU and GPU outputs are
             bit-identical.
  scaling    OpenMP thread sweep (--scaling-threads) on --scaling-fields fields
             per dataset at --scaling-bound: full compression and decompression
             (detection, ranking, encoding / decoding, restoration), plain SZp
             times, and whether the output is identical for every thread count.
  baselines  TopoSZp+, plain SZp, SZ3, SZ2, ZFP (and TTHRESH if built) on every
             field x --baseline-bounds; FN/FP/FT counted with the same checker.
  bitrate    the compressors of `baselines` over --bitrate-bounds on
             --bitrate-datasets (bit rate vs FN/FP/FT curves).
  pl         piecewise-linear (TTK) critical points and persistence of the
             TopoSZp+ output (--pl-fields fields per dataset, --pl-bound).
  vis        original / plain SZp / TopoSZp+ fields of --vis-field for the
             visualization figure (.f32 and, if VTK for Python is installed, .vti).

Bounds are written mode:value. abs:1e-3 is an absolute bound; rel:1e-3 is
1e-3 x (max - min) of each field. A bound is skipped (status format_limit) when
max|value| / bound exceeds the 32-bit bin index of the format.

Critical points use the paper's definition (4 face neighbors in 2D, 6 in 3D,
strict comparisons) via topo_check; FN = critical point that became regular,
FT = critical point with another type, FP = new critical point at a regular
point (each point is counted once).
"""
import argparse, csv, datetime, math, filecmp, hashlib, os, platform, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

# ----------------------------------------------------------------------------- columns
TOPO = ['max_orig', 'max_kept', 'min_orig', 'min_kept', 'sad_orig', 'sad_kept', 'FP', 'FN', 'FT', 'err']
COLS = {
    'main': ['dataset', 'field', 'dims', 'n', 'range', 'absmax', 'mode', 'eb', 'abs_eb', 'status', 'threads',
             'cpu_comp_s', 'cpu_decomp_s', 'toposzp_bytes', 'toposzp_ratio', 'szp_bytes', 'szp_ratio',
             'szp_comp_s', 'szp_decomp_s'] +
            [k + '_before' for k in TOPO] + [k + '_after' for k in TOPO] +
            ['gpu_comp_s', 'gpu_decomp_s', 'cpu_gpu_identical', 'gpu_FP', 'gpu_FN', 'gpu_FT', 'gpu_err', 'note'],
    'scaling': ['dataset', 'field', 'dims', 'n', 'mode', 'eb', 'abs_eb', 'status', 'threads', 'comp_s', 'decomp_s',
                'szp_comp_s', 'szp_decomp_s', 'err_after', 'identical_to_first', 'note'],
    'baselines': ['compressor', 'dataset', 'field', 'dims', 'n', 'mode', 'eb', 'abs_eb', 'status', 'bytes', 'ratio',
                  'bitrate', 'comp_s', 'decomp_s'] + TOPO + ['note'],
    'pl': ['dataset', 'field', 'dims', 'mode', 'eb', 'abs_eb', 'status', 'pl_extrema', 'pl_extrema_kept_final_pct',
           'pl_saddles', 'pl_saddles_kept_raw_pct', 'pl_saddles_kept_final_pct', 'pl_saddles_kept_final_1cell_pct',
           'pl_saddles_pers_ge_2eb_kept_final_pct', 'axis_saddles', 'axis_saddles_that_are_pl_pct',
           'total_orig', 'kept_raw', 'FN_raw', 'FT_raw', 'FP_raw', 'kept_final', 'FN_final', 'FT_final', 'FP_final',
           'err_final', 'note'],
}
COLS['bitrate'] = COLS['baselines']
KEYS = {'main': ('dataset', 'field', 'mode', 'eb'), 'scaling': ('dataset', 'field', 'mode', 'eb', 'threads'),
        'baselines': ('compressor', 'dataset', 'field', 'mode', 'eb'),
        'bitrate': ('compressor', 'dataset', 'field', 'mode', 'eb'), 'pl': ('dataset', 'field', 'mode', 'eb')}
DONE = ('ok', 'format_limit')   # statuses that are not retried on resume

# ----------------------------------------------------------------------------- helpers
class Ctx:
    pass

def log(c, msg):
    line = '[%s] %s' % (datetime.datetime.now().strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(os.path.join(c.out, 'run.log'), 'a') as f:
        f.write(line + '\n')

def run(c, cmd, logname, env=None, timeout=None):
    """Run a command, append its output to logs/<logname>.log; return (rc, stdout+stderr, seconds)."""
    cmd = [str(x) for x in cmd]
    e = dict(os.environ)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e,
                           timeout=timeout or c.timeout, cwd=c.work)
        out, rc = p.stdout.decode('utf-8', 'replace'), p.returncode
    except subprocess.TimeoutExpired as x:
        out, rc = (x.stdout or b'').decode('utf-8', 'replace') + '\nTIMEOUT\n', 124
    dt = time.perf_counter() - t0
    path = os.path.join(c.out, 'logs', logname + '.log')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write('$ %s\n%s\n[rc=%d, %.3f s]\n\n' % (' '.join(map(str, cmd)), out, rc, dt))
    return rc, out, dt

def num(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None

def first_error(out):
    for line in out.splitlines():
        if re.search(r'error|exceeds|fail|cannot|TIMEOUT', line, re.I):
            return line.strip()[:200]
    return out.strip().splitlines()[-1][:200] if out.strip() else ''

class Table:
    """CSV with resume support."""
    def __init__(self, c, name):
        self.path, self.cols, self.key = os.path.join(c.out, name + '.csv'), COLS[name], KEYS[name]
        self.done = set()
        if os.path.exists(self.path):
            with open(self.path) as f:
                for r in csv.DictReader(f):
                    if r.get('status') in DONE:
                        self.done.add(tuple(r[k] for k in self.key))
        else:
            with open(self.path, 'w', newline='') as f:
                csv.writer(f).writerow(self.cols)
    def has(self, **kv):
        return tuple(str(kv[k]) for k in self.key) in self.done
    def add(self, row):
        with open(self.path, 'a', newline='') as f:
            csv.DictWriter(f, self.cols, extrasaction='ignore').writerow(row)
        if row.get('status') in DONE:
            self.done.add(tuple(str(row[k]) for k in self.key))

def fmt(x, p='%.6g'):
    return '' if x is None else (p % x if isinstance(x, float) else str(x))

# ----------------------------------------------------------------------------- data
def load_manifest(c):
    rows = list(csv.DictReader(open(os.path.join(c.data, 'manifest.csv'))))
    fields = []
    for r in rows:
        if c.datasets and r['dataset'] not in c.datasets:
            continue
        dims = [int(r['d1']), int(r['d2'])] + ([int(r['d3'])] if r.get('d3') else [])
        fields.append({'dataset': r['dataset'], 'field': r['field'], 'path': r['path'], 'dims': dims,
                       'n': math.prod(dims)})
    by = {}
    for f in fields:
        by.setdefault(f['dataset'], []).append(f)
    for d in by:
        by[d].sort(key=lambda f: f['field'])
        if c.fields_per_dataset:
            by[d] = by[d][:c.fields_per_dataset]
    return by

def stats(c, f):
    """Value range and max |value| (cached in fstats.csv)."""
    key = f['path']
    if key in c.stats:
        return c.stats[key]
    rc, out, _ = run(c, [c.tool('fstats'), f['path']], 'fstats')
    s = {k: float(num(k + r'=([-0-9.eE+naif]+)', out)) for k in ('range', 'absmax')}
    c.stats[key] = s
    with open(os.path.join(c.out, 'fstats.csv'), 'a', newline='') as fh:
        csv.writer(fh).writerow([key, s['range'], s['absmax']])
    return s

def resolve(c, f, bound):
    """'abs:1e-3' / 'rel:1e-3' -> (mode, eb string, absolute bound, status or None)."""
    mode, eb = bound.split(':')
    s = stats(c, f)
    a = float(eb) if mode == 'abs' else float(eb) * (s['range'] if s['range'] > 0 else 1.1920929e-07)
    st = 'format_limit' if s['absmax'] / a >= 2.0 ** 31 else None
    return mode, eb, float('%.9g' % a), st

def dimargs(f):
    return [str(d) for d in f['dims']]

def fast_first(f):   # dimension order for SZ3 / SZ2 / ZFP / TTHRESH: fastest-varying first
    return [str(d) for d in reversed(f['dims'])]

def link(c, f, tag):
    """Symlink the field into the work folder (tools write their output next to the input)."""
    p = os.path.join(c.work, '%s__%s__%s.f32' % (f['dataset'], f['field'], tag))
    for q in (p, p + '.SZp', p + '.SZp.out'):
        if os.path.lexists(q):
            os.remove(q)
    os.symlink(f['path'], p)
    return p

def cleanup(*paths):
    for p in paths:
        if p and os.path.lexists(p):
            os.remove(p)

def topo(c, f, out_path, abs_eb, logname):
    """topo_check original vs output -> {'before': {...}, 'after': {...}} (or None)."""
    rc, out, _ = run(c, [c.tool('topo_check'), f['path'], out_path, repr(abs_eb)] + dimargs(f), logname,
                     env={'OMP_NUM_THREADS': c.nproc})
    res = {}
    for stage in ('before', 'after'):
        m = re.search(r'^TOPOROW %s (.+)$' % stage, out, re.M)
        if m:
            v = m.group(1).split()
            res[stage] = dict(zip(TOPO, [int(x) for x in v[:9]] + [float(v[9])]))
    return res if len(res) == 2 else None

def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()

# ----------------------------------------------------------------------------- compressors
def toposzp_cpu(c, f, abs_eb, threads, logname, keep_output=False):
    """TopoSZp+ OpenMP compress + decompress (min time over --reps). Returns dict."""
    p = link(c, f, 'cpu')
    env = {'OMP_NUM_THREADS': threads}
    best = {'comp_s': None, 'decomp_s': None}
    r = {'status': 'ok', 'note': ''}
    for rep in range(c.reps):
        rc, out, _ = run(c, [c.tool('toposzp_comp'), p, c.bs, repr(abs_eb)] + dimargs(f), logname, env)
        tc = num(r'compression time=([0-9.eE+-]+)', out)
        if rc != 0 or tc is None or not os.path.exists(p + '.SZp'):
            cleanup(p, p + '.SZp', p + '.SZp.out')
            return {'status': 'compress_failed', 'note': first_error(out)}
        r['bytes'] = os.path.getsize(p + '.SZp')
        rc, out, _ = run(c, [c.tool('toposzp_decomp'), p + '.SZp', f['n'], c.bs, repr(abs_eb)] + dimargs(f), logname, env)
        td = num(r'decompression time=([0-9.eE+-]+)', out)
        if rc != 0 or td is None or not os.path.exists(p + '.SZp.out'):
            cleanup(p, p + '.SZp', p + '.SZp.out')
            return {'status': 'decompress_failed', 'note': first_error(out)}
        best['comp_s'] = tc if best['comp_s'] is None else min(best['comp_s'], tc)
        best['decomp_s'] = td if best['decomp_s'] is None else min(best['decomp_s'], td)
    r.update(best)
    r['output'] = p + '.SZp.out'
    cleanup(p + '.SZp')
    if not keep_output:
        r['_cleanup'] = [p, p + '.SZp.out']
    else:
        r['_cleanup'] = [p]
    return r

def plain_szp(c, f, abs_eb, threads, logname, output=None):
    best = None
    for rep in range(c.reps):
        cmd = [c.tool('plain_szp'), f['path'], repr(abs_eb), c.bs] + ([output] if output and rep == c.reps - 1 else [])
        rc, out, _ = run(c, cmd, logname, {'OMP_NUM_THREADS': threads})
        b = num(r'bytes=(\d+)', out, int)
        if rc != 0 or b is None:
            return {'status': 'failed', 'note': first_error(out)}
        cur = {'bytes': b, 'comp_s': num(r'comp_s=([0-9.eE+-]+)', out), 'decomp_s': num(r'decomp_s=([0-9.eE+-]+)', out)}
        if best is None:
            best = cur
        else:
            best['comp_s'] = min(best['comp_s'], cur['comp_s']); best['decomp_s'] = min(best['decomp_s'], cur['decomp_s'])
    best['status'] = 'ok'
    return best

def toposzp_gpu(c, f, abs_eb, logname):
    exe = c.tool('test_cuda_topology_3d' if len(f['dims']) == 3 else 'test_cuda_topology')
    dump = os.path.join(c.work, 'gpu_%s__%s.out' % (f['dataset'], f['field']))
    cleanup(dump)
    best = {'comp_s': None, 'decomp_s': None}
    for rep in range(c.gpu_reps):
        rc, out, _ = run(c, [exe, f['path']] + dimargs(f) + [repr(abs_eb), c.bs], logname,
                         {'TOPOSZP_DUMP': dump, 'OMP_NUM_THREADS': c.threads})
        m = re.search(r'^TIMING\s+(.*)$', out, re.M)
        if not m or not os.path.exists(dump):
            return {'status': 'gpu_failed', 'note': first_error(out)}
        kv = dict(x.split('=') for x in m.group(1).split() if '=' in x)
        tc, td = float(kv['comp_s']), float(kv['decomp_s'])
        best['comp_s'] = tc if best['comp_s'] is None else min(best['comp_s'], tc)
        best['decomp_s'] = td if best['decomp_s'] is None else min(best['decomp_s'], td)
    best.update({'status': 'ok', 'output': dump})
    return best

def baseline(c, name, f, abs_eb, logname):
    """Compress + decompress with one compressor. Returns dict with bytes, times, output path."""
    if name == 'toposzp':
        r = toposzp_cpu(c, f, abs_eb, c.threads, logname, keep_output=True)
        if r['status'] != 'ok':
            return r
        r['_cleanup'] = r['_cleanup'] + [r['output']]
        return r
    if name == 'szp':
        out = os.path.join(c.work, 'szp__%s__%s.out' % (f['dataset'], f['field']))
        r = plain_szp(c, f, abs_eb, c.threads, logname, output=out)
        if r['status'] == 'ok':
            r.update({'output': out, '_cleanup': [out]})
        return r
    p = link(c, f, name)
    cmp_, dec = p + '.cmp', p + '.dec'
    nd = '-%d' % len(f['dims'])
    eb = repr(abs_eb)
    if name == 'sz3':
        cc = [c.tool('sz3'), '-f', '-i', p, '-z', cmp_, '-M', 'ABS', eb, nd] + fast_first(f)
        dc = [c.tool('sz3'), '-f', '-z', cmp_, '-o', dec, '-M', 'ABS', eb, nd] + fast_first(f)
    elif name == 'sz':
        cmp_, dec = p + '.sz', p + '.sz.out'
        cc = [c.tool('sz'), '-z', '-f', '-i', p, '-M', 'ABS', '-A', eb, nd] + fast_first(f)
        dc = [c.tool('sz'), '-x', '-f', '-s', cmp_, nd] + fast_first(f)
    elif name == 'zfp':
        cc = [c.tool('zfp'), '-f', nd] + fast_first(f) + ['-a', eb, '-i', p, '-z', cmp_]
        dc = [c.tool('zfp'), '-f', nd] + fast_first(f) + ['-a', eb, '-z', cmp_, '-o', dec]
    elif name == 'tthresh':   # note: -e is TTHRESH's relative (Frobenius) target, not a pointwise bound
        dd = fast_first(f) + (['1'] if len(f['dims']) == 2 else [])
        cc = [c.tool('tthresh'), '-i', p, '-t', 'float', '-s'] + dd + ['-e', eb, '-c', cmp_]
        dc = [c.tool('tthresh'), '-c', cmp_, '-o', dec]
    else:
        raise ValueError(name)
    env = {'OMP_NUM_THREADS': c.threads}
    rc, out, tc = run(c, cc, logname, env)
    if rc != 0 or not os.path.exists(cmp_):
        cleanup(p, cmp_, dec)
        return {'status': 'compress_failed', 'note': first_error(out)}
    rc, out, td = run(c, dc, logname, env)
    if rc != 0 or not os.path.exists(dec) or os.path.getsize(dec) != 4 * f['n']:
        cleanup(p, cmp_, dec)
        return {'status': 'decompress_failed', 'note': first_error(out)}
    return {'status': 'ok', 'bytes': os.path.getsize(cmp_), 'comp_s': tc, 'decomp_s': td, 'output': dec,
            '_cleanup': [p, cmp_, dec]}

# ----------------------------------------------------------------------------- experiments
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

def exp_compare(c, by, name, bounds, datasets=None):
    t = Table(c, name)
    for ds, fields in by.items():
        if datasets and ds not in datasets:
            continue
        for f in fields[:(c.bitrate_fields if name == 'bitrate' else c.baseline_fields) or None]:
            for bound in bounds:
                mode, eb, a, st = resolve(c, f, bound)
                for comp in c.compressors:
                    if t.has(compressor=comp, dataset=ds, field=f['field'], mode=mode, eb=eb):
                        continue
                    row = {'compressor': comp, 'dataset': ds, 'field': f['field'], 'dims': 'x'.join(map(str, f['dims'])),
                           'n': f['n'], 'mode': mode, 'eb': eb, 'abs_eb': a}
                    if st and comp in ('toposzp', 'szp'):
                        t.add(dict(row, status=st)); continue
                    lg = '%s/%s/%s/%s_%s_%s' % (name, comp, ds, f['field'], mode, eb)
                    r = baseline(c, comp, f, a, lg)
                    if r['status'] != 'ok':
                        t.add(dict(row, status=r['status'], note=r.get('note', ''))); log(c, '%s %s %s/%s %s: %s' % (name, comp, ds, f['field'], bound, r['status'])); continue
                    tp = topo(c, f, r['output'], a, lg)
                    cleanup(*r.get('_cleanup', []))
                    if not tp:
                        t.add(dict(row, status='check_failed')); continue
                    row.update(tp['after'])
                    row.update({'status': 'ok', 'bytes': r['bytes'], 'ratio': '%.4f' % (4.0 * f['n'] / r['bytes']),
                                'bitrate': '%.4f' % (32.0 * r['bytes'] / (4.0 * f['n'])),
                                'comp_s': r['comp_s'], 'decomp_s': r['decomp_s'],
                                'note': 'tthresh -e is not a pointwise bound' if comp == 'tthresh' else ''})
                    t.add(row)
                    log(c, '%s %s %s/%s %s: bitrate %s  FN %d FP %d FT %d  err %.3f eps' % (
                        name, comp, ds, f['field'], bound, row['bitrate'], row['FN'], row['FP'], row['FT'], row['err']))

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

def exp_vis(c, by):
    ds, fld = c.vis_field.split('/')
    f = next((x for x in by.get(ds, []) if x['field'] == fld), None)
    if f is None:
        log(c, 'vis: field %s not found in the manifest' % c.vis_field); return
    mode, eb, a, st = resolve(c, f, c.vis_bound)
    d = os.path.join(c.out, 'vis'); os.makedirs(d, exist_ok=True)
    base = os.path.join(d, '%s_%s_%s%s' % (ds, fld, mode, eb))
    shutil.copyfile(f['path'], base + '_original.f32')
    plain_szp(c, f, a, c.threads, 'vis', output=base + '_szp.f32')
    r = toposzp_cpu(c, f, a, c.threads, 'vis', keep_output=True)
    if r['status'] == 'ok':
        shutil.copyfile(r['output'], base + '_toposzp.f32'); cleanup(*r['_cleanup'])
    for k in ('original', 'szp', 'toposzp'):
        p = base + '_%s.f32' % k
        if os.path.exists(p) and k != 'original':
            topo(c, f, p, a, 'vis')
        dims = f['dims']
        nx, ny = dims[-1], dims[-2]
        cmd = [sys.executable, os.path.join(REPO, 'example', 'dat_to_vti.py'), p, '--nx', str(nx), '--ny', str(ny),
               '--scalar-name', fld, '-o', p[:-4] + '.vti'] + (['--nz', str(dims[0])] if len(dims) == 3 else [])
        rc, out, _ = run(c, cmd, 'vis')
        if rc != 0:
            log(c, 'vis: no .vti for %s (VTK for Python missing?); the raw .f32 files are in %s' % (k, d)); break
    log(c, 'vis: %s at %s -> %s_{original,szp,toposzp}.f32 (topo_check results in logs/vis.log)' % (c.vis_field, c.vis_bound, base))

# ----------------------------------------------------------------------------- main
def machine_info(c):
    def sh(cmd):
        try:
            p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)
            return p.stdout.decode().strip() if p.returncode == 0 else 'n/a'
        except Exception:
            return ''
    info = ['date: %s' % datetime.datetime.now().isoformat(), 'host: %s' % platform.node(),
            'cpu: %s' % sh("lscpu | grep -m1 'Model name' | sed 's/.*: *//'"),
            'cores (nproc): %s' % c.nproc, 'sockets / NUMA: %s' % sh("lscpu | grep -E '^(Socket|NUMA node)\\(s\\)' | tr -s ' ' | tr '\\n' ';'"),
            'memory: %s' % sh("free -g | awk '/Mem/{print $2\" GB\"}'"),
            'gpu: %s' % sh('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'),
            'nvcc: %s' % sh('nvcc --version | tail -1'), 'gcc: %s' % sh('g++ --version | head -1'),
            'repo: %s (%s)' % (sh('git -C %s rev-parse --short HEAD' % REPO), sh('git -C %s branch --show-current' % REPO)),
            'settings: threads=%s block=%s reps=%s gpu_reps=%s bounds=%s' % (c.threads, c.bs, c.reps, c.gpu_reps, ','.join(c.bounds))]
    with open(os.path.join(c.out, 'machine.txt'), 'w') as fh:
        fh.write('\n'.join(info) + '\n')
    for l in info:
        log(c, l)

def main():
    nproc = os.cpu_count() or 1
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='folder with manifest.csv (download_sdrbench.sh)')
    ap.add_argument('--out', default=None, help='results folder (default: results/<date>); reuse it to resume')
    ap.add_argument('--bin', default=os.path.join(HERE, 'bin'), help='tools folder (setup.sh)')
    ap.add_argument('--work', default=None, help='scratch folder for outputs (default: <out>/work); use fast local disk')
    ap.add_argument('--exp', default='main,scaling,baselines,bitrate,pl,vis')
    ap.add_argument('--datasets', default='', help='comma-separated dataset names (default: all in the manifest)')
    ap.add_argument('--fields-per-dataset', type=int, default=0, help='limit fields per dataset (0 = all)')
    ap.add_argument('--bounds', default='abs:1e-3,abs:1e-4,abs:1e-5,rel:1e-3,rel:1e-4,rel:1e-5')
    ap.add_argument('--threads', type=int, default=nproc, help='OpenMP threads for main/baselines (default: all cores)')
    ap.add_argument('--gpu', choices=['auto', 'on', 'off'], default='auto')
    ap.add_argument('--block-size', type=int, default=64)
    ap.add_argument('--reps', type=int, default=3, help='timed repetitions; the minimum is reported')
    ap.add_argument('--gpu-reps', type=int, default=1, help='timed repetitions of the CUDA program (it also runs an OpenMP comparison internally)')
    ap.add_argument('--timeout', type=int, default=3600, help='seconds per command')
    ap.add_argument('--scaling-threads', default=','.join(str(t) for t in [1, 2, 4, 8, 16, 32, 64, 128] if t <= nproc))
    ap.add_argument('--scaling-bound', default='rel:1e-3')
    ap.add_argument('--scaling-fields', type=int, default=5, help='fields per dataset (0 = all)')
    ap.add_argument('--compressors', default='toposzp,szp,sz3,sz,zfp,tthresh', help='baselines/bitrate; missing tools are skipped')
    ap.add_argument('--baseline-bounds', default='abs:1e-3,abs:1e-4,abs:1e-5,rel:1e-3,rel:1e-4,rel:1e-5')
    ap.add_argument('--baseline-fields', type=int, default=0, help='fields per dataset for baselines (0 = all)')
    ap.add_argument('--bitrate-datasets', default='CESM-ATM-2D')
    ap.add_argument('--bitrate-bounds', default='abs:1e-2,abs:5e-3,abs:1e-3,abs:5e-4,abs:1e-4,abs:5e-5,abs:1e-5,abs:5e-6,abs:1e-6')
    ap.add_argument('--bitrate-fields', type=int, default=0, help='fields per dataset for the bit-rate sweep (0 = all)')
    ap.add_argument('--pl-bound', default='rel:1e-3')
    ap.add_argument('--pl-fields', type=int, default=3, help='fields per dataset for the PL check (0 = all)')
    ap.add_argument('--vis-field', default='CESM-ATM-2D/CLDHGH_1_1800_3600')
    ap.add_argument('--vis-bound', default='abs:1e-3')
    a = ap.parse_args()

    c = Ctx()
    c.data = os.path.abspath(a.data)
    c.out = os.path.abspath(a.out or os.path.join(HERE, 'results', datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    c.work = os.path.abspath(a.work or os.path.join(c.out, 'work'))
    os.makedirs(c.out, exist_ok=True); os.makedirs(c.work, exist_ok=True)
    c.bin = os.path.abspath(a.bin)
    c.tool = lambda n: os.path.join(c.bin, n)
    c.nproc, c.threads, c.bs, c.reps, c.timeout = nproc, a.threads, str(a.block_size), a.reps, a.timeout
    c.gpu_reps = a.gpu_reps
    c.datasets = [x for x in a.datasets.split(',') if x]
    c.fields_per_dataset = a.fields_per_dataset
    c.bounds = [x for x in a.bounds.split(',') if x]
    c.scaling_threads = [int(x) for x in a.scaling_threads.split(',') if x]
    c.scaling_bound, c.scaling_fields = a.scaling_bound, a.scaling_fields
    c.baseline_fields, c.bitrate_fields, c.pl_fields = a.baseline_fields, a.bitrate_fields, a.pl_fields
    c.pl_bound, c.vis_field, c.vis_bound = a.pl_bound, a.vis_field, a.vis_bound
    envsh = os.path.join(c.bin, 'env.sh')   # shared-library path of SZ2 / SZ3
    if os.path.exists(envsh):
        m = re.search(r'LD_LIBRARY_PATH=([^:$]+):([^:$]+)', open(envsh).read())
        if m:
            os.environ['LD_LIBRARY_PATH'] = '%s:%s:%s' % (m.group(1), m.group(2), os.environ.get('LD_LIBRARY_PATH', ''))
    for need in ('toposzp_comp', 'toposzp_decomp', 'plain_szp', 'topo_check', 'pl_topo_check', 'fstats'):
        if not os.access(c.tool(need), os.X_OK):
            sys.exit('missing tool %s: run setup.sh first' % c.tool(need))
    have_gpu = os.access(c.tool('test_cuda_topology'), os.X_OK) and shutil.which('nvidia-smi') and \
        subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    c.gpu = a.gpu == 'on' or (a.gpu == 'auto' and bool(have_gpu))
    if a.gpu == 'on' and not have_gpu:
        sys.exit('--gpu on, but the CUDA programs or the GPU are not available')
    c.compressors = [x for x in a.compressors.split(',') if x in ('toposzp', 'szp') or os.access(c.tool(x), os.X_OK)]
    c.stats = {}
    sp = os.path.join(c.out, 'fstats.csv')
    if os.path.exists(sp):
        for r in csv.reader(open(sp)):
            c.stats[r[0]] = {'range': float(r[1]), 'absmax': float(r[2])}

    machine_info(c)
    by = load_manifest(c)
    log(c, 'datasets: %s' % ', '.join('%s (%d fields)' % (d, len(v)) for d, v in by.items()))
    log(c, 'GPU: %s; compressors: %s' % ('yes' if c.gpu else 'no', ', '.join(c.compressors)))
    exps = [x for x in a.exp.split(',') if x]
    for e in exps:
        log(c, '==== experiment: %s' % e)
        if e == 'main':
            exp_main(c, by)
        elif e == 'scaling':
            exp_scaling(c, by)
        elif e == 'baselines':
            exp_compare(c, by, 'baselines', [x for x in a.baseline_bounds.split(',') if x])
        elif e == 'bitrate':
            exp_compare(c, by, 'bitrate', [x for x in a.bitrate_bounds.split(',') if x],
                        [x for x in a.bitrate_datasets.split(',') if x])
        elif e == 'pl':
            exp_pl(c, by)
        elif e == 'vis':
            exp_vis(c, by)
        else:
            log(c, 'unknown experiment %s' % e)
    log(c, 'done. Results in %s; tables: python3 %s %s' % (c.out, os.path.join(HERE, 'summarize.py'), c.out))

if __name__ == '__main__':
    main()
