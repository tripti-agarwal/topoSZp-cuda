"""Shared code of the TopoSZp+ SDRBench experiments (exp_*.py, run_experiments.py).

Holds the result-table definitions, command execution and logging, the data
manifest, error-bound resolution, the compressor wrappers (TopoSZp+ CPU / GPU,
plain SZp, SZ3, SZ2, ZFP, TTHRESH) and the topology check, plus the options and
setup shared by every experiment script.
"""
from collections import OrderedDict
import argparse, csv, datetime, filecmp, glob, hashlib, math, os, platform, re, shutil, subprocess, sys, time

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
def usable_cpus():
    """(logical CPUs, physical cores) this job may use. os.cpu_count() counts the
    whole node, but a batch job (e.g. one GPU of a Swing node) gets only part of it."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:
        cpus = list(range(os.cpu_count() or 1))
    cores = set()
    for i in cpus:
        try:
            base = '/sys/devices/system/cpu/cpu%d/topology/' % i
            cores.add((open(base + 'physical_package_id').read().strip(), open(base + 'core_id').read().strip()))
        except OSError:
            cores.add(i)
    return len(cpus), max(1, len(cores))

class Ctx:
    pass

def log(c, msg):
    line = '[%s] %s' % (datetime.datetime.now().strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(os.path.join(c.out, 'run%s.log' % ('.' + c.tag if c.tag else '')), 'a') as f:
        f.write(line + '\n')

def run(c, cmd, logname, env=None, timeout=None):
    """Run a command, append its output to logs/[<tag>/]<logname>.log; return (rc, stdout+stderr, seconds)."""
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
    path = os.path.join(c.out, 'logs', c.tag, logname + '.log')   # logs/<tag>/ for a tagged job
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

def result_files(out, name):
    """<name>.csv and <name>.<tag>.csv of every job, in a fixed order."""
    p = os.path.join(out, name + '.csv')
    return ([p] if os.path.exists(p) else []) + sorted(glob.glob(os.path.join(out, name + '.*.csv')))

def read_results(out, name):
    """All rows of an experiment from every job; for a run repeated after a
    failure, only its latest row is kept."""
    rows = OrderedDict()
    for p in result_files(out, name):
        with open(p) as f:
            for r in csv.DictReader(f):
                rows[tuple(r.get(k, '') for k in KEYS[name])] = r
    return list(rows.values())

class Table:
    """CSV with resume support."""
    def __init__(self, c, name):
        fname = name + ('.' + c.tag if c.tag else '') + '.csv'
        self.path, self.cols, self.key = os.path.join(c.out, fname), COLS[name], KEYS[name]
        self.done = set()
        # Runs finished by any job (untagged or tagged) count as done, so a
        # per-dataset job can resume a run that started as one big job.
        for p in result_files(c.out, name):
            with open(p) as f:
                for r in csv.DictReader(f):
                    if r.get('status') in DONE:
                        self.done.add(tuple(r[k] for k in self.key))
        if not os.path.exists(self.path):
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
    with open(os.path.join(c.out, 'fstats%s.csv' % ('.' + c.tag if c.tag else '')), 'a', newline='') as fh:
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
                     env={'OMP_NUM_THREADS': c.ncores})
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

# ----------------------------------------------------------------------------- compare (baselines, bitrate)
def compare(c, by, name, bounds, datasets=None):
    t = Table(c, name)
    for ds, fields in by.items():
        if datasets and ds not in datasets:
            continue
        for f in fields[:c.cmp_fields or None]:
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

# ----------------------------------------------------------------------------- setup
def machine_info(c):
    def sh(cmd):
        try:
            p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)
            return p.stdout.decode().strip() if p.returncode == 0 else 'n/a'
        except Exception:
            return ''
    info = ['date: %s' % datetime.datetime.now().isoformat(), 'host: %s' % platform.node(),
            'cpu: %s' % sh("lscpu | grep -m1 'Model name' | sed 's/.*: *//'"),
            'CPUs usable by this job: %s logical, %s physical cores (node: %s logical)' % (c.nproc, c.ncores, os.cpu_count()),
            'OMP_PROC_BIND=%s OMP_PLACES=%s' % (os.environ.get('OMP_PROC_BIND', ''), os.environ.get('OMP_PLACES', '')), 'sockets / NUMA: %s' % sh("lscpu | grep -E '^(Socket|NUMA node)\\(s\\)' | tr -s ' ' | tr '\\n' ';'"),
            'memory: %s' % sh("free -g | awk '/Mem/{print $2\" GB\"}'"),
            'gpu: %s' % sh('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'),
            'nvcc: %s' % sh('nvcc --version | tail -1'), 'gcc: %s' % sh('g++ --version | head -1'),
            'repo: %s (%s)' % (sh('git -C %s rev-parse --short HEAD' % REPO), sh('git -C %s branch --show-current' % REPO)),
            'settings: threads=%s block=%s reps=%s gpu_reps=%s' % (c.threads, c.bs, c.reps, c.gpu_reps)]
    with open(os.path.join(c.out, 'machine%s.txt' % ('.' + c.tag if c.tag else '')), 'w') as fh:
        fh.write('\n'.join(info) + '\n')
    for l in info:
        log(c, l)

def common_args(ap):
    """Options shared by every experiment script."""
    nproc = os.cpu_count() or 1
    ap.add_argument('--data', required=True, help='folder with manifest.csv (download_sdrbench.sh)')
    ap.add_argument('--out', default=None, help='results folder (default: results/<date>); reuse it to resume')
    ap.add_argument('--bin', default=os.path.join(HERE, 'bin'), help='tools folder (setup.sh)')
    ap.add_argument('--work', default=None, help='scratch folder for outputs (default: <out>/work); use fast local disk')
    ap.add_argument('--tag', default='', help='suffix for this job\'s files (main.<tag>.csv, ...), so that jobs running '
                    'at the same time write separate files; summarize.py merges them')
    ap.add_argument('--datasets', default='', help='comma-separated dataset names (default: all in the manifest)')
    ap.add_argument('--fields-per-dataset', type=int, default=0, help='limit fields per dataset (0 = all)')
    ap.add_argument('--threads', type=int, default=nproc, help='OpenMP threads (default: all cores)')
    ap.add_argument('--block-size', type=int, default=64)
    ap.add_argument('--reps', type=int, default=3, help='timed repetitions; the minimum is reported')
    ap.add_argument('--timeout', type=int, default=3600, help='seconds per command')

def context(a, need_gpu=False, need_compressors=None):
    """Build the run context from parsed options; returns (c, fields by dataset)."""
    c = Ctx()
    c.nproc, c.ncores = usable_cpus()
    # Pin OpenMP threads (one per core first) so that timings are repeatable; set
    # OMP_PROC_BIND / OMP_PLACES yourself to override.
    os.environ.setdefault('OMP_PROC_BIND', 'close')
    os.environ.setdefault('OMP_PLACES', 'cores')
    c.data = os.path.abspath(a.data)
    c.out = os.path.abspath(a.out or os.path.join(HERE, 'results', datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    c.work = os.path.abspath(a.work or os.path.join(c.out, 'work'))
    c.tag = a.tag
    if c.tag:
        c.work = os.path.join(c.work, c.tag)
    os.makedirs(c.out, exist_ok=True); os.makedirs(c.work, exist_ok=True)
    c.bin = os.path.abspath(a.bin)
    c.tool = lambda n: os.path.join(c.bin, n)
    c.threads, c.bs, c.reps, c.timeout = a.threads, str(a.block_size), a.reps, a.timeout
    c.gpu_reps = getattr(a, 'gpu_reps', 1)
    c.datasets = [x for x in a.datasets.split(',') if x]
    c.fields_per_dataset = a.fields_per_dataset
    envsh = os.path.join(c.bin, 'env.sh')   # shared-library path of SZ2 / SZ3
    if os.path.exists(envsh):
        m = re.search(r'LD_LIBRARY_PATH=([^:$]+):([^:$]+)', open(envsh).read())
        if m:
            os.environ['LD_LIBRARY_PATH'] = '%s:%s:%s' % (m.group(1), m.group(2), os.environ.get('LD_LIBRARY_PATH', ''))
    for need in ('toposzp_comp', 'toposzp_decomp', 'plain_szp', 'topo_check', 'pl_topo_check', 'fstats'):
        if not os.access(c.tool(need), os.X_OK):
            sys.exit('missing tool %s: run setup.sh first' % c.tool(need))
    c.gpu = False
    if need_gpu:
        have_gpu = os.access(c.tool('test_cuda_topology'), os.X_OK) and shutil.which('nvidia-smi') and \
            subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        mode = getattr(a, 'gpu', 'auto')
        c.gpu = mode == 'on' or (mode == 'auto' and bool(have_gpu))
        if mode == 'on' and not have_gpu:
            sys.exit('--gpu on, but the CUDA programs or the GPU are not available')
    c.compressors = []
    if need_compressors:
        c.compressors = [x for x in need_compressors.split(',') if x in ('toposzp', 'szp') or os.access(c.tool(x), os.X_OK)]
    c.stats = {}
    for sp in glob.glob(os.path.join(c.out, 'fstats*.csv')):   # value ranges cached by earlier / parallel jobs
        for r in csv.reader(open(sp)):
            c.stats[r[0]] = {'range': float(r[1]), 'absmax': float(r[2])}
    c.bounds = getattr(a, 'bounds', '').split(',') if getattr(a, 'bounds', '') else []
    machine_info(c)
    by = load_manifest(c)
    log(c, 'datasets: %s' % ', '.join('%s (%d fields)' % (d, len(v)) for d, v in by.items()))
    if need_gpu:
        log(c, 'GPU: %s' % ('yes' if c.gpu else 'no'))
    if need_compressors:
        log(c, 'compressors: %s' % ', '.join(c.compressors))
    return c, by

def split(s):
    return [x for x in s.split(',') if x]

