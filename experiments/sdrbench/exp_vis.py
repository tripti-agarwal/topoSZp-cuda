#!/usr/bin/env python3
"""Experiment "vis": fields for the visualization figure.

    python3 exp_vis.py --data DATA --out RESULTS [--vis-field CESM-ATM-2D/CLDHGH_1_1800_3600] [--vis-bound abs:1e-3]

Writes <out>/vis/<field>_{original,szp,toposzp}.f32 (and .vti if VTK for Python
is installed) for ParaView / TTK.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def add_args(ap):
    ap.add_argument('--vis-field', default='CESM-ATM-2D/CLDHGH_1_1800_3600')
    ap.add_argument('--vis-bound', default='abs:1e-3')

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
    log(c, 'vis: %s at %s -> %s_{original,szp,toposzp}.f32 (topo_check results in %s)' % (c.vis_field, c.vis_bound, base, os.path.join('logs', c.tag, 'vis.log')))

def run_exp(c, by, a):
    c.vis_field, c.vis_bound = a.vis_field, a.vis_bound
    log(c, '==== experiment: vis (%s, %s)' % (a.vis_field, a.vis_bound))
    exp_vis(c, by)

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
