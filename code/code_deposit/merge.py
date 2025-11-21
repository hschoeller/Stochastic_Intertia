#!/usr/bin/env python3
import os
import glob
import pickle
import tempfile

IN_DIR = "./cdv_model/data/eps_results"
OUT_DIR = "./cdv_model/data"
os.makedirs(OUT_DIR, exist_ok=True)


def load_pair(p):
    try:
        with open(p, 'rb') as f:
            d = pickle.load(f)
        if isinstance(d, dict) and 'eps' in d and 'value' in d:
            return float(d['eps']), d['value']
        if isinstance(d, tuple) and len(d) == 2:
            return float(d[0]), d[1]
        if isinstance(d, dict) and len(d) == 1:
            k = next(iter(d))
            return float(k), d[k]
    except Exception:
        pass
    raise ValueError("bad format: "+p)


def merge(pattern):
    out = {}
    for p in sorted(glob.glob(os.path.join(IN_DIR, pattern))):
        if p.endswith('.tmp'):
            continue
        try:
            eps, val = load_pair(p)
            out[eps] = val
        except Exception as e:
            print("skip", p, ":", e)
    return dict(sorted(out.items()))


def atomic_write(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


t_escape = merge("TOesc_*.pkl")
regime = merge("eps_*.pkl")

atomic_write(t_escape, os.path.join(OUT_DIR, "dataOro20_detTOesc.pkl"))
atomic_write(regime,   os.path.join(OUT_DIR, "dataOro20_TORegdet.pkl"))

print("Wrote", len(t_escape), "t_escape entries and",
      len(regime), "regime entries")
