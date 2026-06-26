"""Loader for mi_config.yaml (Phase-2 measured-intrinsic configs).

Merges the top-level ``defaults`` block into each per-param_set entry (deep
merge, so nested ``build`` / ``split`` dicts combine key-by-key), and resolves
the scalar-or-list ``n_dof`` / ``n_keep`` convention.
"""
import copy
from pathlib import Path

import yaml


def _deep_merge(base, over):
    out = copy.deepcopy(base or {})
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_config_path():
    return Path('mi_config.yaml')


def load_doc(config_path=None):
    return yaml.safe_load(open(config_path or default_config_path()))


def mi_names(param_set, config_path=None, doc=None):
    doc = doc or load_doc(config_path)
    return [e['name'] for e in doc.get('measured_intrinsics', {}).get(param_set, [])]


def load_mi_config(param_set, mi_name, config_path=None, doc=None):
    """Fully merged config dict for one (param_set, mi_name): defaults + entry."""
    doc = doc or load_doc(config_path)
    defaults = doc.get('defaults', {})
    for e in doc.get('measured_intrinsics', {}).get(param_set, []):
        if e.get('name') == mi_name:
            return _deep_merge(defaults, e)
    raise KeyError(f'MI config {mi_name!r} not found for param_set {param_set!r}')


def analysis_config_path():
    return Path('analysis_config.yaml')


def load_analysis_doc(config_path=None):
    """Load analysis_config.yaml (LUT / aberration-pairs knobs).  Returns {} if
    the file is absent so callers can fall back to code-level defaults."""
    p = Path(config_path or analysis_config_path())
    return (yaml.safe_load(open(p)) or {}) if p.exists() else {}


def analysis_section(section, param_set=None, mi_name=None,
                     config_path=None, doc=None):
    """Merged analysis knobs for one ``section`` ('lut' | 'aberration_pairs').

    Resolution (deep-merge, later wins):
      defaults[section]
        <- overrides[param_set][section]
        <- overrides[param_set][mi_name][section]
    Kept separate from mi_config.yaml so editing these does not re-trigger the
    measured-intrinsic build (which lists mi_config.yaml as an input)."""
    doc = load_analysis_doc(config_path) if doc is None else doc
    out = (doc.get('defaults') or {}).get(section) or {}
    ov = ((doc.get('overrides') or {}).get(param_set) or {}) if param_set else {}
    out = _deep_merge(out, ov.get(section) or {})
    if mi_name and isinstance(ov.get(mi_name), dict):
        out = _deep_merge(out, ov[mi_name].get(section) or {})
    return out


def resolve_indices(spec):
    """Resolve the scalar-or-list convention to a sorted list of int indices.
    ``int n`` -> [0, 1, ..., n-1];  ``list`` -> exactly those indices."""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        return sorted(int(x) for x in spec)
    return list(range(int(spec)))


def rotator_bins(cfg):
    return [(float(lo), float(hi)) for lo, hi in cfg['rotator_bins']]


def build_source(cfg, mi_name):
    """The mi_name whose build grids this entry's split should read.

    ``build_from`` lets a derived entry (e.g. a rotator-subset split) reuse a
    parent entry's already-built per-rotator-bin grids instead of triggering its
    own build.  Absent -> self."""
    return (cfg.get('build_from') or mi_name)


def rotator_select(cfg):
    """Optional subset of rotator bins the split actually uses, as
    [(lo, hi), ...].  Read from ``split.rotator_select``; None -> use all bins."""
    sel = (cfg.get('split') or {}).get('rotator_select')
    if not sel:
        return None
    return [(float(lo), float(hi)) for lo, hi in sel]


def selected_rotator_bins(cfg):
    """The entry's rotator bins, filtered by ``split.rotator_select`` if present
    (else all bins).  Shared by run_intrinsic_split and the grid-reading study/wfs
    scripts so they decompose/plot the same subset.  Raises if a selected bin is
    not one of the entry's rotator_bins (catches config typos)."""
    bins = rotator_bins(cfg)
    sel = rotator_select(cfg)
    if sel is None:
        return bins
    key = lambda lo, hi: (round(lo, 3), round(hi, 3))
    have = {key(lo, hi) for lo, hi in bins}
    missing = {key(lo, hi) for lo, hi in sel} - have
    if missing:
        raise ValueError(f'rotator_select bins not in rotator_bins: {sorted(missing)}')
    sel_set = {key(lo, hi) for lo, hi in sel}
    return [(lo, hi) for lo, hi in bins if key(lo, hi) in sel_set]


def ocs_only_js(cfg, noll_list):
    """Set of Noll j for which the camera (CCS) term is forced to zero.

    Resolution (split.* keys, later wins is not needed — they are alternatives):
      * ``split.split_js``  -> j to KEEP a full OCS/CCS split; all OTHERS in
        ``noll_list`` become OCS-only.  (Convenient "only split Z4..Z8" form.)
      * ``split.ocs_only``  -> explicit j to force OCS-only.
    If both are given, their effects are unioned (a j is OCS-only if it is in
    ocs_only OR not in split_js).  Neither -> empty set (full split for all)."""
    split = cfg.get('split') or {}
    noll = [int(j) for j in noll_list]
    out = set()
    split_js = split.get('split_js')
    if split_js is not None:
        keep = {int(j) for j in split_js}
        out |= {j for j in noll if j not in keep}
    out |= {int(j) for j in (split.get('ocs_only') or [])}
    return out


def as_band_list(filt):
    if filt is None:
        return None
    return [filt] if isinstance(filt, str) else list(filt)
