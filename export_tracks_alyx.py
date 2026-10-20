"""
export_tracks_alyx.py
=====================
Download the histology tracks stored in Alyx (the ``xyz_picks`` of each probe insertion) and write
them back out as Lasagna ``_pts.csv`` files, named with their current Alyx labels, in a separate
``from_alyx`` folder. Read-only on Alyx.

The conversion is the exact inverse of ibllib's ``histology.load_track_csv``, so registering an
exported file unchanged reproduces the stored points; every file is checked for this after writing.

Needs ONE-api >= 3.4 (run in iblenv3.11)::

    python export_tracks_alyx.py KM_048 [KM_033 ...] [--date 2025-06-03] [--probe probe00]
                                 [--out D:/.../KM_048/electrodetracks/from_alyx]

Without --out, each subject's files go to ``<current folder>/<subject>_from_alyx``.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ALYX_URL = "https://alyx.internationalbrainlab.org"
CCF25_AP_MAX = 527
MANIFEST = "export_manifest.csv"


@dataclass
class Exported:
    subject: str
    date: str
    number: int
    probe: str
    pid: str
    path: Path | None
    n_points: int
    status: str        # written / unchanged / skipped: <reason>


def track_file_name(date: str, subject: str, number: int, probe: str) -> str:
    return f"{date}_{subject}_{number:03d}_{probe}_pts.csv"


def picks_to_lasagna_rows(xyz_picks_um, allen) -> np.ndarray:
    """Stored Alyx picks (IBL-Allen, integer um) -> Lasagna voxel rows (AP posterior-first, ML, DV)."""
    xyz = np.asarray(xyz_picks_um, dtype=float) / 1e6
    ixiyiz = allen.bc.xyz2i(xyz, round=True, mode='raise')      # [ML, AP anterior-first, DV]
    return np.c_[CCF25_AP_MAX - ixiyiz[:, 1], ixiyiz[:, 0], ixiyiz[:, 2]].astype(float)


def rows_to_text(rows: np.ndarray) -> str:
    return "".join(",".join(f"{v:.1f}" for v in r) + "\n" for r in rows)


def default_out_dir(subject: str, base: Path | None = None) -> Path:
    return Path(base or Path.cwd()) / f"{subject}_from_alyx"


def export_subject(alyx, subject: str, allen, out_dir: Path | None = None, dates=None, probes=None,
                   server: str = DEFAULT_ALYX_URL) -> list[Exported]:
    """Write one _pts.csv per insertion of `subject` that has stored picks. Never overwrites a different file."""
    from ibllib.pipes.histology import load_track_csv
    out_dir = Path(out_dir) if out_dir else default_out_dir(subject)
    sessions = {s['id']: s for s in alyx.rest('sessions', 'list', subject=subject, no_cache=True)
                if s.get('subject') == subject}
    results = []
    for ins in sorted(alyx.rest('insertions', 'list', subject=subject, no_cache=True),
                      key=lambda i: (str(sessions.get(i.get('session'), {}).get('start_time', '')), i.get('name', ''))):
        s = sessions.get(ins.get('session'))
        if s is None:
            continue
        date, number, probe = str(s['start_time'])[:10], int(s.get('number') or 0), ins.get('name') or ''
        if (dates and date not in dates) or (probes and not any(probe.startswith(p) for p in probes)):
            continue
        rec = Exported(subject, date, number, probe, ins['id'], None, 0, '')
        picks = (ins.get('json') or {}).get('xyz_picks')
        if not picks:
            rec.status = 'skipped: no stored track'
            results.append(rec)
            continue
        try:
            rows = picks_to_lasagna_rows(picks, allen)
        except Exception as e:
            rec.status = f'skipped: stored points could not be converted to the 25 um atlas grid ({e})'
            results.append(rec)
            continue
        rec.n_points = len(rows)
        text = rows_to_text(rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        rec.path = out_dir / track_file_name(date, subject, number, probe)
        if rec.path.exists():
            if rec.path.read_text() == text:
                rec.status = 'unchanged'
            else:
                rec.status = 'skipped: a different file with this name is already there (rename or move it first)'
            results.append(rec)
            continue
        rec.path.write_text(text)
        back = np.int32(load_track_csv(rec.path, brain_atlas=allen) * 1e6).tolist()
        if sorted(map(tuple, back)) != sorted(map(tuple, picks)):
            rec.path.unlink()
            rec.status = 'skipped: round trip did not reproduce the stored points (file removed)'
        else:
            rec.status = 'written'
        results.append(rec)
    _write_manifest(out_dir, results, server)
    return results


def _write_manifest(out_dir: Path, results: list[Exported], server: str) -> None:
    written = [r for r in results if r.path is not None and r.status in ('written', 'unchanged')]
    if not written:
        return
    log = out_dir / MANIFEST
    new = not log.exists()
    with open(log, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['exported_at', 'server', 'file', 'subject', 'date', 'session', 'probe', 'insertion_id',
                        'n_points', 'sha256', 'status'])
        now = datetime.datetime.now().isoformat(timespec='seconds')
        for r in written:
            w.writerow([now, server, r.path.name, r.subject, r.date, r.number, r.probe, r.pid, r.n_points,
                        hashlib.sha256(r.path.read_bytes()).hexdigest(), r.status])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('subjects', nargs='+')
    ap.add_argument('--date', action='append', help='only this session date (repeatable)')
    ap.add_argument('--probe', action='append', help='only probes starting with this, e.g. probe00 (repeatable)')
    ap.add_argument('--out', help='output folder (default: <current folder>/<subject>_from_alyx)')
    args = ap.parse_args(argv)
    from one.webclient import AlyxClient
    from iblatlas.atlas import AllenAtlas
    alyx = AlyxClient(base_url=DEFAULT_ALYX_URL, silent=True)
    alyx.authenticate()
    allen = AllenAtlas(25)
    for subject in args.subjects:
        res = export_subject(alyx, subject, allen, args.out, args.date, args.probe, server=alyx.base_url)
        n = {k: sum(r.status.startswith(k) for r in res) for k in ('written', 'unchanged', 'skipped')}
        out = next((r.path.parent for r in res if r.path), args.out or default_out_dir(subject))
        print(f"{subject}: {n['written']} written, {n['unchanged']} unchanged, {n['skipped']} skipped -> {out}")
        for r in res:
            if r.status.startswith('skipped') and 'no stored track' not in r.status:
                print(f"   {r.date} {r.probe}: {r.status}")


if __name__ == '__main__':
    main()
