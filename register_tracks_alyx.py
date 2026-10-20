"""
register_tracks_alyx.py
=======================
Review Lasagna-traced electrode tracks (``*_pts.csv``) and register the ones you tick to Alyx
as "Histology track" trajectories, using ibllib's own ``histology.register_track``.

Unlike ibllib's ``register_track_files``, this never creates probe insertions: each file must
match exactly one existing session and insertion, and every track is checked against all of the
subject's micro-manipulator plans before it can be uploaded.

Needs ONE-api >= 3.4 (run in iblenv3.11)::

    python register_tracks_alyx.py [--folder D:/.../electrodetracks]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_ALYX_URL = "https://alyx.internationalbrainlab.org"
LOG_NAME = "alyx_track_upload_log.csv"

# Lasagna saves voxel indices of the CCF 25 um stack as (AP, ML, DV); the CCF grid is 528 x 456 x 320
CCF25_MAX_INDEX = (527, 455, 319)
CCF_MIDLINE_ML_UM = 5739.0
MIN_POINTS = 2
FEW_POINTS = 5
FAR_FROM_PLAN_UM = 700          # observed tracing-to-plan offsets are 100-500 um
HEMISPHERE_MARGIN_UM = 300
CROSS_SESSION_MARGIN_UM = 150   # nearby sites across days are common; only flag a clearly closer plan
SAME_TRACK_UM = 50

NAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<subject>\S+)_(?P<number>\d{3})_"
    r"(?P<probe>probe\d{2}[a-z]?)_pts(?P<ext>(\.csv)+)$")

OK, WARN, BLOCK = "OK", "Warning", "Blocked"
SAME = "Already in Alyx"     # identical to the stored track: nothing to upload
UPLOADABLE = (OK, WARN)


@dataclass
class TrackReview:
    path: Path
    subject: str = ""
    date: str = ""
    number: int = -1
    probe: str = ""
    sha256: str = ""
    eid: str | None = None
    pid: str | None = None
    session_label: str = ""
    alyx_session: str = ""                   # session as Alyx describes it (shown in the confirm dialog)
    n_points: int = 0
    picks: np.ndarray | None = None          # IBL-Allen xyz in meters, as ibllib registers them
    traced_ccf: np.ndarray | None = None     # the same points in CCF um [ML, AP, DV]
    hemisphere: str = ""
    deepest_region: str = ""
    own_plan_um: float | None = None         # median distance to the closest planned shank of this probe
    has_histology: bool = False
    has_alignment: bool = False
    identical_in_alyx: bool = False
    blocks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.blocks:
            return BLOCK
        if self.identical_in_alyx:
            return SAME
        return WARN if self.warnings else OK

    @property
    def details(self) -> list[str]:
        same = ["identical to the track already stored in Alyx for this probe; nothing to upload"]
        return self.blocks + (same if self.identical_in_alyx and not self.blocks else []) + self.warnings

    @property
    def probe_root(self) -> str:
        return _root(self.probe)

    @property
    def shank(self) -> str:
        return self.probe[-1] if self.probe and self.probe[-1].isalpha() else ""


def _root(name: str) -> str:
    return name[:-1] if name and name[-1].isalpha() else name


def _session_id(record: dict):
    s = record.get('session')
    return s.get('id') if isinstance(s, dict) else s


def parse_track_name(path: Path) -> dict:
    """Strict parse of ``{yyyy-mm-dd}_{subject}_{nnn}_{probeNN[x]}_pts.csv``; raises ValueError."""
    if '_pts_transformed' in path.name:
        raise ValueError("_pts_transformed files are not supported; register the _pts.csv traced on the "
                         "CCF-registered (STD_ds) stack")
    m = NAME_RE.match(path.name)
    if not m:
        raise ValueError("file name is not in the form yyyy-mm-dd_Subject_001_probe00a_pts.csv")
    return {'date': m['date'], 'subject': m['subject'], 'number': int(m['number']),
            'probe': m['probe'], 'double_ext': m['ext'] != '.csv'}


def read_raw_indices(path: Path) -> np.ndarray:
    if path.stat().st_size == 0:
        return np.zeros((0, 3))
    raw = np.loadtxt(path, delimiter=',', ndmin=2)
    if raw.shape[1] != 3:
        raise ValueError(f"expected 3 columns (AP, ML, DV indices), found {raw.shape[1]}")
    return raw


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((points - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _track_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric median nearest-point distance between two traced point sets (um)."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(np.median(d.min(axis=1)), np.median(d.min(axis=0))))


def subject_plans(alyx, subject: str, needles) -> list[dict]:
    """Every micro-manipulator trajectory of the subject, in CCF um."""
    from iblatlas.atlas import Insertion
    plans = []
    for t in alyx.rest('trajectories', 'list', subject=subject, provenance='Micro-manipulator', no_cache=True):
        if t.get('provenance') != 'Micro-manipulator' or t.get('x') is None or not t.get('probe_name'):
            continue
        s = t.get('session')
        if isinstance(s, dict) and s.get('subject') not in (None, subject):
            continue
        ins = Insertion.from_dict(t, brain_atlas=needles)
        entry, tip = needles.xyz2ccf(ins.xyz[0], mode='clip'), needles.xyz2ccf(ins.tip, mode='clip')
        label = t['probe_name']
        if isinstance(s, dict) and s.get('start_time'):
            label = f"{s['start_time'][:10]} #{int(s.get('number') or 0):03d} {t['probe_name']}"
        plans.append({'name': t['probe_name'], 'eid': _session_id(t), 'entry_ccf': entry, 'tip_ccf': tip,
                      'label': label})
    return plans


def session_insertions(alyx, eid: str) -> dict:
    return {i['name']: i for i in alyx.rest('insertions', 'list', session=eid, no_cache=True)
            if i.get('session') == eid}


def review_track(path: Path, alyx, allen, needles, cache: dict | None = None) -> TrackReview:
    """Run every single-file check. Reads from Alyx only; never writes."""
    from ibllib.pipes.histology import load_track_csv, get_brain_regions
    r = TrackReview(path=Path(path))
    cache = {} if cache is None else cache

    try:
        p = parse_track_name(r.path)
    except ValueError as e:
        r.blocks.append(str(e))
        return r
    r.subject, r.date, r.number, r.probe = p['subject'], p['date'], p['number'], p['probe']
    r.session_label = f"{r.subject} {r.date} #{r.number:03d}"
    if p['double_ext']:
        r.warnings.append("file name ends in .csv.csv")

    try:
        r.sha256 = file_sha256(r.path)
        raw = read_raw_indices(r.path)
    except Exception as e:
        r.blocks.append(f"could not read file: {e}")
        return r
    r.n_points = len(raw)
    if r.n_points < MIN_POINTS:
        r.blocks.append(f"only {r.n_points} traced point(s); at least {MIN_POINTS} needed")
        return r
    if np.any(raw < 0) or np.any(raw > np.array(CCF25_MAX_INDEX)):
        r.blocks.append("point indices fall outside the 25 um CCF grid; was this traced on a CCF-registered stack?")
        return r
    if r.n_points < FEW_POINTS:
        r.warnings.append(f"only {r.n_points} traced points")

    r.picks = load_track_csv(r.path, brain_atlas=allen)
    try:
        get_brain_regions(r.picks, brain_atlas=allen)
    except Exception as e:
        r.blocks.append(f"ibllib cannot build a track from these points: {e}")
        return r
    deepest = r.picks[np.argmin(r.picks[:, 2])]
    r.deepest_region = str(allen.regions.get(allen.get_labels(deepest[np.newaxis, :])).acronym[0])
    r.traced_ccf = allen.xyz2ccf(r.picks, mode='clip')
    traced_ml = float(np.median(r.traced_ccf[:, 0]))
    r.hemisphere = "right" if traced_ml > CCF_MIDLINE_ML_UM else "left"

    # session and insertion: query, then verify every returned field ourselves
    sessions = [s for s in alyx.rest('sessions', 'list', subject=r.subject, date_range=[r.date, r.date],
                                     number=r.number, no_cache=True)
                if s.get('subject') == r.subject and str(s.get('start_time', ''))[:10] == r.date
                and s.get('number') == r.number]
    if len(sessions) != 1:
        r.blocks.append(f"{len(sessions)} Alyx sessions match {r.session_label}; exactly 1 needed")
        return r
    r.eid = sessions[0]['id']
    r.alyx_session = f"{sessions[0]['subject']} {str(sessions[0]['start_time'])[:16]} #{sessions[0]['number']:03d}"
    key = ('ins', r.eid)
    if key not in cache:
        cache[key] = session_insertions(alyx, r.eid)
    ins = cache[key].get(r.probe)
    if ins is None:
        r.blocks.append(f"no probe insertion named {r.probe} in this session (this tool never creates insertions)")
        return r
    r.pid = ins['id']

    # geometry: compare with every plan of this subject
    key = ('plans', r.subject)
    if key not in cache:
        cache[key] = subject_plans(alyx, r.subject, needles)
    scored = []
    for pl in cache[key]:
        d = float(np.median(_segment_distance(r.traced_ccf, pl['entry_ccf'], pl['tip_ccf'])))
        scored.append((d, pl))
    own = [(d, pl) for d, pl in scored if pl['eid'] == r.eid and _root(pl['name']) == r.probe_root]
    same_session = [(d, pl) for d, pl in scored if pl['eid'] == r.eid and _root(pl['name']) != r.probe_root]
    other_sessions = [(d, pl) for d, pl in scored if pl['eid'] != r.eid]
    if own:
        r.own_plan_um, own_pl = min(own, key=lambda t: t[0])
        if same_session:
            d, pl = min(same_session, key=lambda t: t[0])
            if d < r.own_plan_um:
                r.blocks.append(f"track is closer to the plan for {pl['name']} ({d:.0f} um) than to "
                                f"{r.probe_root} ({r.own_plan_um:.0f} um); check the probe in the file name")
        if other_sessions:
            d, pl = min(other_sessions, key=lambda t: t[0])
            if d < r.own_plan_um - CROSS_SESSION_MARGIN_UM:
                r.warnings.append(f"track is closer to another session's plan, {pl['label']} ({d:.0f} um), than to "
                                  f"this session's {r.probe_root} ({r.own_plan_um:.0f} um); check the date and "
                                  f"session number in the file name")
        # hemisphere of the plan at the traced depth, not at the brain surface
        e, t = own_pl['entry_ccf'], own_pl['tip_ccf']
        dv = float(np.median(r.traced_ccf[:, 2]))
        frac = 0.0 if t[2] == e[2] else float(np.clip((dv - e[2]) / (t[2] - e[2]), 0, 1))
        plan_ml = e[0] + frac * (t[0] - e[0])
        if (abs(plan_ml - CCF_MIDLINE_ML_UM) > HEMISPHERE_MARGIN_UM and abs(traced_ml - CCF_MIDLINE_ML_UM) >
                HEMISPHERE_MARGIN_UM and np.sign(plan_ml - CCF_MIDLINE_ML_UM) != np.sign(traced_ml - CCF_MIDLINE_ML_UM)):
            r.blocks.append(f"traced track is in the {r.hemisphere} hemisphere but {r.probe_root} was planned in the "
                            f"{'right' if plan_ml > CCF_MIDLINE_ML_UM else 'left'} hemisphere")
        if r.own_plan_um > FAR_FROM_PLAN_UM:
            r.warnings.append(f"track is {r.own_plan_um:.0f} um from the planned {r.probe_root} trajectory")
    else:
        r.warnings.append(f"no micro-manipulator plan for {r.probe_root} in this session to check this track against")

    # what already exists on this probe
    hist = [t for t in alyx.rest('trajectories', 'list', probe_insertion=r.pid, provenance='Histology track',
                                 no_cache=True)
            if t.get('probe_insertion') == r.pid and t.get('provenance') == 'Histology track']
    r.has_histology = len(hist) > 0
    stored = (ins.get('json') or {}).get('xyz_picks')
    r.identical_in_alyx = bool(r.has_histology and stored and sorted(map(tuple, stored)) ==
                               sorted(map(tuple, np.int32(r.picks * 1e6).tolist())))
    if len(hist) > 1:
        r.blocks.append(f"{len(hist)} histology tracks already exist for this probe; ibllib would only replace one. "
                        f"Clean this up in Alyx first")
    elif hist and not r.identical_in_alyx:
        r.warnings.append("a histology track already exists for this probe and will be replaced")
    aligned = [t for t in alyx.rest('trajectories', 'list', probe_insertion=r.pid,
                                    provenance='Ephys aligned histology track', no_cache=True)
               if t.get('probe_insertion') == r.pid and t.get('provenance') == 'Ephys aligned histology track']
    qc = (ins.get('json') or {}).get('extended_qc', {}) or {}
    r.has_alignment = len(aligned) > 0 or (qc.get('alignment_count') or 0) > 0
    if r.has_alignment and not r.identical_in_alyx:
        resolved = " (resolved)" if qc.get('alignment_resolved') else ""
        r.warnings.append(f"this probe has an ephys alignment{resolved}; replacing its picks may invalidate it")
    return r


def _stored_ccf(ins: dict, allen) -> np.ndarray | None:
    picks = (ins.get('json') or {}).get('xyz_picks')
    if not picks:
        return None
    return allen.xyz2ccf(np.array(picks, dtype=float) / 1e6, mode='clip')


def cross_check(reviews: list[TrackReview], cache: dict, allen) -> None:
    """Checks across files: duplicate probes, identical tracks, and shank order within each probe."""
    by_pid: dict[str, list[TrackReview]] = {}
    for r in reviews:
        if r.pid:
            by_pid.setdefault(r.pid, []).append(r)
    for group in by_pid.values():
        if len(group) > 1:
            names = ", ".join(g.path.name for g in group)
            for g in group:
                g.blocks.append(f"several files map to the same probe insertion: {names}")

    traced = [r for r in reviews if r.traced_ccf is not None]
    for i, a in enumerate(traced):
        for b in traced[i + 1:]:
            if _track_distance(a.traced_ccf, b.traced_ccf) < SAME_TRACK_UM:
                a.blocks.append(f"same points as {b.path.name}")
                b.blocks.append(f"same points as {a.path.name}")

    groups: dict[tuple, list[TrackReview]] = {}
    for r in reviews:
        if r.eid and r.shank and r.traced_ccf is not None and not r.blocks:
            groups.setdefault((r.eid, r.probe_root, r.subject), []).append(r)
    for (eid, root, subject), group in groups.items():
        plans = {pl['name']: pl for pl in cache.get(('plans', subject), [])
                 if pl['eid'] == eid and _root(pl['name']) == root and pl['name'][-1].isalpha()}
        members = {g.probe: g.traced_ccf for g in group}
        for name, ins in cache.get(('ins', eid), {}).items():
            if _root(name) == root and name not in members and name[-1:].isalpha():
                stored = _stored_ccf(ins, allen)
                if stored is not None:
                    members[name] = stored
                    for g in group:
                        if _track_distance(g.traced_ccf, stored) < SAME_TRACK_UM:
                            g.blocks.append(f"same points as the track already stored in Alyx for {name}")
        members = {n: pts for n, pts in members.items() if n in plans}
        if len(members) < 2 or len(plans) < 2:
            for g in group:
                if not g.blocks:
                    g.warnings.append(f"shank identity not verified: no other shank of {root} is traced here or "
                                      f"stored in Alyx to compare its position with")
            continue
        letters = sorted(plans)
        direction = np.mean([(plans[n]['tip_ccf'] - plans[n]['entry_ccf']) /
                             np.linalg.norm(plans[n]['tip_ccf'] - plans[n]['entry_ccf']) for n in letters], axis=0)
        direction /= np.linalg.norm(direction)
        spread = plans[letters[-1]]['entry_ccf'] - plans[letters[0]]['entry_ccf']
        axis = spread - (spread @ direction) * direction     # shank spacing, independent of depth
        if np.linalg.norm(axis) == 0:
            continue
        axis /= np.linalg.norm(axis)
        planned_order = sorted(members, key=lambda n: plans[n]['entry_ccf'] @ axis)
        traced_order = sorted(members, key=lambda n: float(np.median(members[n] @ axis)))
        if planned_order != traced_order:
            msg = (f"shank order across the probe is {'-'.join(n[-1] for n in traced_order)} in the tracing but "
                   f"{'-'.join(n[-1] for n in planned_order)} in the plan: possible shank label swap (A/D)")
            for g in group:
                g.warnings.append(msg)


def rest_problem(alyx, raise_error=False) -> str:
    """Empty string if this ONE version can make Alyx REST calls, otherwise an explanation.

    ONE-api < 3.4 reads the REST description from /docs, which the current Alyx server refuses (406).
    """
    try:
        alyx.rest_schemes
        return ""
    except Exception as e:
        import importlib.metadata
        msg = (f"ONE-api {importlib.metadata.version('ONE-api')} cannot make REST calls to {alyx.base_url} "
               f"({e}). Upgrade ONE-api to 3.4 or newer in this environment.")
        if raise_error:
            raise RuntimeError(msg) from e
        return msg


def upload_track(r: TrackReview, one, allen) -> str:
    """Register one reviewed track with ibllib, then read it back to confirm. Returns a short result."""
    from ibllib.pipes.histology import register_track
    if r.status not in UPLOADABLE or r.pid is None or r.picks is None:
        raise RuntimeError(f"refusing to upload a track whose status is {r.status!r}")
    if file_sha256(r.path) != r.sha256:
        raise RuntimeError("the file changed after it was checked; click Re-check")
    rest_problem(one.alyx, raise_error=True)    # fail before ibllib's first write, not halfway through
    register_track(r.pid, picks=r.picks, one=one, overwrite=True, brain_atlas=allen)
    hist = [t for t in one.alyx.rest('trajectories', 'list', probe_insertion=r.pid, provenance='Histology track',
                                     no_cache=True)
            if t.get('probe_insertion') == r.pid and t.get('provenance') == 'Histology track']
    if len(hist) != 1:
        raise RuntimeError(f"after upload Alyx has {len(hist)} histology tracks for this probe")
    stored = (one.alyx.rest('insertions', 'read', id=r.pid, no_cache=True).get('json') or {}).get('xyz_picks')
    if stored != np.int32(r.picks * 1e6).tolist():
        raise RuntimeError("picks read back from Alyx do not match the file")
    channels = one.alyx.rest('channels', 'list', trajectory_estimate=hist[0]['id'], no_cache=True)
    if len(channels) == 0:
        raise RuntimeError("the new histology track has no channel locations")
    return f"uploaded and verified ({len(channels)} channels)"


def append_log(folder: Path, server: str, user: str, r: TrackReview, result: str) -> None:
    log = Path(folder) / LOG_NAME
    new = not log.exists()
    with open(log, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['timestamp', 'server', 'alyx_user', 'file', 'sha256', 'subject', 'date', 'session',
                        'probe', 'insertion_id', 'n_points', 'replaced_existing', 'had_alignment', 'result'])
        w.writerow([datetime.datetime.now().isoformat(timespec='seconds'), server, user, r.path.name,
                    r.sha256, r.subject, r.date, r.number, r.probe, r.pid, r.n_points,
                    r.has_histology, r.has_alignment, result])


def review_all(paths, alyx, allen, needles) -> list[TrackReview]:
    cache, reviews = {}, []
    for p in paths:
        try:
            reviews.append(review_track(p, alyx, allen, needles, cache))
        except Exception as e:
            r = TrackReview(path=Path(p))
            r.blocks.append(f"check failed: {e}")
            reviews.append(r)
    cross_check(reviews, cache, allen)
    return reviews


# ===========================================================================
# GUI
# ===========================================================================
def _gui():
    from qtpy import QtWidgets, QtCore, QtGui
    from one.webclient import AlyxClient

    class LoginDialog(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Connect to Alyx")
            self.setMinimumWidth(400)
            self.alyx = None
            try:
                alyx = AlyxClient(base_url=DEFAULT_ALYX_URL, silent=True)
                alyx.authenticate()
                if alyx.is_logged_in:
                    self.alyx = alyx
                    QtCore.QTimer.singleShot(0, self.accept)
                    return
            except Exception:
                pass
            self.url_edit = QtWidgets.QLineEdit(DEFAULT_ALYX_URL)
            self.user_edit = QtWidgets.QLineEdit()
            self.pw_edit = QtWidgets.QLineEdit()
            self.pw_edit.setEchoMode(QtWidgets.QLineEdit.Password)
            self.status_lbl = QtWidgets.QLabel("")
            self.status_lbl.setWordWrap(True)
            ok = QtWidgets.QPushButton("Connect")
            ok.setDefault(True)
            ok.clicked.connect(self._connect)
            cancel = QtWidgets.QPushButton("Cancel")
            cancel.clicked.connect(self.reject)
            form = QtWidgets.QFormLayout()
            form.addRow("Alyx URL:", self.url_edit)
            form.addRow("Username:", self.user_edit)
            form.addRow("Password:", self.pw_edit)
            row = QtWidgets.QHBoxLayout()
            row.addStretch(); row.addWidget(cancel); row.addWidget(ok)
            lay = QtWidgets.QVBoxLayout(self)
            lay.addLayout(form); lay.addWidget(self.status_lbl); lay.addLayout(row)

        def _connect(self):
            url, user, pw = self.url_edit.text().strip(), self.user_edit.text().strip(), self.pw_edit.text()
            if not user or not pw:
                self.status_lbl.setText("Enter username and password.")
                return
            try:
                alyx = AlyxClient(base_url=url, silent=True)
                alyx.authenticate(username=user, password=pw, cache_token=True, force=True)
                if not alyx.is_logged_in:
                    self.status_lbl.setText("Authentication failed.")
                    return
                self.alyx = alyx
                self.accept()
            except Exception as exc:
                self.status_lbl.setText(f"Login failed: {exc}")

    STATUS_COLORS = {OK: '#d8f0d8', WARN: '#fbe8c0', BLOCK: '#f6d0d0', SAME: '#e3e6ea'}
    COLUMNS = ["Upload", "File", "Session", "Probe", "Points", "Hemisphere", "Deepest region",
               "Distance to plan", "Already in Alyx", "Status", "Details"]
    PATH_ROLE = QtCore.Qt.UserRole

    class ConfirmDialog(QtWidgets.QDialog):
        def __init__(self, server, user, rows, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Review before upload")
            self.setMinimumWidth(820)
            lay = QtWidgets.QVBoxLayout(self)
            lay.addWidget(QtWidgets.QLabel(
                f"<b>{len(rows)} track(s) will be registered to</b> {server} <b>as</b> {user}"))
            text = QtWidgets.QPlainTextEdit()
            text.setReadOnly(True)
            lines = []
            for r in rows:
                lines.append(f"{r.path.name}\n    -> Alyx session {r.alyx_session}, probe {r.probe} "
                             f"(insertion {r.pid})\n       {r.n_points} points, deepest point in {r.deepest_region}")
                lines += [f"    WARNING: {w}" for w in r.warnings]
            text.setPlainText("\n".join(lines))
            lay.addWidget(text)
            n_warn = sum(1 for r in rows if r.warnings)
            self.ack = None
            if n_warn:
                self.ack = QtWidgets.QCheckBox(
                    f"I have read the warnings for {n_warn} track(s) and want to upload them anyway")
                lay.addWidget(self.ack)
            box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
            self.go = box.addButton("Upload to Alyx", QtWidgets.QDialogButtonBox.AcceptRole)
            box.accepted.connect(self.accept)
            box.rejected.connect(self.reject)
            lay.addWidget(box)
            if self.ack is not None:
                self.go.setEnabled(False)
                self.ack.toggled.connect(self.go.setEnabled)

    class Window(QtWidgets.QMainWindow):
        def __init__(self, alyx, folder):
            super().__init__()
            from iblatlas.atlas import AllenAtlas, NeedlesAtlas
            self.alyx = alyx
            self.server = alyx.base_url
            self.setWindowTitle(f"Register Lasagna tracks to Alyx  ({self.server}, {getattr(alyx, 'user', '')})")
            self.resize(1500, 650)
            self.statusBar().showMessage("Loading atlases...")
            QtWidgets.QApplication.processEvents()
            self.allen, self.needles = AllenAtlas(25), NeedlesAtlas()
            self.one = None
            self.reviews: dict[str, TrackReview] = {}
            self.folder = Path(folder) if folder else None
            self.rest_ok = True

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            lay = QtWidgets.QVBoxLayout(central)
            top = QtWidgets.QHBoxLayout()
            self.folder_lbl = QtWidgets.QLabel("No folder chosen")
            b_folder = QtWidgets.QPushButton("Choose folder...")
            b_folder.clicked.connect(self._choose_folder)
            b_files = QtWidgets.QPushButton("Choose files...")
            b_files.clicked.connect(self._choose_files)
            b_recheck = QtWidgets.QPushButton("Re-check")
            b_recheck.clicked.connect(lambda: self._review([Path(p) for p in self.reviews]))
            self.b_download = QtWidgets.QPushButton("Download from Alyx...")
            self.b_download.setToolTip("Save the tracks already stored in Alyx for a subject as _pts.csv files "
                                       "in a separate from_alyx folder (reads Alyx only)")
            self.b_download.clicked.connect(self._download)
            top.addWidget(b_folder); top.addWidget(b_files); top.addWidget(b_recheck)
            top.addWidget(self.b_download)
            top.addSpacing(12); top.addWidget(self.folder_lbl, 1)
            lay.addLayout(top)

            self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
            self.table.setHorizontalHeaderLabels(COLUMNS)
            self.table.verticalHeader().setVisible(False)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            self.table.horizontalHeader().setStretchLastSection(True)
            lay.addWidget(self.table, 1)

            bottom = QtWidgets.QHBoxLayout()
            b_all = QtWidgets.QPushButton("Check all boxes for upload")
            b_all.clicked.connect(lambda: self._set_all(True))
            b_none = QtWidgets.QPushButton("Uncheck all")
            b_none.clicked.connect(lambda: self._set_all(False))
            self.b_upload = QtWidgets.QPushButton("Upload checked tracks...")
            self.b_upload.clicked.connect(self._upload)
            bottom.addWidget(b_all); bottom.addWidget(b_none); bottom.addStretch(); bottom.addWidget(self.b_upload)
            lay.addLayout(bottom)
            lay.addWidget(QtWidgets.QLabel(
                "<small>Blocked tracks cannot be uploaded; fix the problem in Details and click Re-check. "
                "Nothing is written to Alyx until you confirm in the review dialog.</small>"))
            self.controls = [b_folder, b_files, b_recheck, self.b_download, b_all, b_none, self.b_upload,
                             self.table]

            problem = rest_problem(self.alyx)
            if problem:
                self.rest_ok = False
                self.b_upload.setEnabled(False)
                self.b_download.setEnabled(False)
                QtWidgets.QMessageBox.critical(self, "Alyx is not reachable from this ONE version", problem)
                self.statusBar().showMessage(problem)
            elif self.folder:
                self._review(sorted(self.folder.glob('*_pts*.csv')))
            else:
                self.statusBar().showMessage("Choose a folder or files to review.")

        def _busy(self, busy):
            for c in self.controls:
                c.setEnabled(not busy)
            if not self.rest_ok:
                self.b_upload.setEnabled(False)
                self.b_download.setEnabled(False)
            QtWidgets.QApplication.processEvents()

        def _download(self):
            from export_tracks_alyx import export_subject
            guess = ""
            if self.folder is not None:
                parts = self.folder.parts
                if self.folder.name == 'electrodetracks':
                    guess = parts[-2]
                elif self.folder.name == 'from_alyx' and len(parts) >= 3:
                    guess = parts[-3]
            subject, ok = QtWidgets.QInputDialog.getText(
                self, "Download tracks from Alyx", "Subject whose stored tracks to download:", text=guess)
            subject = subject.strip()
            if not ok or not subject:
                return
            f = self.folder
            if f is not None and f.name == 'electrodetracks' and f.parent.name == subject:
                out = f / 'from_alyx'
            elif f is not None and f.name == 'from_alyx' and f.parent.name == 'electrodetracks' \
                    and f.parent.parent.name == subject:
                out = f
            else:
                chosen = QtWidgets.QFileDialog.getExistingDirectory(
                    self, f"Where should {subject}'s tracks go? (a from_alyx folder is created inside)",
                    str(f or ''))
                if not chosen:
                    return
                chosen = Path(chosen)
                out = chosen if chosen.name == 'from_alyx' else chosen / 'from_alyx'
            self._busy(True)
            try:
                self.statusBar().showMessage(f"Downloading {subject}'s tracks from Alyx...")
                QtWidgets.QApplication.processEvents()
                res = export_subject(self.alyx, subject, self.allen, out, server=self.server)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Download failed", f"{subject}: {e}")
                return
            finally:
                self._busy(False)
            n = {k: sum(r.status.startswith(k) for r in res) for k in ('written', 'unchanged', 'skipped')}
            problems = [f"{r.date} {r.probe}: {r.status}" for r in res
                        if r.status.startswith('skipped') and 'no stored track' not in r.status]
            msg = (f"{n['written']} new file(s), {n['unchanged']} already there and unchanged, "
                   f"{n['skipped'] - len(problems)} probe(s) with no stored track.\n\nSaved in:\n{out}")
            if problems:
                msg += "\n\nNot written:\n" + "\n".join(problems)
            if not (n['written'] or n['unchanged']):
                QtWidgets.QMessageBox.information(self, "Nothing downloaded", msg)
                return
            msg += "\n\nReview the downloaded files here now?"
            if QtWidgets.QMessageBox.question(self, "Downloaded", msg) == QtWidgets.QMessageBox.Yes:
                self.folder = out
                self._review(sorted(out.glob('*_pts*.csv')))
            else:
                self.statusBar().showMessage(f"Downloaded {subject}'s tracks to {out}")

        def _choose_folder(self):
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder with _pts.csv files",
                                                           str(self.folder or ''))
            if d:
                self.folder = Path(d)
                self._review(sorted(self.folder.glob('*_pts*.csv')))

        def _choose_files(self):
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose _pts.csv files", str(self.folder or ''),
                                                              "Lasagna tracks (*_pts*.csv)")
            if files:
                self.folder = Path(files[0]).parent
                self._review([Path(f) for f in files])

        def _review(self, paths):
            self.table.setRowCount(0)
            self.reviews = {}
            self._busy(True)
            try:
                self.folder_lbl.setText(str(self.folder) if self.folder else "")
                self.statusBar().showMessage(f"Checking {len(paths)} file(s) against Alyx...")
                QtWidgets.QApplication.processEvents()
                self.reviews = {str(r.path): r for r in review_all(paths, self.alyx, self.allen, self.needles)}
                self._fill_table()
                n = {s: sum(r.status == s for r in self.reviews.values()) for s in (OK, WARN, BLOCK, SAME)}
                self.statusBar().showMessage(f"{len(self.reviews)} file(s): {n[OK]} OK, {n[WARN]} with warnings, "
                                             f"{n[BLOCK]} blocked, {n[SAME]} already in Alyx unchanged.")
            finally:
                self._busy(False)

        def _fill_table(self):
            self.table.setRowCount(len(self.reviews))
            for row, r in enumerate(self.reviews.values()):
                box = QtWidgets.QTableWidgetItem()
                box.setCheckState(QtCore.Qt.Unchecked)
                box.setData(PATH_ROLE, str(r.path))
                box.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable if r.status in UPLOADABLE
                             else QtCore.Qt.NoItemFlags)
                self.table.setItem(row, 0, box)
                existing = ", ".join(s for s, f in (("track", r.has_histology), ("alignment", r.has_alignment)) if f)
                dist = "" if r.own_plan_um is None else f"{r.own_plan_um:.0f} um"
                values = [r.path.name, r.session_label, r.probe, str(r.n_points or ""), r.hemisphere,
                          r.deepest_region, dist, existing or "no", r.status,
                          "; ".join(r.details)]
                for col, v in enumerate(values, start=1):
                    it = QtWidgets.QTableWidgetItem(v)
                    it.setToolTip("\n".join(r.details) or "All checks passed")
                    if col == 9:
                        it.setBackground(QtGui.QColor(STATUS_COLORS[r.status]))
                        it.setFont(QtGui.QFont(it.font().family(), weight=QtGui.QFont.Bold))
                    self.table.setItem(row, col, it)
            self.table.resizeColumnsToContents()

        def _set_all(self, checked):
            for row in range(self.table.rowCount()):
                box = self.table.item(row, 0)
                if box.flags() & QtCore.Qt.ItemIsUserCheckable:
                    box.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)

        def _checked_paths(self):
            out = []
            for row in range(self.table.rowCount()):
                box = self.table.item(row, 0)
                path = box.data(PATH_ROLE)
                r = self.reviews.get(path)
                if r is not None and r.status in UPLOADABLE and box.checkState() == QtCore.Qt.Checked:
                    out.append(path)
            return out

        def _upload(self):
            chosen = self._checked_paths()
            if not chosen:
                QtWidgets.QMessageBox.information(self, "Nothing to upload", "Tick at least one track first.")
                return
            self._busy(True)
            try:
                self._do_upload(chosen)
            finally:
                self._busy(False)

        def _do_upload(self, chosen):
            # re-check everything right before writing, in case Alyx or the files changed
            self.statusBar().showMessage("Re-checking before upload...")
            fresh = {str(r.path): r for r in review_all([Path(p) for p in self.reviews], self.alyx,
                                                        self.allen, self.needles)}
            rows = [fresh[p] for p in chosen]
            newly_blocked = [r for r in rows if r.status not in UPLOADABLE]
            self.reviews = fresh
            self._fill_table()
            if newly_blocked:
                QtWidgets.QMessageBox.warning(self, "Checks changed",
                                              "These tracks can no longer be uploaded, so nothing was uploaded:\n" +
                                              "\n".join(f"{r.path.name}: {r.status}" for r in newly_blocked))
                return
            user = getattr(self.alyx, 'user', '')
            if ConfirmDialog(self.server, user, rows, self).exec_() != QtWidgets.QDialog.Accepted:
                self.statusBar().showMessage("Upload cancelled; nothing was written.")
                return
            if self.one is None:
                from one.api import ONE
                one = ONE(base_url=self.server, username=user, mode='remote', silent=True)
                if not one.alyx.is_logged_in or one.alyx.user != user:
                    QtWidgets.QMessageBox.critical(
                        self, "Login mismatch", f"ONE is logged in as {one.alyx.user!r}, not {user!r}; "
                        f"nothing was uploaded.")
                    return
                self.one = one
            done = []
            for i, r in enumerate(rows):
                self.statusBar().showMessage(f"Uploading {r.path.name} ({i + 1}/{len(rows)})...")
                QtWidgets.QApplication.processEvents()
                try:
                    append_log(r.path.parent, self.server, user, r, "started")
                except Exception as e:
                    done.append(f"{r.path.name}: NOT uploaded, the log could not be written ({e})")
                    continue
                try:
                    result = upload_track(r, self.one, self.allen)
                except Exception as e:
                    result = f"FAILED: {e}"
                try:
                    append_log(r.path.parent, self.server, user, r, result)
                except Exception as e:
                    result += f" (result not logged: {e})"
                done.append(f"{r.path.name}: {result}")
            QtWidgets.QMessageBox.information(self, "Upload finished",
                                              "\n".join(done) + f"\n\nLogged in {LOG_NAME} next to the files.")
            self._review([Path(p) for p in self.reviews])

    return QtWidgets, LoginDialog, Window


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--folder', help="folder containing *_pts.csv files")
    args = ap.parse_args(argv)
    QtWidgets, LoginDialog, Window = _gui()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    dlg = LoginDialog()
    if dlg.exec_() != QtWidgets.QDialog.Accepted or dlg.alyx is None:
        sys.exit(0)
    win = Window(dlg.alyx, args.folder)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
