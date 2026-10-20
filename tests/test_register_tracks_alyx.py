"""Offline tests for register_tracks_alyx.py. No network: Alyx and ONE are replaced by in-memory fakes.

Run in iblenv3.11:  python tests/test_register_tracks_alyx.py -v
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import register_tracks_alyx as rta  # noqa: E402
from iblatlas.atlas import AllenAtlas, NeedlesAtlas, Insertion  # noqa: E402

ALLEN, NEEDLES = AllenAtlas(25), NeedlesAtlas()
EID = 'eid-KM_077-2026-05-21-001'
EID2 = 'eid-KM_077-2026-05-29-001'
SESSIONS = {EID: {'id': EID, 'subject': 'KM_077', 'start_time': '2026-05-21T10:00:00', 'number': 1},
            EID2: {'id': EID2, 'subject': 'KM_077', 'start_time': '2026-05-29T10:00:00', 'number': 1}}


def planned(name, x, y, eid=EID, theta=15.0, phi=180.0, depth=4000.0):
    return {'probe_name': name, 'x': x, 'y': y, 'z': 0.0, 'phi': phi, 'theta': theta, 'depth': depth,
            'roll': 0.0, 'provenance': 'Micro-manipulator', 'session': SESSIONS[eid],
            'probe_insertion': f'pid-{eid}-{name}'}


PLANS = [planned(f'probe00{s}', -2288.0, -3106.0 - 250.0 * i) for i, s in enumerate('abcd')]
PLANS += [planned(f'probe01{s}', 2000.0, -1500.0 - 250.0 * i) for i, s in enumerate('abcd')]
PLANS2 = [planned(f'probe00{s}', 1500.0, -5500.0 - 250.0 * i, eid=EID2) for i, s in enumerate('abcd')]


def lasagna_points(traj, mirror=False, n=30, start=0.1, stop=1.0):
    """Lasagna-style voxel rows (AP index posterior-first, ML, DV) lying along a planned trajectory."""
    ins = Insertion.from_dict(traj, brain_atlas=NEEDLES)
    pts = ins.xyz[0] + np.linspace(start, stop, n)[:, None] * (ins.tip - ins.xyz[0])
    ccf = NEEDLES.xyz2ccf(pts, mode='clip') / 25.0            # [ML, AP, DV] voxels, AP anterior-first
    if mirror:
        ccf[:, 0] = 2 * rta.CCF_MIDLINE_ML_UM / 25.0 - ccf[:, 0]
    return np.round(np.c_[527 - ccf[:, 1], ccf[:, 0], ccf[:, 2]])


class FakeAlyx:
    """Read-only Alyx: list/read work from in-memory tables; any other action fails the test.

    ignore_filters=True mimics a server that silently ignores query parameters and returns everything.
    """
    base_url = 'https://fake.alyx'
    user = 'tester'
    rest_schemes = {'insertions': {}, 'sessions': {}, 'trajectories': {}, 'channels': {}}

    def __init__(self, plans=None, ignore_filters=False):
        plans = PLANS if plans is None else plans
        self.sessions = list(SESSIONS.values()) + [
            {'id': 'eid-other', 'subject': 'KM_081', 'start_time': '2026-05-21T09:00:00', 'number': 1}]
        self.insertions = [{'id': p['probe_insertion'], 'name': p['probe_name'], 'session': p['session']['id'],
                            'json': {}} for p in plans]
        self.insertions.append({'id': 'pid-other', 'name': 'probe00a', 'session': 'eid-other', 'json': {}})
        self.trajectories = [dict(p) for p in plans]
        self.channels = []
        self.ignore = ignore_filters
        self.writes = []

    def _match(self, rec, **kw):
        if self.ignore:
            return True
        for k, v in kw.items():
            if k == 'no_cache':
                continue
            if k == 'date_range':
                if str(rec.get('start_time', ''))[:10] != v[0]:
                    return False
            elif k == 'subject' and 'session' in rec and isinstance(rec['session'], dict):
                if rec['session']['subject'] != v:
                    return False
            elif k == 'subject' and 'session' in rec:
                ses = next((s for s in self.sessions if s['id'] == rec['session']), {})
                if ses.get('subject') != v:
                    return False
            elif k == 'session':
                if rta._session_id(rec) != v:
                    return False
            elif rec.get(k) != v:
                return False
        return True

    def rest(self, endpoint, action, **kw):
        if action not in ('list', 'read'):
            self.writes.append((endpoint, action, kw))
            raise AssertionError(f"review tried to write to Alyx: {endpoint} {action}")
        if endpoint == 'insertions' and action == 'read':
            return next(i for i in self.insertions if i['id'] == kw['id'])
        table = {'sessions': self.sessions, 'insertions': self.insertions,
                 'trajectories': self.trajectories, 'channels': self.channels}[endpoint]
        return [dict(r) for r in table if self._match(r, **kw)]


def write(folder, name, rows):
    p = Path(folder) / name
    np.savetxt(p, rows, delimiter=',', fmt='%.1f')
    return p


def plan(name, plans=PLANS):
    return next(p for p in plans if p['probe_name'] == name)


class TrackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def w(self, name, rows):
        return write(self.dir, name, rows)

    def review(self, paths, alyx=None):
        alyx = alyx or FakeAlyx()
        reviews = rta.review_all(paths, alyx, ALLEN, NEEDLES)
        self.assertEqual(alyx.writes, [])
        return reviews

    def four_shanks(self, swap=None, starts=None):
        swap, starts = swap or {}, starts or {}
        return [self.w(f'2026-05-21_KM_077_001_probe00{s}_pts.csv',
                       lasagna_points(plan(f'probe00{swap.get(s, s)}'), start=starts.get(s, 0.1)))
                for s in 'abcd']

    # ---------- matching ----------
    def test_four_correct_shanks_ok(self):
        reviews = self.review(self.four_shanks())
        self.assertTrue(all(r.status == rta.OK for r in reviews), [r.blocks + r.warnings for r in reviews])
        self.assertEqual([r.pid for r in reviews], [f'pid-{EID}-probe00{s}' for s in 'abcd'])
        self.assertEqual(reviews[0].hemisphere, 'left')

    def test_bad_name_blocked(self):
        r, = self.review([self.w('KM_077_probe00a_pts.csv', lasagna_points(plan('probe00a')))])
        self.assertEqual(r.status, rta.BLOCK)

    def test_transformed_file_explained(self):
        r, = self.review([self.w('2026-05-21_KM_077_001_probe00a_pts_transformed.csv',
                                 lasagna_points(plan('probe00a')))])
        self.assertIn('_pts_transformed', r.blocks[0])

    def test_unknown_probe_blocked_never_created(self):
        r, = self.review([self.w('2026-05-21_KM_077_001_probe03a_pts.csv', lasagna_points(plan('probe00a')))])
        self.assertEqual(r.status, rta.BLOCK)
        self.assertIn('never creates', r.blocks[0])

    def test_missing_session_blocked(self):
        r, = self.review([self.w('2026-05-22_KM_077_001_probe00a_pts.csv', lasagna_points(plan('probe00a')))])
        self.assertEqual(r.status, rta.BLOCK)

    def test_server_ignoring_filters_still_matches_correctly(self):
        paths = self.four_shanks()
        reviews = self.review(paths, FakeAlyx(ignore_filters=True))
        self.assertEqual([r.pid for r in reviews], [f'pid-{EID}-probe00{s}' for s in 'abcd'])
        self.assertTrue(all(r.status == rta.OK for r in reviews), [r.blocks + r.warnings for r in reviews])

    # ---------- geometry ----------
    def test_track_on_other_probe_blocked(self):
        r, = self.review([self.w('2026-05-21_KM_077_001_probe00a_pts.csv', lasagna_points(plan('probe01b')))])
        self.assertTrue(any('closer to the plan for probe01' in b for b in r.blocks), r.blocks)

    def test_track_from_other_session_warns(self):
        alyx = FakeAlyx(plans=PLANS + PLANS2)
        r, = self.review([self.w('2026-05-21_KM_077_001_probe00a_pts.csv',
                                 lasagna_points(plan('probe00a', PLANS2)))], alyx)
        self.assertTrue(any("another session's plan" in w for w in r.warnings), r.warnings)

    def test_nearby_site_on_other_day_not_flagged(self):
        nearby = [planned(f'probe00{s}', -2288.0 + 100.0, -3106.0 - 250.0 * i, eid=EID2) for i, s in enumerate('abcd')]
        own_offset = [dict(p, x=p['x'] - 150.0) for p in PLANS]           # own plan 150 um away, other day's 100 um
        r, = self.review([self.w('2026-05-21_KM_077_001_probe00a_pts.csv', lasagna_points(plan('probe00a')))],
                         FakeAlyx(plans=own_offset + nearby))
        self.assertFalse(any("another session's plan" in w for w in r.warnings), r.warnings)

    def test_mirrored_hemisphere_blocked(self):
        only_00 = [p for p in PLANS if p['probe_name'].startswith('probe00')]
        r, = self.review([self.w('2026-05-21_KM_077_001_probe00a_pts.csv',
                                 lasagna_points(plan('probe00a'), mirror=True))], FakeAlyx(plans=only_00))
        self.assertTrue(any('hemisphere' in b for b in r.blocks), r.blocks)

    def test_steep_probe_crossing_midline_not_falsely_blocked(self):
        steep = [planned('probe00a', 700.0, -2000.0, theta=25.0, phi=0.0, depth=6000.0)]
        r, = self.review([self.w('2026-05-21_KM_077_001_probe00a_pts.csv',
                                 lasagna_points(steep[0], start=0.6))], FakeAlyx(plans=steep))
        self.assertFalse(any('hemisphere' in b for b in r.blocks), r.blocks)

    def test_far_from_plan_warns(self):
        far = dict(plan('probe00a'), x=-2288.0 - 1000.0)
        only = [far]
        r, = self.review([self.w('2026-05-21_KM_077_001_probe00a_pts.csv', lasagna_points(plan('probe00a')))],
                         FakeAlyx(plans=only))
        self.assertTrue(any('um from the planned' in w for w in r.warnings), r.warnings)

    # ---------- shank identity ----------
    def test_swapped_shank_labels_warn(self):
        reviews = self.review(self.four_shanks(swap={'a': 'd', 'd': 'a'}))
        self.assertTrue(all(any('shank order' in w for w in r.warnings) for r in reviews),
                        [r.warnings for r in reviews])

    def test_uneven_tracing_depth_no_false_swap(self):
        # shanks enter a sloped brain surface (different entry depths), and are traced over different depths
        sloped = [dict(planned(f'probe00{s}', -2288.0, -3106.0 - 250.0 * i, theta=25.0, phi=270.0),
                       z=-400.0 * i) for i, s in enumerate('abcd')]
        starts = {'a': 0.75, 'b': 0.05, 'c': 0.75, 'd': 0.05}
        paths = [self.w(f'2026-05-21_KM_077_001_probe00{s}_pts.csv',
                        lasagna_points(plan(f'probe00{s}', sloped), start=starts[s], stop=starts[s] + 0.2))
                 for s in 'abcd']
        reviews = self.review(paths, FakeAlyx(plans=sloped))
        self.assertFalse(any(any('shank order' in w for w in r.warnings) for r in reviews),
                         [r.warnings for r in reviews])

    def test_lone_shank_warns_unverified(self):
        r, = self.review([self.w('2026-05-21_KM_077_001_probe00a_pts.csv', lasagna_points(plan('probe00d')))])
        self.assertTrue(any('shank identity not verified' in w for w in r.warnings), r.warnings)

    def test_lone_shank_checked_against_stored_sibling(self):
        alyx = FakeAlyx()
        sib = next(i for i in alyx.insertions if i['name'] == 'probe00d' and i['session'] == EID)
        xyz = ALLEN.bc.i2xyz(np.c_[lasagna_points(plan('probe00a'))[:, 1],
                                   527 - lasagna_points(plan('probe00a'))[:, 0],
                                   lasagna_points(plan('probe00a'))[:, 2]])
        sib['json'] = {'xyz_picks': np.int32(xyz * 1e6).tolist()}      # d's stored track is really a's
        r, = self.review([self.w('2026-05-21_KM_077_001_probe00a_pts.csv', lasagna_points(plan('probe00d')))],
                         alyx)
        self.assertTrue(any('shank order' in w for w in r.warnings), r.warnings)

    def test_same_points_under_two_names_blocked(self):
        rows = lasagna_points(plan('probe00a'))
        reviews = self.review([self.w('2026-05-21_KM_077_001_probe00a_pts.csv', rows),
                               self.w('2026-05-21_KM_077_001_probe00b_pts.csv', rows)])
        self.assertTrue(all(any('same points' in b for b in r.blocks) for r in reviews))

    def test_duplicate_probe_files_blocked(self):
        rows = lasagna_points(plan('probe00a'))
        reviews = self.review([self.w('2026-05-21_KM_077_001_probe00a_pts.csv', rows),
                               self.w('2026-05-21_KM_077_001_probe00a_pts.csv.csv', rows)])
        self.assertTrue(all(r.status == rta.BLOCK for r in reviews))

    # ---------- existing state and bad files ----------
    def test_existing_track_and_alignment_warn(self):
        alyx = FakeAlyx()
        pid = f'pid-{EID}-probe00a'
        alyx.trajectories.append({'provenance': 'Histology track', 'session': SESSIONS[EID], 'probe_insertion': pid})
        next(i for i in alyx.insertions if i['id'] == pid)['json'] = {
            'extended_qc': {'alignment_count': 2, 'alignment_resolved': True}}
        r, = self.review([self.w('2026-05-21_KM_077_001_probe00a_pts.csv', lasagna_points(plan('probe00a')))],
                         alyx)
        self.assertEqual(r.status, rta.WARN, r.blocks)
        self.assertTrue(r.has_histology and r.has_alignment)
        self.assertTrue(any('(resolved)' in w for w in r.warnings))

    def test_two_existing_tracks_blocked(self):
        alyx = FakeAlyx()
        pid = f'pid-{EID}-probe00a'
        for _ in range(2):
            alyx.trajectories.append({'provenance': 'Histology track', 'session': SESSIONS[EID],
                                      'probe_insertion': pid})
        r, = self.review([self.w('2026-05-21_KM_077_001_probe00a_pts.csv', lasagna_points(plan('probe00a')))],
                         alyx)
        self.assertTrue(any('2 histology tracks' in b for b in r.blocks), r.blocks)

    def test_bad_points_blocked(self):
        out = lasagna_points(plan('probe00a')).copy()
        out[0, 1] = 900
        paths = [self.w('2026-05-21_KM_077_001_probe00a_pts.csv', out),
                 self.w('2026-05-21_KM_077_001_probe00b_pts.csv', lasagna_points(plan('probe00b'))[:1])]
        (self.dir / '2026-05-21_KM_077_001_probe00c_pts.csv').write_text('')
        paths.append(self.dir / '2026-05-21_KM_077_001_probe00c_pts.csv')
        reviews = self.review(paths)
        self.assertTrue(all(r.status == rta.BLOCK for r in reviews), [r.blocks for r in reviews])

    def test_picks_match_ibllib_conversion(self):
        from ibllib.pipes.histology import load_track_csv
        p = self.w('2026-05-21_KM_077_001_probe00a_pts.csv', lasagna_points(plan('probe00a')))
        r, = self.review([p])
        np.testing.assert_array_equal(r.picks, load_track_csv(p, brain_atlas=ALLEN))


def fake_register(alyx, store_picks=True, channels=True, fail=False):
    def register(pid, picks=None, one=None, overwrite=False, brain_atlas=None, **kw):
        register.calls.append((pid, picks, overwrite, brain_atlas))
        if fail:
            raise RuntimeError("simulated Alyx failure")
        alyx.trajectories = [t for t in alyx.trajectories
                             if not (t.get('probe_insertion') == pid and t['provenance'] == 'Histology track')]
        alyx.trajectories.append({'id': f'traj-{pid}', 'provenance': 'Histology track', 'probe_insertion': pid})
        if channels:
            alyx.channels += [{'trajectory_estimate': f'traj-{pid}'}] * 384
        ins = next(i for i in alyx.insertions if i['id'] == pid)
        ins['json'] = {'xyz_picks': np.int32(picks * 1e6).tolist() if store_picks else []}
    register.calls = []
    return register


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = write(self.tmp.name, '2026-05-21_KM_077_001_probe00a_pts.csv', lasagna_points(PLANS[0]))
        self.alyx = FakeAlyx()
        self.review = rta.review_all([self.path], self.alyx, ALLEN, NEEDLES)[0]
        self.one = mock.Mock(alyx=self.alyx)

    def tearDown(self):
        self.tmp.cleanup()

    def upload(self, register):
        with mock.patch('ibllib.pipes.histology.register_track', register):
            return rta.upload_track(self.review, self.one, ALLEN)

    def test_upload_calls_ibllib_once_with_reviewed_values(self):
        register = fake_register(self.alyx)
        self.assertIn('uploaded and verified (384 channels)', self.upload(register))
        (pid, picks, overwrite, atlas), = register.calls
        self.assertEqual(pid, f'pid-{EID}-probe00a')
        np.testing.assert_array_equal(picks, self.review.picks)
        self.assertTrue(overwrite)
        self.assertIs(atlas, ALLEN)

    def test_bad_readback_detected(self):
        with self.assertRaises(RuntimeError):
            self.upload(fake_register(self.alyx, store_picks=False))

    def test_missing_channels_detected(self):
        with self.assertRaisesRegex(RuntimeError, 'no channel'):
            self.upload(fake_register(self.alyx, channels=False))

    def test_changed_file_refused_before_write(self):
        self.path.write_text(self.path.read_text() + "100.0,100.0,100.0\n")
        register = fake_register(self.alyx)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            self.upload(register)
        self.assertEqual(register.calls, [])

    def test_blocked_track_never_uploaded(self):
        self.review.blocks.append('test block')
        register = fake_register(self.alyx)
        with self.assertRaises(RuntimeError):
            self.upload(register)
        self.assertEqual(register.calls, [])

    def test_old_one_refused_before_any_write(self):
        class BrokenSchemes(FakeAlyx):
            @property
            def rest_schemes(self):
                raise ConnectionError("[Errno 406] /docs: '406 Not Acceptable'")
        self.alyx.__class__ = BrokenSchemes
        register = fake_register(self.alyx)
        with self.assertRaisesRegex(RuntimeError, 'Upgrade ONE-api'):
            self.upload(register)
        self.assertEqual(register.calls, [])

    def test_log_uses_review_time_hash(self):
        rta.append_log(Path(self.tmp.name), 'https://fake.alyx', 'tester', self.review, 'started')
        rows = (Path(self.tmp.name) / rta.LOG_NAME).read_text().splitlines()
        self.assertEqual(len(rows), 2)
        self.assertIn(self.review.sha256, rows[1])


def store_picks(alyx, name, rows, eid=EID):
    """Put a Lasagna track into a fake insertion's json, exactly as ibllib's register_track stores it."""
    from ibllib.pipes.histology import load_track_csv
    with tempfile.TemporaryDirectory() as d:
        p = write(d, 'tmp_pts.csv', rows)
        picks = np.int32(load_track_csv(p, brain_atlas=ALLEN) * 1e6).tolist()
    ins = next(i for i in alyx.insertions if i['name'] == name and i['session'] == eid)
    ins['json'] = {'xyz_picks': picks}
    alyx.trajectories = [t for t in alyx.trajectories
                         if not (t.get('probe_insertion') == ins['id'] and t['provenance'] == 'Histology track')]
    alyx.trajectories.append({'id': f"traj-{ins['id']}", 'provenance': 'Histology track', 'session': SESSIONS[eid],
                              'probe_insertion': ins['id']})
    return picks


class ExportTests(unittest.TestCase):
    def setUp(self):
        import export_tracks_alyx as eta
        self.eta = eta
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / 'from_alyx'
        self.alyx = FakeAlyx()
        self.picks = {s: store_picks(self.alyx, f'probe00{s}', lasagna_points(plan(f'probe00{s}'))) for s in 'abcd'}

    def tearDown(self):
        self.tmp.cleanup()

    def export(self, **kw):
        res = self.eta.export_subject(self.alyx, 'KM_077', ALLEN, self.out, **kw)
        self.assertEqual(self.alyx.writes, [])
        return res

    def test_exported_files_reproduce_stored_picks(self):
        from ibllib.pipes.histology import load_track_csv
        res = self.export()
        written = [r for r in res if r.status == 'written']
        self.assertEqual(sorted(r.probe for r in written), [f'probe00{s}' for s in 'abcd'])
        for r in written:
            self.assertEqual(r.path.name, f'2026-05-21_KM_077_001_{r.probe}_pts.csv')
            back = np.int32(load_track_csv(r.path, brain_atlas=ALLEN) * 1e6).tolist()
            self.assertEqual(back, self.picks[r.probe[-1]])
        self.assertTrue(all(r.status.startswith('skipped: no stored') for r in res if r.status != 'written'))
        self.assertEqual(len((self.out / self.eta.MANIFEST).read_text().splitlines()), 5)

    def test_export_matches_original_lasagna_rows(self):
        original = lasagna_points(plan('probe00a'))
        self.export(probes=['probe00a'])
        exported = np.loadtxt(self.out / '2026-05-21_KM_077_001_probe00a_pts.csv', delimiter=',')
        self.assertEqual(sorted(map(tuple, exported)), sorted(map(tuple, original)))

    def test_rerun_is_unchanged_and_never_overwrites_a_different_file(self):
        self.export()
        target = self.out / '2026-05-21_KM_077_001_probe00b_pts.csv'
        target.write_text("1.0,2.0,3.0\n")                    # e.g. a file the user already relabeled
        res = {r.probe: r.status for r in self.export()}
        self.assertEqual(res['probe00a'], 'unchanged')
        self.assertTrue(res['probe00b'].startswith('skipped: a different file'))
        self.assertEqual(target.read_text(), "1.0,2.0,3.0\n")

    def test_filters(self):
        res = self.export(dates=['2026-05-21'], probes=['probe00c'])
        self.assertEqual([r.probe for r in res], ['probe00c'])
        self.assertEqual(self.export(dates=['2026-05-22']), [])

    def test_exported_files_pass_review_as_identical_to_alyx(self):
        self.export()
        reviews = rta.review_all(sorted(self.out.glob('*_pts.csv')), self.alyx, ALLEN, NEEDLES)
        for r in reviews:
            self.assertEqual(np.int32(r.picks * 1e6).tolist(), self.picks[r.shank], r.path.name)

    def test_default_out_dir_is_relative_to_current_folder(self):
        self.assertEqual(self.eta.default_out_dir('KM_077'), Path.cwd() / 'KM_077_from_alyx')
        self.assertNotIn('flatiron', Path(self.eta.__file__).read_text())

    def test_one_bad_probe_does_not_stop_the_subject(self):
        ins = next(i for i in self.alyx.insertions if i['name'] == 'probe00b' and i['session'] == EID)
        ins['json'] = {'xyz_picks': [[90_000_000, 0, -1000], [90_000_000, 0, -2000]]}   # far outside the atlas
        res = {r.probe: r.status for r in self.export()}
        self.assertTrue(res['probe00b'].startswith('skipped: stored points could not be converted'), res['probe00b'])
        self.assertEqual([p for p, s in res.items() if s == 'written'], ['probe00a', 'probe00c', 'probe00d'])


class AlreadyInAlyxTests(unittest.TestCase):
    """Downloaded files that were not relabeled must not be uploadable; relabeled ones must be."""

    def setUp(self):
        import export_tracks_alyx as eta
        self.eta = eta
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / 'from_alyx'
        self.alyx = FakeAlyx()

    def tearDown(self):
        self.tmp.cleanup()

    def store(self, mapping):
        """mapping: Alyx shank letter -> shank whose planned track is actually stored there."""
        for s, src in mapping.items():
            store_picks(self.alyx, f'probe00{s}', lasagna_points(plan(f'probe00{src}')))
        self.eta.export_subject(self.alyx, 'KM_077', ALLEN, self.out)

    def swap(self, x, y):
        a = self.out / f'2026-05-21_KM_077_001_probe00{x}_pts.csv'
        b = self.out / f'2026-05-21_KM_077_001_probe00{y}_pts.csv'
        tmp = a.with_suffix('.tmp')
        a.rename(tmp); b.rename(a); tmp.rename(b)

    def review(self):
        return {r.probe: r for r in rta.review_all(sorted(self.out.glob('*_pts.csv')), self.alyx, ALLEN, NEEDLES)}

    def test_unchanged_downloads_are_already_in_alyx_and_cannot_be_uploaded(self):
        self.store({s: s for s in 'abcd'})
        reviews = self.review()
        self.assertEqual({r.status for r in reviews.values()}, {rta.SAME})
        self.assertFalse(any('will be replaced' in d for r in reviews.values() for d in r.details))
        with self.assertRaisesRegex(RuntimeError, 'refusing'):
            rta.upload_track(reviews['probe00a'], mock.Mock(alyx=self.alyx), ALLEN)

    def test_relabeling_a_fully_reversed_probe(self):
        self.store({'a': 'd', 'b': 'c', 'c': 'b', 'd': 'a'})
        before = self.review()
        self.assertTrue(all(r.status == rta.SAME for r in before.values()))
        self.assertTrue(all(any('shank order' in d for d in r.details) for r in before.values()))
        self.swap('a', 'd'); self.swap('b', 'c')
        after = self.review()
        self.assertEqual({r.status for r in after.values()}, {rta.WARN}, {p: r.details for p, r in after.items()})
        self.assertFalse(any('shank order' in d for r in after.values() for d in r.details))

    def test_relabeling_two_shanks_keeps_the_other_two_in_the_order_check(self):
        self.store({'a': 'd', 'b': 'b', 'c': 'c', 'd': 'a'})
        self.swap('a', 'd')
        after = self.review()
        self.assertEqual({p: r.status for p, r in after.items()},
                         {'probe00a': rta.WARN, 'probe00b': rta.SAME, 'probe00c': rta.SAME, 'probe00d': rta.WARN})
        self.assertFalse(any('shank order' in d for r in after.values() for d in r.details))
        self.assertFalse(any('shank identity not verified' in d for r in after.values() for d in r.details))


class WindowTests(unittest.TestCase):
    """The real window, offscreen, with the fake Alyx and a fake ONE."""

    @classmethod
    def setUpClass(cls):
        cls.QtWidgets, _, cls.Window = rta._gui()
        cls.app = cls.QtWidgets.QApplication.instance() or cls.QtWidgets.QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        write(d, '2026-05-21_KM_077_001_probe00a_pts.csv', lasagna_points(plan('probe00a')))
        write(d, '2026-05-21_KM_077_001_probe00b_pts.csv', lasagna_points(plan('probe00b')))
        write(d, '2026-05-21_KM_077_001_probe00c_pts.csv', lasagna_points(plan('probe01c')))   # blocked
        self.alyx = FakeAlyx()
        self.win = self.Window(self.alyx, str(d))

    def tearDown(self):
        self.win.close()
        self.tmp.cleanup()

    def row_of(self, name):
        for row in range(self.win.table.rowCount()):
            if self.win.table.item(row, 1).text() == name:
                return row

    def test_blocked_row_cannot_be_ticked(self):
        from qtpy import QtCore
        self.win._set_all(True)
        blocked = self.win.table.item(self.row_of('2026-05-21_KM_077_001_probe00c_pts.csv'), 0)
        self.assertFalse(blocked.flags() & QtCore.Qt.ItemIsUserCheckable)
        self.assertEqual(len(self.win._checked_paths()), 2)

    def test_only_ticked_files_uploaded_and_logged(self):
        from qtpy import QtCore
        self.win.table.item(self.row_of('2026-05-21_KM_077_001_probe00b_pts.csv'), 0).setCheckState(
            QtCore.Qt.Checked)
        register = fake_register(self.alyx)
        one = mock.Mock(alyx=self.alyx)
        self.alyx.is_logged_in = True
        with mock.patch('ibllib.pipes.histology.register_track', register), \
                mock.patch('one.api.ONE', return_value=one), \
                mock.patch.object(self.QtWidgets.QDialog, 'exec_', return_value=self.QtWidgets.QDialog.Accepted), \
                mock.patch.object(self.QtWidgets.QMessageBox, 'information'):
            self.win._upload()
        self.assertEqual([c[0] for c in register.calls], [f'pid-{EID}-probe00b'])
        log = (Path(self.tmp.name) / rta.LOG_NAME).read_text().splitlines()
        self.assertEqual([line.split(',')[-1] for line in log[1:]],
                         ['started', 'uploaded and verified (384 channels)'])
        self.assertTrue(all(c.isEnabled() for c in self.win.controls))

    def test_download_button_writes_to_from_alyx_folder(self):
        sub = Path(self.tmp.name) / 'KM_077' / 'electrodetracks'
        sub.mkdir(parents=True)
        self.win.folder = sub
        store_picks(self.alyx, 'probe00a', lasagna_points(plan('probe00a')))
        with mock.patch.object(self.QtWidgets.QInputDialog, 'getText', return_value=('KM_077', True)) as ask, \
                mock.patch.object(self.QtWidgets.QMessageBox, 'question',
                                  return_value=self.QtWidgets.QMessageBox.Yes):
            self.win._download()
        self.assertEqual(ask.call_args.kwargs.get('text'), 'KM_077')
        self.assertTrue((sub / 'from_alyx' / '2026-05-21_KM_077_001_probe00a_pts.csv').exists())
        self.assertEqual(self.win.folder, sub / 'from_alyx')
        self.assertEqual(list(self.win.reviews), [str(sub / 'from_alyx' / '2026-05-21_KM_077_001_probe00a_pts.csv')])
        self.assertTrue(all(c.isEnabled() for c in self.win.controls))

    def test_download_asks_where_when_folder_is_not_the_subjects(self):
        store_picks(self.alyx, 'probe00a', lasagna_points(plan('probe00a')))
        elsewhere = Path(self.tmp.name) / 'chosen'
        elsewhere.mkdir()
        Q = self.QtWidgets
        with mock.patch.object(Q.QInputDialog, 'getText', return_value=('KM_077', True)), \
                mock.patch.object(Q.QFileDialog, 'getExistingDirectory', return_value='') as pick:
            self.win._download()
        pick.assert_called_once()
        self.assertFalse(any(Path(self.tmp.name).rglob('from_alyx')))
        with mock.patch.object(Q.QInputDialog, 'getText', return_value=('KM_077', True)), \
                mock.patch.object(Q.QFileDialog, 'getExistingDirectory', return_value=str(elsewhere)), \
                mock.patch.object(Q.QMessageBox, 'question', return_value=Q.QMessageBox.No):
            self.win._download()
        self.assertTrue((elsewhere / 'from_alyx' / '2026-05-21_KM_077_001_probe00a_pts.csv').exists())

    def test_already_in_alyx_rows_cannot_be_ticked(self):
        import export_tracks_alyx as eta
        for s in 'abcd':
            store_picks(self.alyx, f'probe00{s}', lasagna_points(plan(f'probe00{s}')))
        out = Path(self.tmp.name) / 'from_alyx'
        eta.export_subject(self.alyx, 'KM_077', ALLEN, out)
        self.win.folder = out
        self.win._review(sorted(out.glob('*_pts.csv')))
        self.win._set_all(True)
        self.assertEqual(self.win._checked_paths(), [])
        self.assertTrue(all(r.status == rta.SAME for r in self.win.reviews.values()))

    def test_login_mismatch_uploads_nothing(self):
        from qtpy import QtCore
        self.win._set_all(True)
        register = fake_register(self.alyx)
        other = mock.Mock(alyx=mock.Mock(is_logged_in=True, user='someone_else'))
        with mock.patch('ibllib.pipes.histology.register_track', register), \
                mock.patch('one.api.ONE', return_value=other), \
                mock.patch.object(self.QtWidgets.QDialog, 'exec_', return_value=self.QtWidgets.QDialog.Accepted), \
                mock.patch.object(self.QtWidgets.QMessageBox, 'critical') as crit:
            self.win._upload()
        self.assertEqual(register.calls, [])
        crit.assert_called_once()


if __name__ == '__main__':
    unittest.main()
