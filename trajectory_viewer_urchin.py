"""
trajectory_viewer_urchin.py
============================
Standalone GUI that sends micro-manipulator trajectories to the Urchin / Pinpoint 3-D brain viewer
and displays them on coronal/sagittal/horizontal atlas slices.

Run
---
    C:\\Users\\kimil\\anaconda3\\envs\\iblenv\\python.exe trajectory_viewer_urchin.py

Requires iblenv conda env (iblatlas, one.api, oursin, qtpy, matplotlib).
"""
from __future__ import annotations

import sys

import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from iblatlas.atlas import NeedlesAtlas, Insertion
from one.webclient import AlyxClient
import oursin as urchin


DEFAULT_ALYX_URL = "https://alyx.internationalbrainlab.org"

SHANK_COLORS = {
    'a': '#e74c3c',
    'b': '#3498db',
    'c': '#2ecc71',
    'd': '#f39c12',
}
PROBE_COLORS = ['#9b59b6', '#1abc9c', '#e67e22', '#2c3e50']

INTERSECT_MARKER_SIZE = 10
PROJ_LINE_ALPHA = 0.25   # faint projected trajectory always visible on each panel
SCROLL_STEP_UM = 50   # µm per scroll tick


def _probe_in_range(xyz_um: np.ndarray, axis: int, val: float) -> bool:
    """Return True if the slice at `val` is within the probe's extent on `axis`
    (using the same snap tolerance as _line_slice_intersect)."""
    v0, v1 = xyz_um[0, axis], xyz_um[1, axis]
    if abs(v1 - v0) < SCROLL_STEP_UM:
        return abs((v0 + v1) / 2.0 - val) <= SCROLL_STEP_UM
    return min(v0, v1) <= val <= max(v0, v1)


def _line_slice_intersect(xyz_um: np.ndarray, axis: int, val: float):
    """Return [ML, AP, DV] point where the probe line intersects an axis-aligned slice.

    xyz_um : shape (2, 3), [entry, tip] x [ML, AP, DV] in µm
    axis   : 0=ML, 1=AP, 2=DV
    val    : slice position in µm along that axis

    For probes that barely change along the slice axis (nearly parallel to the plane),
    a snap tolerance of one scroll step is used so the dot still appears.
    Returns None if no intersection or snap is possible.
    """
    v0, v1 = xyz_um[0, axis], xyz_um[1, axis]
    extent = abs(v1 - v0)

    if extent < SCROLL_STEP_UM:
        # Probe barely changes in this direction -- snap if slice is within one step
        vmean = (v0 + v1) / 2.0
        if abs(val - vmean) <= SCROLL_STEP_UM:
            return (xyz_um[0] + xyz_um[1]) / 2.0
        return None

    t = (val - v0) / (v1 - v0)
    if not (0.0 <= t <= 1.0):
        return None
    return xyz_um[0] + t * (xyz_um[1] - xyz_um[0])


def _probe_root(probe_name: str) -> str:
    """Strip shank-letter suffix: 'probe01a' -> 'probe01'."""
    if probe_name and probe_name[-1].lower() in SHANK_COLORS:
        return probe_name[:-1]
    return probe_name


def _hex_to_urchin(hex_color: str) -> list[float]:
    """'#rrggbb' -> [r, g, b] in [0,1]."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return [r / 255.0, g / 255.0, b / 255.0]


# ---------------------------------------------------------------------------
# Login dialog
# ---------------------------------------------------------------------------
class LoginDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect to Alyx")
        self.setMinimumWidth(380)
        self.alyx: AlyxClient | None = None

        try:
            alyx = AlyxClient(base_url=DEFAULT_ALYX_URL, silent=True)
            alyx.authenticate()
            if alyx.is_logged_in:
                self.alyx = alyx
                QtCore.QTimer.singleShot(0, self.accept)
                return
        except Exception:
            pass

        lbl_url = QtWidgets.QLabel("Alyx URL:")
        self.url_edit = QtWidgets.QLineEdit(DEFAULT_ALYX_URL)
        lbl_user = QtWidgets.QLabel("Username:")
        self.user_edit = QtWidgets.QLineEdit()
        lbl_pw = QtWidgets.QLabel("Password:")
        self.pw_edit = QtWidgets.QLineEdit()
        self.pw_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.status_lbl = QtWidgets.QLabel("")
        self.status_lbl.setWordWrap(True)

        btn_connect = QtWidgets.QPushButton("Connect")
        btn_connect.setDefault(True)
        btn_connect.clicked.connect(self._do_connect)
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        form = QtWidgets.QFormLayout()
        form.addRow(lbl_url, self.url_edit)
        form.addRow(lbl_user, self.user_edit)
        form.addRow(lbl_pw, self.pw_edit)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_connect)

        vlay = QtWidgets.QVBoxLayout(self)
        vlay.addLayout(form)
        vlay.addWidget(self.status_lbl)
        vlay.addLayout(btn_row)

    def _do_connect(self):
        url = self.url_edit.text().strip()
        user = self.user_edit.text().strip()
        pw = self.pw_edit.text()
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


# ---------------------------------------------------------------------------
# Data builder
# ---------------------------------------------------------------------------
def _build_entry(traj: dict, session_label: str, ba: NeedlesAtlas,
                 color: str) -> dict | None:
    """Convert an Alyx trajectory dict to a rendering-ready entry dict."""
    try:
        ins = Insertion.from_dict(traj, brain_atlas=ba)
    except Exception:
        return None

    xyz_tip_m = ins.tip
    xyz_entry_m = ins.xyz[0]

    # CCF positions: ba.xyz2ccf returns [ML, AP, DV] in µm
    tip_ccf = ba.xyz2ccf(xyz_tip_m)
    entry_ccf = ba.xyz2ccf(xyz_entry_m)

    # Urchin world-space mm, reordered [ML,AP,DV] -> [AP,ML,DV]
    position_mm = [tip_ccf[1] / 1000.0, tip_ccf[0] / 1000.0, tip_ccf[2] / 1000.0]
    entry_mm    = [entry_ccf[1] / 1000.0, entry_ccf[0] / 1000.0, entry_ccf[2] / 1000.0]

    depth_mm = float(np.linalg.norm(
        np.array(entry_ccf) - np.array(tip_ccf))) / 1000.0

    phi_deg   = float(traj.get('phi', 0.0))
    theta_deg = float(traj.get('theta', 0.0))
    roll_deg  = float(traj.get('roll', 0.0))

    # IBL phi=0 tilts toward -ML; Urchin az=0 faces +AP axis.
    # IBL theta=0 is vertical; Urchin elevation=0 is horizontal (docstring wrong).
    urchin_az   = (270.0 - phi_deg) % 360.0
    urchin_elev = 90.0 - theta_deg
    angles = [urchin_az, urchin_elev, roll_deg]

    # For slice plotting: [entry, tip] x [ML, AP, DV] in µm
    xyz_um = ins.xyz * 1e6

    return {
        'session_label': session_label,
        'probe_name':    traj.get('probe_name', ''),
        'color':         color,
        'position':      position_mm,
        'entry':         entry_mm,
        'depth':         depth_mm,
        'angles':        angles,
        'xyz_um':        xyz_um,
        'visible':       True,
        'probe_obj':     None,
        'particle_obj':  None,
    }


# ---------------------------------------------------------------------------
# Main viewer window
# ---------------------------------------------------------------------------
class UrchinViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, alyx: AlyxClient):
        super().__init__()
        self.alyx = alyx
        self.setWindowTitle("Urchin + Slice Trajectory Viewer")
        self.resize(1600, 900)

        self.statusBar().showMessage("Loading NeedlesAtlas...")
        QtWidgets.QApplication.processEvents()
        self.ba = NeedlesAtlas()
        self.statusBar().showMessage("Atlas loaded.")

        self._sessions_map: dict[str, str] = {}
        self._data: list[dict] = []
        self._urchin_ready = False

        # Debounce timer: fires _redraw() 100 ms after the last scroll event
        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(100)
        self._redraw_timer.timeout.connect(self._redraw)

        # Per-panel zoom state: None = auto, else (xlim, ylim)
        self._zoom: dict[str, tuple | None] = {
            'cor': None, 'sag': None, 'hor': None}

        # Crosshair navigation state (plain left drag)
        self._crosshair_ax = None

        # Pan state (Ctrl+left drag)
        self._pan_ax = None
        self._pan_key = None
        self._pan_inv = None    # frozen inverse transform at press time
        self._pan_data0 = None  # data coords at press
        self._pan_xlim0 = None
        self._pan_ylim0 = None

        self._build_ui()
        self._populate_subjects()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        # Top Urchin controls
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(6)

        self.open_btn = QtWidgets.QPushButton("1. Open Urchin")
        self.open_btn.setMinimumHeight(28)
        self.open_btn.clicked.connect(self._open_urchin)
        top.addWidget(self.open_btn)

        top.addSpacing(8)
        top.addWidget(QtWidgets.QLabel("Brain alpha:"))
        self.alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha_slider.setRange(0, 50)
        self.alpha_slider.setValue(10)
        self.alpha_slider.setFixedWidth(100)
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        top.addWidget(self.alpha_slider)
        self.alpha_lbl = QtWidgets.QLabel("0.10")
        self.alpha_lbl.setFixedWidth(32)
        top.addWidget(self.alpha_lbl)
        top.addStretch()
        root.addLayout(top)

        # Main horizontal splitter: sidebar | canvas.
        # stretch=1 gives this all vertical space as the window grows.
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        # --- Sidebar ---
        sidebar = QtWidgets.QWidget()
        sidebar.setMinimumWidth(220)
        vlay = QtWidgets.QVBoxLayout(sidebar)
        vlay.setSpacing(4)

        vlay.addWidget(self._hdr("Subject"))
        self.subj_combo = QtWidgets.QComboBox()
        self.subj_combo.setMinimumWidth(140)
        self.subj_combo.currentIndexChanged.connect(self._on_subject_changed)
        vlay.addWidget(self.subj_combo)

        vlay.addSpacing(4)
        vlay.addWidget(self._hdr("Session"))
        self.sess_combo = QtWidgets.QComboBox()
        self.sess_combo.setMinimumWidth(180)
        vlay.addWidget(self.sess_combo)

        vlay.addSpacing(4)
        btn_row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("2. Add Session")
        self.add_btn.setMinimumHeight(28)
        self.add_btn.clicked.connect(self._add_session)
        self.load_all_btn = QtWidgets.QPushButton("Load All")
        self.load_all_btn.setMinimumHeight(28)
        self.load_all_btn.clicked.connect(self._load_all_sessions)
        self.clear_btn = QtWidgets.QPushButton("Clear All")
        self.clear_btn.setMinimumHeight(28)
        self.clear_btn.clicked.connect(self._clear_sessions)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.load_all_btn)
        btn_row.addWidget(self.clear_btn)
        vlay.addLayout(btn_row)

        vlay.addSpacing(6)
        sep1 = QtWidgets.QFrame()
        sep1.setFrameShape(QtWidgets.QFrame.HLine)
        sep1.setFrameShadow(QtWidgets.QFrame.Sunken)
        vlay.addWidget(sep1)
        vlay.addWidget(self._hdr("Slice positions"))
        vlay.addWidget(QtWidgets.QLabel(
            "<small><i>scroll: navigate  |  drag: crosshair  |  Ctrl+drag: pan</i></small>"))

        vlay.addWidget(QtWidgets.QLabel("Coronal AP (µm):"))
        self.ap_spin = self._make_spin(-8000, 5000, SCROLL_STEP_UM, " µm")
        self.ap_spin.valueChanged.connect(self._redraw)
        vlay.addWidget(self.ap_spin)

        vlay.addWidget(QtWidgets.QLabel("Sagittal ML (µm):"))
        self.ml_spin = self._make_spin(-5500, 5500, SCROLL_STEP_UM, " µm")
        self.ml_spin.valueChanged.connect(self._redraw)
        vlay.addWidget(self.ml_spin)

        vlay.addWidget(QtWidgets.QLabel("Horizontal DV (µm):"))
        self.dv_spin = self._make_spin(-7000, 0, SCROLL_STEP_UM, " µm")
        self.dv_spin.valueChanged.connect(self._redraw)
        vlay.addWidget(self.dv_spin)

        vlay.addSpacing(6)
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        sep2.setFrameShadow(QtWidgets.QFrame.Sunken)
        vlay.addWidget(sep2)

        hdr_row = QtWidgets.QHBoxLayout()
        hdr_row.addWidget(self._hdr("Probes / Legend"))
        hdr_row.addWidget(QtWidgets.QLabel(
            "<small><i>dbl-click swatch to recolor  |  ● entry  ▼ tip</i></small>"))
        hdr_row.addStretch()
        vlay.addLayout(hdr_row)

        self.probe_tree = QtWidgets.QTreeWidget()
        self.probe_tree.setColumnCount(2)
        self.probe_tree.setHeaderLabels(["Session / Probe / Shank", ""])
        self.probe_tree.header().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch)
        self.probe_tree.header().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Fixed)
        self.probe_tree.header().resizeSection(1, 38)
        self.probe_tree.setMinimumHeight(80)
        self.probe_tree.itemChanged.connect(self._on_tree_changed)
        self.probe_tree.itemDoubleClicked.connect(self._on_tree_dbl_click)
        vlay.addWidget(self.probe_tree)

        ca_row = QtWidgets.QHBoxLayout()
        ca = QtWidgets.QPushButton("Check All")
        ca.setFixedHeight(22)
        ca.clicked.connect(lambda: self._set_all_visible(True))
        ua = QtWidgets.QPushButton("Uncheck All")
        ua.setFixedHeight(22)
        ua.clicked.connect(lambda: self._set_all_visible(False))
        ca_row.addWidget(ca)
        ca_row.addWidget(ua)
        vlay.addLayout(ca_row)

        urchin_note = QtWidgets.QLabel(
            "<small><i>Urchin view: posterior view — left=left, anterior away</i></small>")
        urchin_note.setWordWrap(True)
        vlay.addWidget(urchin_note)

        vlay.addStretch()
        splitter.addWidget(sidebar)

        # --- Matplotlib canvas ---
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        self.ax_cor = self.fig.add_subplot(gs[0, 0])
        self.ax_sag = self.fig.add_subplot(gs[0, 1])
        self.ax_hor = self.fig.add_subplot(gs[1, :])
        self._label_axes()
        self.canvas.draw()
        splitter.addWidget(self.canvas)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 1320])

        # Scroll to navigate slices / zoom panels
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
        self.canvas.mpl_connect('button_release_event', self._on_canvas_release)

    @staticmethod
    def _hdr(text: str) -> QtWidgets.QLabel:
        return QtWidgets.QLabel(f"<b>{text}</b>")

    @staticmethod
    def _make_spin(lo: float, hi: float, step: float,
                   suffix: str = "") -> QtWidgets.QDoubleSpinBox:
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(0)
        sp.setSuffix(suffix)
        return sp

    def _label_axes(self):
        self.ax_cor.set_title("Coronal (AP slice)", fontsize=10)
        self.ax_cor.set_xlabel("ML (µm)", fontsize=8)
        self.ax_cor.set_ylabel("DV (µm)", fontsize=8)
        self.ax_sag.set_title("Sagittal (ML slice)", fontsize=10)
        self.ax_sag.set_xlabel("AP (µm)", fontsize=8)
        self.ax_sag.set_ylabel("DV (µm)", fontsize=8)
        self.ax_hor.set_title("Horizontal (DV slice)", fontsize=10)
        self.ax_hor.set_xlabel("ML (µm)", fontsize=8)
        self.ax_hor.set_ylabel("AP (µm)", fontsize=8)

    # ------------------------------------------------------------------
    # Alyx
    # ------------------------------------------------------------------
    def _populate_subjects(self):
        self.statusBar().showMessage("Fetching subjects...")
        QtWidgets.QApplication.processEvents()
        try:
            trajs = self.alyx.rest('trajectories', 'list',
                                   provenance='Micro-manipulator', limit=1000)
            subjects = sorted(set(t['session']['subject'] for t in trajs))
            self.subj_combo.blockSignals(True)
            self.subj_combo.clear()
            self.subj_combo.addItems(subjects)
            self.subj_combo.blockSignals(False)
            self.statusBar().showMessage(f"Found {len(subjects)} subject(s).")
            if subjects:
                self._on_subject_changed(0)
        except Exception as exc:
            self.statusBar().showMessage(f"Error: {exc}")

    def _on_subject_changed(self, idx):
        if idx < 0:
            return
        subject = self.subj_combo.currentText()
        try:
            trajs = self.alyx.rest('trajectories', 'list',
                                   provenance='Micro-manipulator',
                                   subject=subject, limit=1000)
            sessions: dict[str, str] = {}
            for t in trajs:
                s = t['session']
                label = f"{s['start_time'][:10]} #{s['number']}"
                sessions[label] = s['id']
            self._sessions_map = sessions
            self.sess_combo.clear()
            self.sess_combo.addItems(sorted(sessions.keys()))
            self.statusBar().showMessage(
                f"Found {len(sessions)} session(s) for {subject}.")
        except Exception as exc:
            self.statusBar().showMessage(f"Error: {exc}")

    @staticmethod
    def _shank_color(probe_name: str, fallback_idx: int) -> str:
        return '#e74c3c'

    def _add_session(self):
        sess_label = self.sess_combo.currentText()
        if not sess_label:
            return
        sess_id = self._sessions_map.get(sess_label)
        if not sess_id:
            return
        if sess_label in {e['session_label'] for e in self._data}:
            self.statusBar().showMessage(f"{sess_label} is already loaded.")
            return

        self.statusBar().showMessage(f"Fetching {sess_label}...")
        QtWidgets.QApplication.processEvents()
        try:
            trajs = self.alyx.rest('trajectories', 'list',
                                   provenance='Micro-manipulator', session=sess_id)
        except Exception as exc:
            self.statusBar().showMessage(f"Error: {exc}")
            return

        if not trajs:
            self.statusBar().showMessage(f"No trajectories for {sess_label}.")
            return

        trajs_sorted = sorted(trajs, key=lambda t: t.get('probe_name', ''))
        first_load = not self._data
        existing_count = len(self._data)

        for i, traj in enumerate(trajs_sorted):
            probe_name = traj.get('probe_name', f'probe{i}')
            color = self._shank_color(probe_name, existing_count + i)
            entry = _build_entry(traj, sess_label, self.ba, color)
            if entry:
                self._data.append(entry)

        # Auto-center slice planes on first load
        if first_load:
            xyz_list = [e['xyz_um'] for e in self._data]
            if xyz_list:
                all_xyz = np.concatenate(xyz_list, axis=0)  # (N*2, 3)
                for sp in (self.ap_spin, self.ml_spin, self.dv_spin):
                    sp.blockSignals(True)
                self.ap_spin.setValue(round(float(np.mean(all_xyz[:, 1])), 0))
                self.ml_spin.setValue(round(float(np.mean(all_xyz[:, 0])), 0))
                self.dv_spin.setValue(round(float(np.mean(all_xyz[:, 2])), 0))
                for sp in (self.ap_spin, self.ml_spin, self.dv_spin):
                    sp.blockSignals(False)

        self._rebuild_tree()
        self._redraw()
        if self._urchin_ready:
            self._render_probes()
        self.statusBar().showMessage(
            f"Added {len(trajs_sorted)} probe(s) from {sess_label}. "
            f"{len(self._data)} total.")

    def _load_all_sessions(self):
        already_loaded = {e['session_label'] for e in self._data}
        to_load = sorted(
            [(lbl, sid) for lbl, sid in self._sessions_map.items()
             if lbl not in already_loaded]
        )
        if not to_load:
            self.statusBar().showMessage("All sessions already loaded.")
            return

        first_load = not self._data
        total_added = 0
        for i, (sess_label, sess_id) in enumerate(to_load):
            self.statusBar().showMessage(
                f"Fetching {sess_label}  ({i + 1}/{len(to_load)})...")
            QtWidgets.QApplication.processEvents()
            try:
                trajs = self.alyx.rest('trajectories', 'list',
                                       provenance='Micro-manipulator',
                                       session=sess_id)
            except Exception as exc:
                self.statusBar().showMessage(f"Error on {sess_label}: {exc}")
                continue
            if not trajs:
                continue
            trajs_sorted = sorted(trajs, key=lambda t: t.get('probe_name', ''))
            existing_count = len(self._data)
            for j, traj in enumerate(trajs_sorted):
                probe_name = traj.get('probe_name', f'probe{j}')
                color = self._shank_color(probe_name, existing_count + j)
                entry = _build_entry(traj, sess_label, self.ba, color)
                if entry:
                    self._data.append(entry)
                    total_added += 1

        if first_load and self._data:
            xyz_list = [e['xyz_um'] for e in self._data]
            all_xyz = np.concatenate(xyz_list, axis=0)
            for sp in (self.ap_spin, self.ml_spin, self.dv_spin):
                sp.blockSignals(True)
            self.ap_spin.setValue(round(float(np.mean(all_xyz[:, 1])), 0))
            self.ml_spin.setValue(round(float(np.mean(all_xyz[:, 0])), 0))
            self.dv_spin.setValue(round(float(np.mean(all_xyz[:, 2])), 0))
            for sp in (self.ap_spin, self.ml_spin, self.dv_spin):
                sp.blockSignals(False)

        self._rebuild_tree()
        self._redraw()
        if self._urchin_ready:
            self._render_probes()
        self.statusBar().showMessage(
            f"Loaded {total_added} probe(s) across {len(to_load)} session(s). "
            f"{len(self._data)} total.")

    def _clear_sessions(self):
        self._delete_all_probe_objs()
        self._data.clear()
        self.probe_tree.blockSignals(True)
        self.probe_tree.clear()
        self.probe_tree.blockSignals(False)
        for ax in (self.ax_cor, self.ax_sag, self.ax_hor):
            ax.cla()
        self._label_axes()
        self.canvas.draw()
        self.statusBar().showMessage("Cleared.")

    def _delete_all_probe_objs(self):
        for entry in self._data:
            if entry.get('probe_obj') is not None:
                try:
                    entry['probe_obj'].delete()
                except Exception:
                    pass
                entry['probe_obj'] = None
            if entry.get('particle_obj') is not None:
                try:
                    entry['particle_obj'].delete()
                except Exception:
                    pass
                entry['particle_obj'] = None

    # ------------------------------------------------------------------
    # Urchin
    # ------------------------------------------------------------------
    def _open_urchin(self):
        self.statusBar().showMessage("Connecting to Urchin renderer...")
        QtWidgets.QApplication.processEvents()
        try:
            urchin.setup()
        except Exception as exc:
            self.statusBar().showMessage(f"Urchin setup error: {exc}")
            return
        QtCore.QTimer.singleShot(1500, self._urchin_load_atlas)

    def _urchin_load_atlas(self):
        self.statusBar().showMessage("Loading CCF25 atlas in Urchin...")
        try:
            urchin.ccf25.load()
        except Exception:
            pass
        QtCore.QTimer.singleShot(3000, self._urchin_post_setup)

    def _urchin_post_setup(self):
        try:
            alpha = self.alpha_slider.value() / 100.0
            urchin.ccf25.set_visibilities([urchin.ccf25.grey], [True])
            urchin.ccf25.set_materials([urchin.ccf25.grey], ['transparent-lit'])
            urchin.ccf25.set_alphas([urchin.ccf25.grey], [alpha])
        except Exception:
            pass
        # Matches horizontal slice panel orientation.
        # Pinpoint displays rotation as [yaw, pitch, roll]; oursin API takes [pitch, yaw, roll].
        # [180, 0, 0] = pitch 180 -> Pinpoint shows "0, 180, 0".
        try:
            cam = urchin.camera.Camera(main=True)
            cam.set_rotation([180, 0, 0])
        except Exception:
            pass
        self._urchin_ready = True
        self.open_btn.setText("Urchin Open")
        self.statusBar().showMessage(
            "Urchin ready. Add a session to render probes.")
        if self._data:
            self._render_probes()

    def _on_alpha_changed(self, val: int):
        alpha = val / 100.0
        self.alpha_lbl.setText(f"{alpha:.2f}")
        if self._urchin_ready:
            try:
                urchin.ccf25.set_alphas([urchin.ccf25.grey], [alpha])
            except Exception:
                pass

    def _render_probes(self):
        self._delete_all_probe_objs()

        visible = [e for e in self._data if e.get('visible', True)]
        if not visible:
            return

        errors = 0
        for entry in visible:
            try:
                p = urchin.probes.Probe(
                    color=entry['color'],
                    position=entry['position'],   # [AP, ML, DV] in mm
                    angle=entry['angles'],         # [urchin_az, urchin_elev, roll]
                    scale=[0.07, entry['depth'], 0.02],
                )
                entry['probe_obj'] = p
            except Exception:
                errors += 1

            # Shank-A bubble marker at brain-surface entry point
            pn = entry['probe_name']
            if pn and pn[-1].lower() == 'a':
                try:
                    bubble = urchin.probes.Probe(
                        color=entry['color'],
                        position=entry['entry'],
                        angle=entry['angles'],
                        scale=[0.25, 0.05, 0.25],
                    )
                    entry['particle_obj'] = bubble
                except Exception:
                    pass

        n = len(visible) - errors
        msg = f"Rendered {n} probe(s) in Urchin."
        if errors:
            msg += f" ({errors} failed)"
        self.statusBar().showMessage(msg)

    # ------------------------------------------------------------------
    # Slice panels
    # ------------------------------------------------------------------
    def _ax_key(self, ax):
        if ax is self.ax_cor: return 'cor'
        if ax is self.ax_sag: return 'sag'
        if ax is self.ax_hor: return 'hor'
        return None

    def _on_scroll(self, event):
        if event.inaxes is None:
            return

        mods = QtWidgets.QApplication.keyboardModifiers()
        ctrl_held  = bool(mods & QtCore.Qt.ControlModifier)
        shift_held = bool(mods & QtCore.Qt.ShiftModifier)

        # Ctrl+scroll: zoom around mouse cursor
        if ctrl_held:
            key = self._ax_key(event.inaxes)
            if key is None or event.xdata is None or event.ydata is None:
                return
            ax = event.inaxes
            factor = 1.25 if event.button == 'down' else 1.0 / 1.25
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xm, ym = event.xdata, event.ydata
            new_xlim = (xm + (xlim[0] - xm) * factor,
                        xm + (xlim[1] - xm) * factor)
            new_ylim = (ym + (ylim[0] - ym) * factor,
                        ym + (ylim[1] - ym) * factor)
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
            self._zoom[key] = (new_xlim, new_ylim)
            self.canvas.draw_idle()
            return

        # Regular / shift scroll: navigate slices
        step = SCROLL_STEP_UM if event.button == 'up' else -SCROLL_STEP_UM
        if shift_held:
            step *= 10
        spin = None
        if event.inaxes is self.ax_cor:
            spin = self.ap_spin
        elif event.inaxes is self.ax_sag:
            spin = self.ml_spin
        elif event.inaxes is self.ax_hor:
            spin = self.dv_spin
        if spin is None:
            return
        spin.blockSignals(True)
        spin.setValue(spin.value() + step)
        spin.blockSignals(False)
        self._redraw_timer.start()

    def _on_canvas_press(self, event):
        if event.button != 1 or event.inaxes is None:
            return
        if event.dblclick:
            self._pan_ax = None
            self._crosshair_ax = None
            key = self._ax_key(event.inaxes)
            if key is not None:
                self._zoom[key] = None
                self._redraw()
            return
        mods = QtWidgets.QApplication.keyboardModifiers()
        if bool(mods & QtCore.Qt.ControlModifier):
            # Ctrl+drag: pan within the panel
            key = self._ax_key(event.inaxes)
            if key is None:
                return
            ax = event.inaxes
            self._pan_ax = ax
            self._pan_key = key
            self._pan_inv = ax.transData.inverted().frozen()
            self._pan_data0 = self._pan_inv.transform((event.x, event.y))
            self._pan_xlim0 = ax.get_xlim()
            self._pan_ylim0 = ax.get_ylim()
        else:
            # Plain drag: crosshair navigation (update the other two panels)
            self._crosshair_ax = event.inaxes
            if event.xdata is not None and event.ydata is not None:
                self._update_crosshair(event.inaxes, event.xdata, event.ydata)

    def _on_canvas_motion(self, event):
        if self._pan_ax is not None:
            if event.x is None or event.y is None:
                return
            p_curr = self._pan_inv.transform((event.x, event.y))
            dx = self._pan_data0[0] - p_curr[0]
            dy = self._pan_data0[1] - p_curr[1]
            new_xlim = (self._pan_xlim0[0] + dx, self._pan_xlim0[1] + dx)
            new_ylim = (self._pan_ylim0[0] + dy, self._pan_ylim0[1] + dy)
            self._pan_ax.set_xlim(new_xlim)
            self._pan_ax.set_ylim(new_ylim)
            self._zoom[self._pan_key] = (new_xlim, new_ylim)
            self.canvas.draw_idle()
        elif self._crosshair_ax is not None:
            if event.inaxes is not self._crosshair_ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            self._update_crosshair(self._crosshair_ax, event.xdata, event.ydata)

    def _update_crosshair(self, ax, xdata, ydata):
        if ax is self.ax_cor:
            # X=ML, Y=DV
            self.ml_spin.blockSignals(True)
            self.dv_spin.blockSignals(True)
            self.ml_spin.setValue(round(xdata))
            self.dv_spin.setValue(round(ydata))
            self.ml_spin.blockSignals(False)
            self.dv_spin.blockSignals(False)
        elif ax is self.ax_sag:
            # X=AP, Y=DV
            self.ap_spin.blockSignals(True)
            self.dv_spin.blockSignals(True)
            self.ap_spin.setValue(round(xdata))
            self.dv_spin.setValue(round(ydata))
            self.ap_spin.blockSignals(False)
            self.dv_spin.blockSignals(False)
        elif ax is self.ax_hor:
            # X=ML, Y=AP (Y is inverted but data coords are unchanged)
            self.ml_spin.blockSignals(True)
            self.ap_spin.blockSignals(True)
            self.ml_spin.setValue(round(xdata))
            self.ap_spin.setValue(round(ydata))
            self.ml_spin.blockSignals(False)
            self.ap_spin.blockSignals(False)
        self._redraw_timer.start()

    def _on_canvas_release(self, event):
        if event.button == 1:
            self._crosshair_ax = None
            self._pan_ax = None
            self._pan_key = None
            self._pan_inv = None
            self._pan_data0 = None
            self._pan_xlim0 = None
            self._pan_ylim0 = None

    def _redraw(self):
        if not self._data:
            return

        ap_m = self.ap_spin.value() / 1e6
        ml_m = self.ml_spin.value() / 1e6
        dv_m = self.dv_spin.value() / 1e6

        for ax in (self.ax_cor, self.ax_sag, self.ax_hor):
            ax.cla()

        self.ba.plot_cslice(ap_m, ax=self.ax_cor)
        self.ba.plot_sslice(ml_m, ax=self.ax_sag)
        self.ax_sag.invert_xaxis()
        self.ba.plot_hslice(dv_m, ax=self.ax_hor)
        self.ax_hor.invert_yaxis()

        self._label_axes()
        self.ax_cor.set_title(
            f"Coronal  AP = {self.ap_spin.value():.0f} µm", fontsize=10)
        self.ax_sag.set_title(
            f"Sagittal  ML = {self.ml_spin.value():.0f} µm", fontsize=10)
        self.ax_hor.set_title(
            f"Horizontal  DV = {self.dv_spin.value():.0f} µm", fontsize=10)

        ap_val = self.ap_spin.value()
        ml_val = self.ml_spin.value()
        dv_val = self.dv_spin.value()

        for idx, entry in enumerate(self._data):
            if not entry.get('visible', True):
                continue
            xyz = entry['xyz_um']   # (2,3): [entry, tip] x [ML, AP, DV]
            color = entry['color']
            ltr = entry['probe_name'][-1].upper() if entry['probe_name'] else ''
            ann_kw = dict(xytext=(3, 3), textcoords='offset points',
                          fontsize=7, color=color, fontweight='bold', zorder=5)
            proj_kw = dict(color=color, linewidth=2.0,
                           alpha=PROJ_LINE_ALPHA, zorder=2)

            # Coronal (AP): faint line only when slice is within probe's AP extent
            if _probe_in_range(xyz, axis=1, val=ap_val):
                self.ax_cor.plot(xyz[:, 0], xyz[:, 2], **proj_kw)
            pt = _line_slice_intersect(xyz, axis=1, val=ap_val)
            if pt is not None:
                self.ax_cor.plot(pt[0], pt[2], 'o', color=color,
                                 markersize=INTERSECT_MARKER_SIZE, zorder=4)
                self.ax_cor.annotate(ltr, xy=(pt[0], pt[2]), **ann_kw)

            # Sagittal (ML): faint line only when slice is within probe's ML extent
            if _probe_in_range(xyz, axis=0, val=ml_val):
                self.ax_sag.plot(xyz[:, 1], xyz[:, 2], **proj_kw)
            pt = _line_slice_intersect(xyz, axis=0, val=ml_val)
            if pt is not None:
                self.ax_sag.plot(pt[1], pt[2], 'o', color=color,
                                 markersize=INTERSECT_MARKER_SIZE, zorder=4)
                self.ax_sag.annotate(ltr, xy=(pt[1], pt[2]), **ann_kw)

            # Horizontal (DV): faint line only when slice is within probe's DV extent
            if _probe_in_range(xyz, axis=2, val=dv_val):
                self.ax_hor.plot(xyz[:, 0], xyz[:, 1], **proj_kw)
            pt = _line_slice_intersect(xyz, axis=2, val=dv_val)
            if pt is not None:
                self.ax_hor.plot(pt[0], pt[1], 'o', color=color,
                                 markersize=INTERSECT_MARKER_SIZE, zorder=4)
                self.ax_hor.annotate(ltr, xy=(pt[0], pt[1]), **ann_kw)

        self.fig.tight_layout()

        # Re-apply any stored zoom (overrides tight_layout autoscale)
        for key, ax in (('cor', self.ax_cor),
                         ('sag', self.ax_sag),
                         ('hor', self.ax_hor)):
            z = self._zoom.get(key)
            if z is not None:
                ax.set_xlim(z[0])
                ax.set_ylim(z[1])

        self.canvas.draw()

    # ------------------------------------------------------------------
    # Tree
    # ------------------------------------------------------------------
    def _rebuild_tree(self):
        saved: dict[int, bool] = {}
        self._walk_leaves(
            lambda it: saved.update(
                {it.data(0, QtCore.Qt.UserRole):
                 it.checkState(0) == QtCore.Qt.Checked}
            )
        )

        self.probe_tree.blockSignals(True)
        self.probe_tree.clear()

        groups: dict[str, dict[str, list]] = {}
        for idx, entry in enumerate(self._data):
            sl = entry['session_label']
            pr = _probe_root(entry['probe_name'])
            groups.setdefault(sl, {}).setdefault(pr, []).append((idx, entry))

        for sess_label, probes in sorted(groups.items()):
            sess_it = QtWidgets.QTreeWidgetItem(self.probe_tree)
            sess_it.setText(0, sess_label)
            sess_it.setFlags(
                QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
            sess_it.setCheckState(0, QtCore.Qt.Checked)
            sess_it.setData(0, QtCore.Qt.UserRole, None)

            sess_first_color = None
            for pr_name, shanks in sorted(probes.items()):
                pr_it = QtWidgets.QTreeWidgetItem(sess_it)
                pr_it.setText(0, pr_name)
                pr_it.setFlags(
                    QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
                pr_it.setCheckState(0, QtCore.Qt.Checked)
                pr_it.setData(0, QtCore.Qt.UserRole, None)

                first_color = None
                for idx, entry in sorted(shanks, key=lambda x: x[1]['probe_name']):
                    ltr = entry['probe_name'][-1].upper() \
                          if entry['probe_name'] else '?'
                    sh_it = QtWidgets.QTreeWidgetItem(pr_it)
                    sh_it.setText(0, f"Shank {ltr}")
                    sh_it.setFlags(
                        QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
                    sh_it.setCheckState(
                        0, QtCore.Qt.Checked if saved.get(idx, True)
                        else QtCore.Qt.Unchecked)
                    sh_it.setData(0, QtCore.Qt.UserRole, idx)
                    color = QtGui.QColor(entry['color'])
                    sh_it.setBackground(1, color)
                    tint = QtGui.QColor(entry['color'])
                    tint.setAlpha(40)
                    sh_it.setBackground(0, tint)
                    if first_color is None:
                        first_color = color
                    if sess_first_color is None:
                        sess_first_color = color

                if first_color is not None:
                    pr_it.setBackground(1, first_color)
                self._refresh_parent(pr_it)

            if sess_first_color is not None:
                sess_it.setBackground(1, sess_first_color)
            self._refresh_parent(sess_it)

        self.probe_tree.collapseAll()
        self.probe_tree.blockSignals(False)

    def _walk_leaves(self, fn):
        def _w(item):
            if item.childCount() == 0:
                fn(item)
            else:
                for i in range(item.childCount()):
                    _w(item.child(i))
        inv = self.probe_tree.invisibleRootItem()
        for i in range(inv.childCount()):
            _w(inv.child(i))

    def _on_tree_changed(self, item, column):
        if column != 0:
            return
        self.probe_tree.blockSignals(True)
        state = item.checkState(0)
        depth = 0
        p = item.parent()
        while p:
            depth += 1
            p = p.parent()
        if depth < 2:
            self._cascade(item, state)
        parent = item.parent()
        if parent:
            self._refresh_parent(parent)
            gp = parent.parent()
            if gp:
                self._refresh_parent(gp)
        self.probe_tree.blockSignals(False)
        self._sync_visibility()
        self._redraw()
        if self._urchin_ready:
            self._render_probes()

    def _on_tree_dbl_click(self, item, column):
        if column != 1:
            return
        leaves: list[tuple[int, QtWidgets.QTreeWidgetItem]] = []
        self._gather_leaves(item, leaves)
        if not leaves:
            return
        init = QtGui.QColor(self._data[leaves[0][0]]['color'])
        color = QtWidgets.QColorDialog.getColor(init, self, "Choose color")
        if not color.isValid():
            return
        hex_c = color.name()
        tint = QtGui.QColor(color)
        tint.setAlpha(40)
        self.probe_tree.blockSignals(True)
        for idx, leaf in leaves:
            self._data[idx]['color'] = hex_c
            leaf.setBackground(1, color)
            leaf.setBackground(0, tint)
            parent = leaf.parent()          # probe row
            if parent and parent.parent() is not None:
                parent.setBackground(1, color)
                grandparent = parent.parent()   # session row
                if grandparent is not None:
                    grandparent.setBackground(1, color)
        self.probe_tree.blockSignals(False)
        self._redraw()
        if self._urchin_ready:
            self._render_probes()

    def _gather_leaves(self, item, result):
        if item.childCount() == 0:
            idx = item.data(0, QtCore.Qt.UserRole)
            if idx is not None:
                result.append((idx, item))
        else:
            for i in range(item.childCount()):
                self._gather_leaves(item.child(i), result)

    def _sync_visibility(self):
        def _w(item):
            if item.childCount() == 0:
                idx = item.data(0, QtCore.Qt.UserRole)
                if idx is not None:
                    self._data[idx]['visible'] = \
                        item.checkState(0) == QtCore.Qt.Checked
            else:
                for i in range(item.childCount()):
                    _w(item.child(i))
        inv = self.probe_tree.invisibleRootItem()
        for i in range(inv.childCount()):
            _w(inv.child(i))

    @staticmethod
    def _cascade(item, state):
        for i in range(item.childCount()):
            c = item.child(i)
            c.setCheckState(0, state)
            UrchinViewerWindow._cascade(c, state)

    @staticmethod
    def _refresh_parent(parent):
        if parent is None:
            return
        n = parent.childCount()
        checked = sum(
            parent.child(i).checkState(0) == QtCore.Qt.Checked
            for i in range(n))
        if checked == 0:
            parent.setCheckState(0, QtCore.Qt.Unchecked)
        elif checked == n:
            parent.setCheckState(0, QtCore.Qt.Checked)
        else:
            parent.setCheckState(0, QtCore.Qt.PartiallyChecked)

    def _set_all_visible(self, visible: bool):
        state = QtCore.Qt.Checked if visible else QtCore.Qt.Unchecked
        self.probe_tree.blockSignals(True)
        inv = self.probe_tree.invisibleRootItem()
        for i in range(inv.childCount()):
            self._cascade(inv.child(i), state)
            inv.child(i).setCheckState(0, state)
        self.probe_tree.blockSignals(False)
        self._sync_visibility()
        self._redraw()
        if self._urchin_ready:
            self._render_probes()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    dlg = LoginDialog()
    if dlg.exec_() != QtWidgets.QDialog.Accepted or dlg.alyx is None:
        sys.exit(0)
    win = UrchinViewerWindow(dlg.alyx)
    win.move(100, 100)
    win.show()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
