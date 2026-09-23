"""
trajectory_viewer_local.py
===========================
Standalone GUI for viewing micro-manipulator electrode trajectories on Allen-atlas brain slices.

Run
---
    C:\\Users\\kimil\\anaconda3\\envs\\iblenv\\python.exe trajectory_viewer_local.py

Requires iblenv conda env (iblatlas, one.api, qtpy, matplotlib).
"""
from __future__ import annotations

import sys

import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

from iblatlas.atlas import NeedlesAtlas, Insertion
from one.webclient import AlyxClient


DEFAULT_ALYX_URL = "https://alyx.internationalbrainlab.org"

SHANK_COLORS = {
    'a': '#e74c3c',
    'b': '#3498db',
    'c': '#2ecc71',
    'd': '#f39c12',
}
PROBE_COLORS = ['#9b59b6', '#1abc9c', '#e67e22', '#2c3e50']

LINE_WIDTH = 3.0
ENTRY_MARKER_SIZE = 8
TIP_MARKER_SIZE = 7


def _probe_root(probe_name: str) -> str:
    """Strip shank-letter suffix: 'probe01a' -> 'probe01'."""
    if probe_name and probe_name[-1].lower() in SHANK_COLORS:
        return probe_name[:-1]
    return probe_name


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
# Main viewer window
# ---------------------------------------------------------------------------
class TrajectoryViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, alyx: AlyxClient):
        super().__init__()
        self.alyx = alyx
        self.setWindowTitle("Micro-Manipulator Trajectory Viewer")
        self.resize(1260, 800)

        self.statusBar().showMessage("Loading NeedlesAtlas (first run may download ~500 MB)...")
        QtWidgets.QApplication.processEvents()
        self.ba = NeedlesAtlas()
        self.statusBar().showMessage("Atlas loaded.")

        self._sessions_map: dict[str, str] = {}
        # Each entry: {session_label, subject, probe_name, color, traj}
        self._probe_items: list[dict] = []

        self._build_ui()
        self._populate_subjects()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(splitter)

        sidebar = QtWidgets.QWidget()
        sidebar.setMinimumWidth(200)
        vlay = QtWidgets.QVBoxLayout(sidebar)
        vlay.setSpacing(4)

        vlay.addWidget(self._header("Subject"))
        self.subj_combo = QtWidgets.QComboBox()
        self.subj_combo.currentIndexChanged.connect(self._on_subject_changed)
        vlay.addWidget(self.subj_combo)

        vlay.addSpacing(4)
        vlay.addWidget(self._header("Session"))
        self.sess_combo = QtWidgets.QComboBox()
        vlay.addWidget(self.sess_combo)

        vlay.addSpacing(4)
        br = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add Session")
        self.add_btn.setMinimumHeight(28)
        self.add_btn.clicked.connect(self._add_session)
        self.clear_btn = QtWidgets.QPushButton("Clear All")
        self.clear_btn.setMinimumHeight(28)
        self.clear_btn.clicked.connect(self._clear_sessions)
        br.addWidget(self.add_btn)
        br.addWidget(self.clear_btn)
        vlay.addLayout(br)

        vlay.addSpacing(8)
        sep1 = QtWidgets.QFrame()
        sep1.setFrameShape(QtWidgets.QFrame.HLine)
        sep1.setFrameShadow(QtWidgets.QFrame.Sunken)
        vlay.addWidget(sep1)
        vlay.addWidget(self._header("Slice positions"))

        vlay.addWidget(QtWidgets.QLabel("Coronal AP (µm):"))
        self.ap_spin = self._make_spin(-8000, 5000, 100, " µm")
        self.ap_spin.valueChanged.connect(self._redraw)
        vlay.addWidget(self.ap_spin)

        vlay.addWidget(QtWidgets.QLabel("Sagittal ML (µm):"))
        self.ml_spin = self._make_spin(-5500, 5500, 100, " µm")
        self.ml_spin.valueChanged.connect(self._redraw)
        vlay.addWidget(self.ml_spin)

        vlay.addWidget(QtWidgets.QLabel("Horizontal DV (µm):"))
        self.dv_spin = self._make_spin(-7000, 0, 100, " µm")
        self.dv_spin.valueChanged.connect(self._redraw)
        vlay.addWidget(self.dv_spin)

        vlay.addSpacing(8)
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        sep2.setFrameShadow(QtWidgets.QFrame.Sunken)
        vlay.addWidget(sep2)

        hdr_row = QtWidgets.QHBoxLayout()
        hdr_row.addWidget(self._header("Probes"))
        tip = QtWidgets.QLabel("<small><i>dbl-click color to change</i></small>")
        hdr_row.addWidget(tip)
        vlay.addLayout(hdr_row)

        self.probe_tree = QtWidgets.QTreeWidget()
        self.probe_tree.setColumnCount(2)
        self.probe_tree.setHeaderLabels(["Session / Probe / Shank", ""])
        self.probe_tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.probe_tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        self.probe_tree.header().resizeSection(1, 36)
        self.probe_tree.setMinimumHeight(80)
        self.probe_tree.itemChanged.connect(self._on_tree_changed)
        self.probe_tree.itemDoubleClicked.connect(self._on_tree_dbl_click)
        vlay.addWidget(self.probe_tree)

        vr = QtWidgets.QHBoxLayout()
        ca = QtWidgets.QPushButton("Check All")
        ca.clicked.connect(lambda: self._set_all_visible(True))
        ua = QtWidgets.QPushButton("Uncheck All")
        ua.clicked.connect(lambda: self._set_all_visible(False))
        vr.addWidget(ca)
        vr.addWidget(ua)
        vlay.addLayout(vr)

        vlay.addStretch()
        splitter.addWidget(sidebar)

        self.fig = Figure(figsize=(9.5, 7.5))
        self.canvas = FigureCanvas(self.fig)
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        self.ax_cor = self.fig.add_subplot(gs[0, 0])
        self.ax_sag = self.fig.add_subplot(gs[0, 1])
        self.ax_hor = self.fig.add_subplot(gs[1, 0])
        self.ax_leg = self.fig.add_subplot(gs[1, 1])
        self.ax_leg.axis('off')
        self._label_axes()
        self.canvas.draw()
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([290, 970])

    @staticmethod
    def _header(text):
        return QtWidgets.QLabel(f"<b>{text}</b>")

    @staticmethod
    def _make_spin(lo, hi, step, suffix=""):
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

    def _shank_color(self, probe_name, fallback_idx):
        if probe_name and probe_name[-1].lower() in SHANK_COLORS:
            return SHANK_COLORS[probe_name[-1].lower()]
        return PROBE_COLORS[fallback_idx % len(PROBE_COLORS)]

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
            self.sess_combo.addItems(sorted(sessions.keys(), reverse=True))
            self.statusBar().showMessage(
                f"Found {len(sessions)} session(s) for {subject}.")
        except Exception as exc:
            self.statusBar().showMessage(f"Error: {exc}")

    def _add_session(self):
        sess_label = self.sess_combo.currentText()
        if not sess_label:
            return
        sess_id = self._sessions_map.get(sess_label)
        if not sess_id:
            return
        if sess_label in {e['session_label'] for e in self._probe_items}:
            self.statusBar().showMessage(f"{sess_label} is already loaded.")
            return

        subject = self.subj_combo.currentText()
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
        first_load = not self._probe_items

        for i, traj in enumerate(trajs_sorted):
            probe_name = traj.get('probe_name', f'probe{i}')
            self._probe_items.append({
                'session_label': sess_label,
                'subject': subject,
                'probe_name': probe_name,
                'color': self._shank_color(probe_name, i),
                'traj': traj,
            })

        if first_load:
            ins_list = []
            for t in trajs_sorted:
                try:
                    ins_list.append(Insertion.from_dict(t, brain_atlas=self.ba))
                except Exception:
                    pass
            if ins_list:
                xyz = np.array([ins.xyz * 1e6 for ins in ins_list])
                for sp in (self.ap_spin, self.ml_spin, self.dv_spin):
                    sp.blockSignals(True)
                self.ap_spin.setValue(round(float(np.mean(xyz[:, :, 1])), 0))
                self.ml_spin.setValue(round(float(np.mean(xyz[:, :, 0])), 0))
                self.dv_spin.setValue(round(float(np.mean(xyz[:, :, 2])), 0))
                for sp in (self.ap_spin, self.ml_spin, self.dv_spin):
                    sp.blockSignals(False)

        self._rebuild_tree()
        self._redraw()
        self.statusBar().showMessage(
            f"Added {len(trajs_sorted)} probe(s) from {sess_label}. "
            f"{len(self._probe_items)} total.")

    def _clear_sessions(self):
        self._probe_items.clear()
        self.probe_tree.blockSignals(True)
        self.probe_tree.clear()
        self.probe_tree.blockSignals(False)
        for ax in (self.ax_cor, self.ax_sag, self.ax_hor, self.ax_leg):
            ax.cla()
        self.ax_leg.axis('off')
        self._label_axes()
        self.canvas.draw()
        self.statusBar().showMessage("Cleared.")

    # ------------------------------------------------------------------
    # Tree
    # ------------------------------------------------------------------
    def _rebuild_tree(self):
        # Preserve check states by flat-list index
        saved: dict[int, bool] = {}
        self._walk_leaves(
            lambda it: saved.update(
                {it.data(0, QtCore.Qt.UserRole):
                 it.checkState(0) == QtCore.Qt.Checked}
            )
        )

        self.probe_tree.blockSignals(True)
        self.probe_tree.clear()

        # Group: session_label -> probe_root -> [(idx, entry)]
        groups: dict[str, dict[str, list]] = {}
        for idx, entry in enumerate(self._probe_items):
            sl = entry['session_label']
            pr = _probe_root(entry['probe_name'])
            groups.setdefault(sl, {}).setdefault(pr, []).append((idx, entry))

        for sess_label, probes in groups.items():
            sess_it = QtWidgets.QTreeWidgetItem(self.probe_tree)
            sess_it.setText(0, sess_label)
            sess_it.setFlags(
                QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
            sess_it.setCheckState(0, QtCore.Qt.Checked)
            sess_it.setData(0, QtCore.Qt.UserRole, None)

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

                # Probe root gets a swatch showing the first shank color
                if first_color is not None:
                    pr_it.setBackground(1, first_color)
                self._refresh_parent(pr_it)
            self._refresh_parent(sess_it)

        self.probe_tree.expandAll()
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
        self._redraw()

    def _on_tree_dbl_click(self, item, column):
        if column != 1:
            return
        leaves: list[tuple[int, QtWidgets.QTreeWidgetItem]] = []
        self._gather_leaves(item, leaves)
        if not leaves:
            return
        init = QtGui.QColor(self._probe_items[leaves[0][0]]['color'])
        color = QtWidgets.QColorDialog.getColor(init, self, "Choose color")
        if not color.isValid():
            return
        hex_c = color.name()
        tint = QtGui.QColor(color)
        tint.setAlpha(40)
        self.probe_tree.blockSignals(True)
        for idx, leaf in leaves:
            self._probe_items[idx]['color'] = hex_c
            leaf.setBackground(1, color)
            leaf.setBackground(0, tint)
            # Keep probe-root swatch in sync
            parent = leaf.parent()
            if parent and parent.parent() is not None:
                parent.setBackground(1, color)
        self.probe_tree.blockSignals(False)
        self._redraw()

    def _gather_leaves(self, item, result):
        if item.childCount() == 0:
            idx = item.data(0, QtCore.Qt.UserRole)
            if idx is not None:
                result.append((idx, item))
        else:
            for i in range(item.childCount()):
                self._gather_leaves(item.child(i), result)

    def _visible_indices(self) -> set:
        vis: set[int] = set()
        def _w(item):
            if item.childCount() == 0:
                idx = item.data(0, QtCore.Qt.UserRole)
                if idx is not None and item.checkState(0) == QtCore.Qt.Checked:
                    vis.add(idx)
            else:
                for i in range(item.childCount()):
                    _w(item.child(i))
        inv = self.probe_tree.invisibleRootItem()
        for i in range(inv.childCount()):
            _w(inv.child(i))
        return vis

    @staticmethod
    def _cascade(item, state):
        for i in range(item.childCount()):
            c = item.child(i)
            c.setCheckState(0, state)
            TrajectoryViewerWindow._cascade(c, state)

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
        self._redraw()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _redraw(self):
        if not self._probe_items:
            return

        vis = self._visible_indices()
        ap_m = self.ap_spin.value() / 1e6
        ml_m = self.ml_spin.value() / 1e6
        dv_m = self.dv_spin.value() / 1e6

        for ax in (self.ax_cor, self.ax_sag, self.ax_hor, self.ax_leg):
            ax.cla()
        self.ax_leg.axis('off')

        self.ba.plot_cslice(ap_m, ax=self.ax_cor)
        self.ba.plot_sslice(ml_m, ax=self.ax_sag)
        self.ba.plot_hslice(dv_m, ax=self.ax_hor)

        self._label_axes()
        self.ax_cor.set_title(
            f"Coronal  AP = {self.ap_spin.value():.0f} µm", fontsize=10)
        self.ax_sag.set_title(
            f"Sagittal  ML = {self.ml_spin.value():.0f} µm", fontsize=10)
        self.ax_hor.set_title(
            f"Horizontal  DV = {self.dv_spin.value():.0f} µm", fontsize=10)

        multi = len(set(e['session_label'] for e in self._probe_items)) > 1
        patches = []

        for idx, entry in enumerate(self._probe_items):
            if idx not in vis:
                continue
            try:
                ins = Insertion.from_dict(entry['traj'], brain_atlas=self.ba)
            except Exception:
                continue

            color = entry['color']
            xyz = ins.xyz * 1e6  # (2, 3): [entry, tip] x [ML, AP, DV]
            ltr = entry['probe_name'][-1].upper() if entry['probe_name'] else ''
            kw = dict(color=color, linewidth=LINE_WIDTH,
                      solid_capstyle='round', zorder=3)
            ann_kw = dict(xytext=(3, 3), textcoords='offset points',
                          fontsize=7, color=color, fontweight='bold', zorder=5)

            # Coronal: x=ML, y=DV
            self.ax_cor.plot(xyz[:, 0], xyz[:, 2], **kw)
            self.ax_cor.plot(xyz[0, 0], xyz[0, 2], 'o', color=color,
                             markersize=ENTRY_MARKER_SIZE, zorder=4)
            self.ax_cor.plot(xyz[1, 0], xyz[1, 2], 'v', color=color,
                             markersize=TIP_MARKER_SIZE, zorder=4)
            self.ax_cor.annotate(ltr, xy=(xyz[0, 0], xyz[0, 2]), **ann_kw)

            # Sagittal: x=AP, y=DV
            self.ax_sag.plot(xyz[:, 1], xyz[:, 2], **kw)
            self.ax_sag.plot(xyz[0, 1], xyz[0, 2], 'o', color=color,
                             markersize=ENTRY_MARKER_SIZE, zorder=4)
            self.ax_sag.plot(xyz[1, 1], xyz[1, 2], 'v', color=color,
                             markersize=TIP_MARKER_SIZE, zorder=4)
            self.ax_sag.annotate(ltr, xy=(xyz[0, 1], xyz[0, 2]), **ann_kw)

            # Horizontal: x=ML, y=AP
            self.ax_hor.plot(xyz[:, 0], xyz[:, 1], **kw)
            self.ax_hor.plot(xyz[0, 0], xyz[0, 1], 'o', color=color,
                             markersize=ENTRY_MARKER_SIZE, zorder=4)
            self.ax_hor.plot(xyz[1, 0], xyz[1, 1], 'v', color=color,
                             markersize=TIP_MARKER_SIZE, zorder=4)
            self.ax_hor.annotate(ltr, xy=(xyz[0, 0], xyz[0, 1]), **ann_kw)

            label = (f"{entry['session_label']} {entry['probe_name']}"
                     if multi else entry['probe_name'])
            patches.append(mpatches.Patch(color=color, label=label))

        if patches:
            n_sess = len(set(e['session_label'] for e in self._probe_items))
            if multi:
                title = f"{n_sess} sessions"
            else:
                e0 = self._probe_items[0]
                title = f"{e0['subject']}\n{e0['session_label']}"
            self.ax_leg.set_title(title, fontsize=9, pad=4)
            self.ax_leg.legend(handles=patches, loc='center', fontsize=9,
                               title='Probe / Shank', title_fontsize=9,
                               frameon=True)
            self.ax_leg.text(0.5, 0.08, "● entry     ▼ tip",
                             transform=self.ax_leg.transAxes,
                             ha='center', va='center',
                             fontsize=8, color='#555555')

        self.fig.tight_layout()
        self.canvas.draw()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    dlg = LoginDialog()
    if dlg.exec_() != QtWidgets.QDialog.Accepted or dlg.alyx is None:
        sys.exit(0)
    win = TrajectoryViewerWindow(dlg.alyx)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
