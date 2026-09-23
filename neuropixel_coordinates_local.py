"""
neuropixel_coordinates_local.py
================================

Standalone / local build of the IBL "neuropixel_coordinates" (micro-manipulator) GUI.

This is a self-contained fork of ``iblrig.gui.ui_micromanipulator`` that runs in a plain
IBL analysis environment (e.g. the ``iblenv`` conda env) WITHOUT installing ``iblrig``:

  * the two ``iblrig.ephys`` helpers are vendored below (compute + register),
  * ``rodrigues_rotation`` and ``create_insertion`` are vendored (missing from older
    iblatlas / ibllib), and
  * the ``iblrig`` rig-settings / login machinery (``RigWizardModel`` / ``LoginWindow``)
    is replaced with ONE's ``AlyxClient`` plus a small login dialog.

Run it with your analysis-env python, e.g.::

    C:\\Users\\kimil\\anaconda3\\envs\\iblenv\\python.exe C:\\Users\\kimil\\int-brain-lab\\neuropixel_coordinates_local.py

NOTE: set ``DEFAULT_ALYX_URL`` below (or the URL field in the login dialog) to match the
``ALYX_URL`` in your ephys-PC ``iblrig_settings.yaml``.  The first ``Compute`` will download
the NeedlesAtlas volume (a few hundred MB, one-time).

Reference-shank toggle
----------------------
The "Ref. shank" dropdown controls which physical shank the entered reference coordinate
belongs to.  'A' = original/prototype behaviour.  'D' = flipped commercial probe: the
reference position is physically shank d, so the shank letters are reversed while every
shank keeps the exact same coordinates (nothing pivots around the reference).
"""
from __future__ import annotations

import sys
import re
import string
from datetime import date
from pathlib import Path
import traceback
from typing import Any

import numpy as np
import pandas as pd
from qtpy import QtWidgets, QtCore, QtGui
from pydantic import BaseModel, Field, field_validator, ValidationError
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import one.alf.path as alfpath
import spikeglx
import iblatlas.atlas
from iblatlas.atlas import NeedlesAtlas
from one.webclient import AlyxClient
from one.webclient import no_cache as no_cache_context


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Set this to the ALYX_URL from your ephys-PC iblrig_settings.yaml.
DEFAULT_ALYX_URL = "https://alyx.internationalbrainlab.org"


default_trajectory = {'x': -1200.1, 'y': -4131.3, 'z': 901.1, 'phi': 270, 'theta': 15, 'depth': 3300.7, 'roll': 0, 'shanks': 4}

TRAJECTORY_KEYS = ['x', 'y', 'z', 'depth', 'theta', 'phi', 'roll']
PROBE_MODELS = ('NP2.4', 'NP2.4 QB', 'NP2.1', '3B2', '3A')


# ===========================================================================
# Vendored helpers (verbatim from your ephys-PC iblrig/ephys.py, plus the two
# functions it depends on that are missing from older iblatlas / ibllib).
# ===========================================================================
def rodrigues_rotation(v: np.ndarray, k: np.ndarray, theta: float) -> np.ndarray:
    """Rotate vector v around unit axis k by angle theta (radians) — Rodrigues' formula.

    Vendored from iblatlas.atlas (not present in iblatlas 0.5.4).
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    k = k / np.linalg.norm(k)  # ensure k is a unit vector
    return (v * np.cos(theta) +
            np.cross(k, v) * np.sin(theta) +
            k * np.dot(k, v) * (1 - np.cos(theta)))


def neuropixel24_micromanipulator_coordinates(
    ref_shank: dict,
    pname: str,
    ba: iblatlas.atlas.BrainAtlas | None = None,
    shank_spacings_um: tuple[float, ...] = (0, 250, 500, 750),
) -> dict[str, dict]:
    """Calculate micro-manipulator coordinates for all shanks of a Neuropixel 2.4 probe.

    Vendored verbatim from your ephys-PC iblrig/ephys.py (uses 250 um shank pitch and applies
    the roll via Rodrigues rotation). ``iblatlas.atlas.rodrigues_rotation`` is replaced by the
    vendored ``rodrigues_rotation`` above so this runs on older iblatlas.
    """
    shank_order = 'abcd'

    ba = iblatlas.atlas.NeedlesAtlas() if ba is None else ba
    trajectories = {}

    for i, d in enumerate(shank_spacings_um):
        dx = np.sin((ref_shank['phi']) / 180 * np.pi) * d
        dy = -np.cos((ref_shank['phi']) / 180 * np.pi) * d
        # apply the roll transformation
        dx, dy, dz = rodrigues_rotation(
            v=np.array([dx, dy, 0]),  # vector to rotate
            k=np.array(iblatlas.atlas.sph2cart(1, ref_shank['theta'], ref_shank['phi'])),  # rotation axis
            theta=ref_shank['roll'] * np.pi / 180,
        )
        shank = {
            'x': ref_shank['x'] + dx,
            'y': ref_shank['y'] + dy,
            'z': ref_shank['z'] + dz,
            'phi': ref_shank['phi'],
            'theta': ref_shank['theta'],
            'depth': ref_shank['depth'],
            'roll': ref_shank['roll'],
        }
        insertion = iblatlas.atlas.Insertion.from_dict(shank, brain_atlas=ba)
        xyz_entry = iblatlas.atlas.Insertion.get_brain_entry(insertion.trajectory, ba)
        if i == 0:
            xyz_ref = xyz_entry
        shank['z'] = xyz_entry[2] * 1e6
        # right now we keep the original x, y coordinates
        shank['depth'] = ref_shank['depth'] + (xyz_entry[2] - xyz_ref[2]) * 1e6
        trajectories[f'{pname}{shank_order[i]}'] = shank
    return trajectories


def create_insertion(alyx: AlyxClient, md: dict, label: str, eid: str) -> tuple[dict, dict]:
    """Create or update a probe insertion in Alyx and return (description, alyx record).

    Vendored from ibllib.ephys.spikes (not present in ibllib 2.39.1).
    """
    # create json description
    description = {'label': label, 'model': md['neuropixelVersion'], 'serial': int(md['serial']),
                   'raw_file_name': md['fileName']}

    # create or update probe insertion on alyx
    alyx_insertion = {'session': eid, 'model': md['neuropixelVersion'], 'serial': md['serial'], 'name': label}
    pi = alyx.rest('insertions', 'list', session=eid, name=label)
    if len(pi) == 0:
        qc_dict = {'qc': 'NOT_SET', 'extended_qc': {}}
        alyx_insertion.update({'json': qc_dict})
        insertion = alyx.rest('insertions', 'create', data=alyx_insertion)
    else:
        insertion = alyx.rest('insertions', 'partial_update', data=alyx_insertion, id=pi[0]['id'])

    return description, insertion


def register_micromanipulator_coordinates(
    alyx, eid: str, trajectories: dict[str, dict] | None = None, metadata: dict | None = None
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Register micro-manipulator coordinates for probe trajectories in Alyx.

    Vendored verbatim from your ephys-PC iblrig/ephys.py.
    """
    # if we do not have access to the fileName or any of the metadata, it will be patched later
    metadata = {'neuropixelVersion': 'NP2.4', 'fileName': None, 'serial': -1} if metadata is None else metadata
    rest_trajectories = {}
    rest_insertions = {}
    with no_cache_context(alyx):
        traj_extra = {}
        for pname, traj in trajectories.items():
            _, rest_insertions[pname] = create_insertion(alyx, metadata, pname, eid=eid)
            pid = rest_insertions[pname]['id']
            traj_extra['probe_insertion'] = pid
            traj_extra['chronic_insertion'] = None
            traj_extra['provenance'] = 'Micro-manipulator'
            traj_extra['coordinate_system'] = 'Needles-Allen'
            rest_trajectory = alyx.rest('trajectories', 'list', probe_insertion=pid, provenance='Micro-manipulator')
            if len(rest_trajectory) == 0:
                rest_trajectories[pname] = alyx.rest('trajectories', 'create', data=traj | traj_extra)
            else:
                rest_trajectories[pname] = alyx.rest(
                    'trajectories', 'update', id=rest_trajectory[0]['id'], data=traj | traj_extra
                )
    return rest_insertions, rest_trajectories


# ===========================================================================
# GUI
# ===========================================================================
class ProbeInsertion(BaseModel):
    """Pydantic model for validating probe insertion data."""
    pname: str = Field(..., title='Probe Name')
    x: float = Field(..., title='X-ML (um)')
    y: float = Field(..., title='Y-AP (um)')
    z: float = Field(..., title='Z-DV (um)')
    depth: float = Field(..., title='Depth (um)')
    theta: float = Field(..., ge=-90, le=90, title='Theta-Elevation (deg)')
    phi: float = Field(..., ge=-180, le=360, title='Phi-Azimuth (deg)')
    roll: float = Field(..., ge=-180, le=360, title='Roll (deg)')
    shanks: int = Field(..., ge=1, le=4, title='# Shanks')

    @field_validator('pname')
    @classmethod
    def pname_no_special_chars(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9_-]*$', v):
            raise ValueError('must not contain spaces or special characters')
        return v


class LoginDialog(QtWidgets.QDialog):
    """Minimal Alyx login dialog, replacing iblrig.gui.wizard.LoginWindow."""

    def __init__(self, parent=None, url=DEFAULT_ALYX_URL, username=''):
        super().__init__(parent)
        self.setWindowTitle("Alyx Login")
        layout = QtWidgets.QFormLayout(self)
        self.url_edit = QtWidgets.QLineEdit(url)
        self.user_edit = QtWidgets.QLineEdit(username)
        self.pwd_edit = QtWidgets.QLineEdit()
        self.pwd_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.remember_cb = QtWidgets.QCheckBox("Remember me")
        self.remember_cb.setChecked(True)
        layout.addRow("Alyx URL", self.url_edit)
        layout.addRow("Username", self.user_edit)
        layout.addRow("Password", self.pwd_edit)
        layout.addRow("", self.remember_cb)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        # focus on password if we already have a username
        (self.pwd_edit if username else self.user_edit).setFocus()

    def values(self):
        return (self.url_edit.text().strip(), self.user_edit.text().strip(),
                self.pwd_edit.text(), self.remember_cb.isChecked())


class MplCanvas(FigureCanvas):
    """Matplotlib canvas widget to embed in a Qt application."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes1 = self.fig.add_subplot(1, 1, 1)
        self.fig.tight_layout()
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QtCore.QSettings("IBL", "MicroManipulatorGUI")
        self.alyx = None  # lazily created AlyxClient
        self.setWindowTitle("Micro-Manipulator GUI (local)")
        self.setGeometry(150, 150, 1100, 900)
        self.atlas = NeedlesAtlas()
        # Main widget and layout
        main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(main_widget)
        layout = QtWidgets.QVBoxLayout(main_widget)

        # --- Create Form on top ---
        self.line_edits = {}
        form_widget = QtWidgets.QWidget()
        form_layout = QtWidgets.QGridLayout(form_widget)

        self.column_info = {key: field.title for key, field in ProbeInsertion.model_fields.items()}
        # Display/entry order for the input row and table columns. This matches the order the
        # values appear in the user's notes so they can be scanned and tab-entered left to right:
        # Depth is placed last (after Roll), and '# Shanks' stays at the end. Validation is by
        # field name (pydantic), so the order here is free to change without affecting anything.
        self.column_keys = ['pname', 'x', 'y', 'z', 'theta', 'phi', 'roll', 'depth', 'shanks']
        for i, key in enumerate(self.column_keys):
            label_text = self.column_info[key]
            label = QtWidgets.QLabel(label_text)
            line_edit = QtWidgets.QLineEdit()
            default_value = str(default_trajectory.get(key, ''))
            line_edit.setText(self.settings.value(key, default_value))
            line_edit.setPlaceholderText(label_text)
            line_edit.returnPressed.connect(self.compute)  # press Enter in any field to Compute
            self.line_edits[key] = line_edit
            form_layout.addWidget(label, 0, i)
            form_layout.addWidget(line_edit, 1, i)

        # Reference-shank selector: which physical shank the entered coordinate belongs to.
        # 'A' = original/prototype behaviour (reference at shank a, spacing 0).
        # 'D' = flipped commercial probe: the reference position is physically shank d, so the
        #       shank letters are reversed while every shank keeps the exact same coordinates.
        ref_col = len(self.column_keys)
        ref_label = QtWidgets.QLabel("Ref. shank")
        self.ref_shank_combo = QtWidgets.QComboBox()
        self.ref_shank_combo.addItem("A (standard)", "a")
        self.ref_shank_combo.addItem("D (flipped probe)", "d")
        saved_ref = self.settings.value("ref_shank", "a")
        saved_idx = self.ref_shank_combo.findData(saved_ref)
        self.ref_shank_combo.setCurrentIndex(saved_idx if saved_idx >= 0 else 0)
        form_layout.addWidget(ref_label, 0, ref_col)
        form_layout.addWidget(self.ref_shank_combo, 1, ref_col)

        # compute button
        compute_button = QtWidgets.QPushButton("Compute")
        compute_button.clicked.connect(self.compute)
        form_layout.addWidget(compute_button, 1, len(self.column_keys) + 1)

        # clear button
        clear_button = QtWidgets.QPushButton("Clear")
        clear_button.clicked.connect(self.clear_table)
        form_layout.addWidget(clear_button, 1, len(self.column_keys) + 2)

        # --- Create Table ---
        self.table = QtWidgets.QTableWidget(0, len(self.column_keys))  # 0 rows initially
        self.table.setHorizontalHeaderLabels([self.column_info[key] for key in self.column_keys])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        # Create Matplotlib canvas
        self.canvas = MplCanvas(self, width=4, height=4, dpi=100)

        # --- Create Registration Form at the bottom ---
        self.reg_line_edits = {}
        reg_form_widget = QtWidgets.QWidget()
        reg_form_widget.setMaximumWidth(500)
        # NOTE: do not cap the height here. In the horizontal bottom splitter a max-height on the
        # registration form would cap the whole bottom row's height and pin the vertical splitter,
        # making the brain image non-resizable. The grid already keeps the form compact at the top.
        reg_form_layout = QtWidgets.QGridLayout(reg_form_widget)
        reg_form_layout.setRowStretch(0, 1)  # Add stretch to push content down

        # Column 0: Manual mode and labels
        self.manual_mode_checkbox = QtWidgets.QCheckBox("Manual Input")
        self.manual_mode_checkbox.toggled.connect(self.toggle_manual_mode)
        reg_form_layout.addWidget(self.manual_mode_checkbox, 1, 0)

        reg_fields = {"subject": "Subject", "date": "Date", "number": "Number", "serial": "Serial", "version": "Version"}
        for i, (key, label_text) in enumerate(reg_fields.items()):
            label = QtWidgets.QLabel(label_text)
            reg_form_layout.addWidget(label, i + 2, 0)

        # Column 1: Browse button and input widgets
        self.browse_button = QtWidgets.QPushButton("AP File...")
        self.browse_button.clicked.connect(self.browse_file)
        reg_form_layout.addWidget(self.browse_button, 1, 1)

        # Subject, Date, Number LineEdits
        for i, key in enumerate(["subject", "date", "number"]):
            line_edit = QtWidgets.QLineEdit()
            if key == 'date':
                line_edit.setText(date.today().isoformat())
            self.reg_line_edits[key] = line_edit
            reg_form_layout.addWidget(line_edit, i + 2, 1)

        # Serial LineEdit
        self.reg_line_edits['serial'] = QtWidgets.QLineEdit()
        reg_form_layout.addWidget(self.reg_line_edits['serial'], 5, 1)

        # Version ComboBox
        self.probe_model_combo = QtWidgets.QComboBox()
        self.probe_model_combo.addItems(PROBE_MODELS)
        reg_form_layout.addWidget(self.probe_model_combo, 6, 1)

        # Register button below everything
        register_button = QtWidgets.QPushButton("Register")
        register_button.clicked.connect(self.register)
        reg_form_layout.addWidget(register_button, len(reg_fields) + 2, 0, 1, 2)  # Span across columns

        # Add the top input form (keeps its natural height)
        layout.addWidget(form_widget)

        # Bottom area: brain-image canvas next to the registration form, in a horizontal
        # splitter so the image can be widened or narrowed against the registration panel.
        bottom_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bottom_splitter.addWidget(self.canvas)
        bottom_splitter.addWidget(reg_form_widget)
        bottom_splitter.setStretchFactor(0, 1)  # canvas absorbs extra width
        bottom_splitter.setStretchFactor(1, 0)

        # Vertical splitter so the trajectory table and the brain image can be resized against
        # each other: drag the divider up to enlarge the image, or down to see more table rows.
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        main_splitter.addWidget(self.table)
        main_splitter.addWidget(bottom_splitter)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)
        main_splitter.setChildrenCollapsible(False)  # don't let a pane be dragged to zero
        layout.addWidget(main_splitter, 1)  # stretch to fill the window

        self.init_images()
        # remember the full-extent view so a double-click can restore it after zooming
        self._home_xlim = self.canvas.axes1.get_xlim()
        self._home_ylim = self.canvas.axes1.get_ylim()
        # mouse-wheel zoom (centered on the cursor) + double-click to reset the view
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        self.toggle_manual_mode(False)  # Set initial state to non-manual

    def closeEvent(self, event):
        """Save settings when the window is closed."""
        for key, line_edit in self.line_edits.items():
            self.settings.setValue(key, line_edit.text())
        self.settings.setValue("ref_shank", self.ref_shank_combo.currentData())
        super().closeEvent(event)

    def compute(self):
        """
        Triggered by the 'Compute' button.
        Validates input fields using a Pydantic model and, if valid, adds a new row to the table.
        """
        raw_trajectory = {key: line_edit.text().strip() for key, line_edit in self.line_edits.items()}

        try:
            # Validate the data using the Pydantic model
            trajectory = ProbeInsertion(**raw_trajectory)
            trajectory = trajectory.model_dump()

            if int(raw_trajectory['shanks']) == 1:
                _traj = {k: trajectory[k] for k in TRAJECTORY_KEYS}
                shanks_trajectories = {trajectory['pname']: _traj}
            else:
                shanks_trajectories = neuropixel24_micromanipulator_coordinates(
                    trajectory, pname=trajectory['pname'], ba=self.atlas)
                # If the entered reference coordinate corresponds to the last shank instead of
                # the first (flipped commercial probe: the Pinpoint 'a' position is physically
                # the 'd' shank), reverse the shank letters while keeping each shank's geometry
                # exactly in place. The reference position stays first so it remains the pivot
                # marked in black on the plot.
                if self.ref_shank_combo.currentData() == 'd':
                    stem = trajectory['pname']
                    items = list(shanks_trajectories.items())
                    n = len(items)
                    shanks_trajectories = {
                        f'{stem}{string.ascii_lowercase[n - 1 - i]}': shank
                        for i, (_, shank) in enumerate(items)
                    }

            for k in shanks_trajectories.keys():
                shank_data = shanks_trajectories[k]
                shank_data['pname'] = k
                shank_data['shanks'] = 1
                self.add_row_to_table(shank_data)

            self.update_plots()
        except ValidationError as e:
            # Display validation errors to the user
            error_messages = []
            for error in e.errors():
                field_name = error['loc'][0]
                label = self.column_info.get(field_name, field_name)
                error_messages.append(f"Error in '{label}': {error['msg']}")
            error_dialog = QtWidgets.QMessageBox()
            error_dialog.setIcon(QtWidgets.QMessageBox.Warning)
            error_dialog.setText("Invalid input")
            error_dialog.setInformativeText("\n".join(error_messages))
            error_dialog.setWindowTitle("Validation Error")
            error_dialog.exec_()
            print("\n".join(error_messages))

    def add_row_to_table(self, data):
        """Adds a new row to the table with the given data, removing any existing rows with the same probe name."""
        # Get the probe name from the data
        probe_name = data.get('pname', '')

        # Find and remove existing rows with the same probe name
        if probe_name:
            # Get the column index for 'pname'
            pname_col_idx = self.column_keys.index('pname')

            # Iterate through rows in reverse to safely remove items
            for row in range(self.table.rowCount() - 1, -1, -1):
                item = self.table.item(row, pname_col_idx)
                if item and item.text() == probe_name:
                    self.table.removeRow(row)

        # Add the new row
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)
        for i, key in enumerate(self.column_keys):
            item = QtWidgets.QTableWidgetItem(str(data.get(key, '')))
            self.table.setItem(row_position, i, item)

    def update_plots(self):
        self.clear_plots()
        df = pd.DataFrame(self.read_table())
        df['shank'] = df['pname'].apply(lambda x: x[-1])
        df['pname'] = df['pname'].apply(lambda x: x[:-1])
        for pname, shanks_trajectories in df.groupby('pname'):
            # we compute the text labels coordinates so they are legible on the overall plot
            x = shanks_trajectories['x'].values
            y = shanks_trajectories['y'].values

            # this is the angle of the labels from the x-axis positive direction, mathematical direction
            angle = np.arctan((y[-1] - y[0]) / (x[-1] - x[0])) - np.pi / 2
            # we dilate the labels by 2.5 and move them orthogonal to the shank alignment
            xlabels = (x - np.mean(x)) * 2.5 + 400 * np.cos(angle) + np.mean(x)
            ylabels = (y - np.mean(y)) * 2.5 + 400 * np.sin(angle) + np.mean(y)
            # the pivot shank is shown in black
            line = self.canvas.axes1.plot(x, y, 'x', label=pname)[0]
            line_color = line.get_color()
            if x.size > 1:
                self.canvas.axes1.plot(x[0], y[0], 'xk')
            i = 0
            for _, rec in shanks_trajectories.iterrows():
                # set the pivot shank in black bold if multishank
                if (i == 0) and (x.size > 1):
                    self.canvas.axes1.text(xlabels[i], ylabels[i], rec.shank, color='k', fontweight=1000)
                else:
                    self.canvas.axes1.text(xlabels[i], ylabels[i], rec.shank, color=line_color, fontweight=800)
                i += 1
            self.canvas.axes1.legend()
            self.canvas.draw()

    def clear_plots(self):
        # Clear the lines and labels on the plot
        for ax in [self.canvas.axes1]:
            [h.remove() for h in ax.lines]
            [h.remove() for h in ax.texts]
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        self.canvas.draw()

    def clear_table(self):
        """Clears all rows from the table and resets the plot."""
        self.table.setRowCount(0)
        self.clear_plots()

    def init_images(self):
        # Plot images
        self.atlas.compute_surface()
        self.atlas.plot_top(volume='image', ax=self.canvas.axes1)
        self.canvas.axes1.set_axis_off()
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def on_scroll(self, event):
        """Zoom the brain image in/out on mouse-wheel, centered on the cursor position."""
        ax = self.canvas.axes1
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        base_scale = 1.2
        if event.button == 'up':        # wheel up -> zoom in
            scale = 1.0 / base_scale
        elif event.button == 'down':    # wheel down -> zoom out
            scale = base_scale
        else:
            return
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        # rescale each axis about the cursor, preserving axis direction (image y is inverted)
        new_w = (xlim[1] - xlim[0]) * scale
        new_h = (ylim[1] - ylim[0]) * scale
        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])
        ax.set_xlim([xdata - new_w * (1 - relx), xdata + new_w * relx])
        ax.set_ylim([ydata - new_h * (1 - rely), ydata + new_h * rely])
        self.canvas.draw_idle()

    def on_click(self, event):
        """Double-click restores the full-extent brain-image view."""
        if getattr(event, 'dblclick', False) and event.inaxes is self.canvas.axes1:
            self.canvas.axes1.set_xlim(self._home_xlim)
            self.canvas.axes1.set_ylim(self._home_ylim)
            self.canvas.draw_idle()

    def toggle_manual_mode(self, checked):
        """Enable or disable manual input fields."""
        self.browse_button.setEnabled(not checked)
        for key, widget in self.reg_line_edits.items():
            widget.setReadOnly(not checked)
            widget.setStyleSheet("background-color: lightgray;" if not checked else "")
        self.probe_model_combo.setEnabled(checked)

    def browse_file(self):
        """Opens a file dialog to select a file and populates fields from its path."""
        options = QtWidgets.QFileDialog.Options()
        start_path = self.settings.value("subjects_path", str(Path.home()))
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select AP Binary File", start_path, "AP Binary Files (*.ap.*bin);;All Files (*)", options=options)
        if fileName:
            print(f"File selected: {fileName}")
            binfile = Path(fileName)
            self.settings.setValue("subjects_path", str(binfile.parent))
            session_path = alfpath.get_session_path(binfile)
            try:
                # Assumes path structure .../subject/date/number/...
                self.reg_line_edits['subject'].setText(session_path.parts[-3])
                self.reg_line_edits['date'].setText(session_path.parts[-2])
                self.reg_line_edits['number'].setText(session_path.parts[-1])
                sr = spikeglx.Reader(binfile)
                self.probe_model_combo.setCurrentText(sr.meta['neuropixelVersion'])
                self.reg_line_edits['serial'].setText(str(sr.meta['serial']))
            except (IndexError, AttributeError):
                print("Could not parse subject/date/number from path. Please check the directory structure.")

    def read_table(self) -> list[dict[str, Any]]:
        """Read and validate all probe insertion trajectories from the table widget."""
        trajectories = []
        for row in range(self.table.rowCount()):
            row_data = {}
            for col_idx, key in enumerate(self.column_keys):
                item = self.table.item(row, col_idx)
                if item:
                    row_data[key] = item.text()
            # Validate and format each row using the Pydantic model
            trajectory = ProbeInsertion(**row_data)
            trajectories.append(trajectory.model_dump())
        return trajectories

    def _get_alyx(self) -> AlyxClient:
        """Return a logged-in AlyxClient, prompting for credentials if needed.

        Replaces iblrig.gui.wizard.RigWizardModel's managed alyx client / LoginWindow.
        """
        if self.alyx is not None and self.alyx.is_logged_in:
            return self.alyx

        url = self.settings.value("alyx_url", DEFAULT_ALYX_URL)
        user = self.settings.value("alyx_user", "")

        # 1) try to reuse a cached ONE token silently
        alyx = None
        try:
            alyx = AlyxClient(base_url=url, silent=True)
        except Exception:
            alyx = None

        # 2) otherwise prompt for credentials
        if alyx is None or not alyx.is_logged_in:
            dlg = LoginDialog(self, url=url, username=user)
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                raise ConnectionError("Alyx login cancelled.")
            url, user, pwd, remember = dlg.values()
            alyx = AlyxClient(base_url=url, silent=True)
            alyx.authenticate(username=user, password=pwd, cache_token=remember, force=True)
            if not alyx.is_logged_in:
                raise ConnectionError("Alyx authentication failed - check URL, username and password.")
            self.settings.setValue("alyx_url", url)
            self.settings.setValue("alyx_user", user)

        self.alyx = alyx
        return alyx

    def register(self):
        """Register the computed shank trajectories to Alyx."""
        subject = self.reg_line_edits['subject'].text()
        date_str = self.reg_line_edits['date'].text()
        number = self.reg_line_edits['number'].text()
        serial = self.reg_line_edits['serial'].text()
        version = self.probe_model_combo.currentText()
        trajectories = {t['pname']: {k: t[k] for k in TRAJECTORY_KEYS} for t in self.read_table()}
        print(f"Registering: Subject={subject}, Date={date_str}, Number={number}, "
              f"Serial={serial}, Version={version}")

        try:
            assert subject != '', "Subject cannot be empty"
            assert number != '', "Number cannot be empty"
            assert date_str != '', "Date cannot be empty"
            assert len(trajectories) > 0, "No shanks to register - click Compute first"

            alyx = self._get_alyx()
            alyx.rest('subjects', 'list', nickname=subject, no_cache=True)
            rest_session = alyx.rest('sessions', 'list', subject=subject,
                                     date_range=[date_str, date_str], number=number)
            if len(rest_session) == 1:
                eid = rest_session[0]['id']
            elif len(rest_session) == 0:
                raise ValueError(f"No session found for subject={subject}, date={date_str}, number={number}")
            else:
                raise ValueError(f"Multiple sessions found for subject={subject}, date={date_str}, number={number}")

            # metadata=None reproduces the ephys-PC behaviour (model NP2.4, serial -1); the
            # per-shank coordinates are what matter for planning. Wire the serial/version fields
            # into a metadata dict here if you want them written to the insertion.
            register_micromanipulator_coordinates(alyx, eid=eid, trajectories=trajectories, metadata=None)

            ok = QtWidgets.QMessageBox()
            ok.setIcon(QtWidgets.QMessageBox.Information)
            ok.setText("Registration successful")
            ok.setInformativeText(f"Registered {len(trajectories)} shank(s) to session "
                                  f"{subject}/{date_str}/{number}.")
            ok.setWindowTitle("Success")
            ok.exec_()
        except Exception as e:
            full_error_message = traceback.format_exc()
            error_message = str(e)
            error_dialog = QtWidgets.QMessageBox()
            error_dialog.setIcon(QtWidgets.QMessageBox.Warning)
            error_dialog.setText("Registration error")
            error_dialog.setInformativeText(error_message)
            error_dialog.setWindowTitle("Error")
            error_dialog.exec_()
            print(full_error_message)


def main():
    app = QtWidgets.QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
