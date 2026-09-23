# ibl-probe-tools

Standalone GUI tools for visualising NP2.4 probe trajectories registered in IBL Alyx.

## Tools

| Script | Description |
|--------|-------------|
| `trajectory_viewer_urchin.py` | Urchin 3D brain viewer + coronal/sagittal/horizontal slice panels |
| `trajectory_viewer_local.py` | Coronal/sagittal/horizontal slice panels only (no Urchin) |
| `neuropixel_coordinates_local.py` | Micro-manipulator coordinates GUI |

## Prerequisites

- IBL `iblenv` conda environment — see the [IBL ephys installation guide](https://int-brain-lab.github.io/iblenv/install_doc.html)
- `oursin` 0.7.2 (required for the Urchin viewer):
  ```
  conda activate iblenv
  pip install oursin==0.7.2
  ```
- After installing oursin, patch line 40 of `…/iblenv/Lib/site-packages/oursin/renderer.py`:
  ```python
  client.sio.connect('https://pinpoint.allenneuraldynamics-test.org:5000')
  ```

## Launching

### Windows

Edit the `.bat` launcher for the tool you want to run and replace `kimil` with your Windows username, then double-click the `.bat` file.

### macOS

**One-time setup** — make the launchers executable after cloning:

```bash
chmod +x trajectory_viewer_urchin.command trajectory_viewer_local.command neuropixel_coordinates_local.command
```

Then double-click the `.command` file, or run it from a terminal.

The `.command` files assume Anaconda is installed at `/opt/anaconda3`. If yours is elsewhere (e.g. `~/anaconda3`), edit the path in the `.command` file accordingly.

## Full usage instructions

See the [Urchin Trajectory Viewer user guide](https://claude.ai/artifact/7iGUfe51u3L8fXF1EoBEYW) for a step-by-step walkthrough of the GUI.
