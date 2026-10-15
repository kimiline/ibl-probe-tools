# ibl-probe-tools

Standalone GUI tools for visualising NP2.4 probe trajectories registered in IBL Alyx.

## Tools

| Script | Description |
|--------|-------------|
| `trajectory_viewer_urchin.py` | Urchin 3D brain viewer + coronal/sagittal/horizontal slice panels |
| `trajectory_viewer_local.py` | Coronal/sagittal/horizontal slice panels only (no Urchin) |
| `neuropixel_coordinates_local.py` | Micro-manipulator coordinates GUI |
| `register_tracks_alyx.py` | Review Lasagna-traced tracks (`*_pts.csv`) and register the ones you tick to Alyx |

## Prerequisites

- IBL `iblenv` conda environment: see the [IBL ephys installation guide](https://int-brain-lab.github.io/iblenv/install_doc.html)
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

**One-time setup:** make the launchers executable after cloning:

```bash
chmod +x trajectory_viewer_urchin.command trajectory_viewer_local.command neuropixel_coordinates_local.command register_tracks_alyx.command
```

Then double-click the `.command` file, or run it from a terminal.

The `.command` files assume Anaconda is installed at `/opt/anaconda3`. If yours is elsewhere (e.g. `~/anaconda3`), edit the path in the `.command` file accordingly.

## Lasagna (lab fork)

We use a modified [Lasagna](https://github.com/kimiline/lasagna) for viewing histology. It opens CCF-registered stacks (e.g. `STD_ds_*_GR.tif` / `STD_ds_*_RD.tif`) in the same orientation as these tools, and adds multi-file open with channel assignment and keyboard shortcuts.

**Install once**, in its own conda environment (not `iblenv`):

```
conda create -n lasagna python=3.11
conda activate lasagna
pip install git+https://github.com/kimiline/lasagna.git
```

**Launch:**

```
conda activate lasagna
lasagna
```

**Update** to the latest version:

```
conda activate lasagna
pip install --upgrade --force-reinstall --no-deps git+https://github.com/kimiline/lasagna.git
```

`--no-deps` reinstalls only Lasagna itself, which is quick and leaves the other packages alone. If an update ever adds a new dependency, run the same command without `--no-deps`.

## Registering Lasagna tracks to Alyx

`register_tracks_alyx.py` replaces ibllib's `register_lasagna_tracks_alyx.py` example. It uploads tracks to the **shared IBL Alyx database**, so it checks every file before anything is written.

**Environment.** It needs ONE-api 3.4 or newer, because Alyx stopped serving the API description that older ONE versions use (every `rest()` call fails with `406 Not Acceptable`). `iblenv` runs Python 3.9 and cannot take ONE 3.x, so run the tool in `iblenv3.11` (the alignment GUI environment) after upgrading ONE there:

```
conda activate iblenv3.11
pip install "ONE-api>=3.4"
```

**Launch** it from the **Register tracks to Alyx...** button under the Name list on Lasagna's Points tab. The first time, Lasagna asks for the `iblenv3.11` python and for `register_tracks_alyx.py`, then remembers both. You can also double-click `register_tracks_alyx.bat` / `.command`.

**Using it.** Every `*_pts.csv` file in the folder gets a row with a checkbox; tick single files or use **Check all boxes for upload**. Each file is checked first:

- **Blocked** (cannot be ticked): a file name not in the form `yyyy-mm-dd_Subject_001_probe00a_pts.csv`; no exact matching session or probe insertion in Alyx (the tool never creates insertions); a track closer to another probe's plan in the same session; the wrong hemisphere; too few points or points outside the CCF grid; identical points under two names; a probe that already has two histology tracks.
- **Warning** (needs an explicit acknowledgement to upload): a histology track or an ephys alignment already exists and will be affected; the shank order differs from the plan (a possible A/D label swap); the track is far from its plan, or clearly closer to another day's plan.

Everything is re-checked just before upload, then a review screen lists exactly what will be written. After each upload the tool reads the track back from Alyx to confirm it, and records it in `alyx_track_upload_log.csv` next to the files.

The tests run offline against a stand-in Alyx: `python tests/test_register_tracks_alyx.py -v`

**Known issue:** the trajectory viewers and the coordinates GUI run in `iblenv`, so their Alyx features hit the same `406` error until they are moved to an environment with ONE 3.4 or newer.

## Updating these tools

If you downloaded with `git clone`, open a terminal in the `ibl-probe-tools` folder and run:

```
git pull
```

If `git pull` refuses because you edited a `.bat` / `.command` launcher (or ran `chmod +x` on macOS), set your edits aside, update, then restore them:

```
git stash
git pull
git stash pop
```

If you downloaded a ZIP instead, download it again and redo the launcher edit.

## Orientation convention

All three tools (Urchin, the slice panels, and Lasagna) use the lab's "surgeon's view": looking down on the top of the brain from behind the animal. The horizontal view has anterior at the top and the animal's right on screen-right; the coronal view has dorsal up and right on right; the sagittal view has anterior on the right. The brain is nearly symmetric, so judge hemispheres by these conventions (or Urchin's axes cross, whose blue arm points to the animal's left), not by the shape on screen.

## Full usage instructions

See the **[user guide](https://kimiline.github.io/ibl-probe-tools/)** for a step-by-step walkthrough of the trajectory viewer, Lasagna, and registering tracks to Alyx. The page source is [`docs/index.html`](docs/index.html).
