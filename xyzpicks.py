import argparse
import sys
import numpy as np
from iblatlas.atlas import AllenAtlas
from one.api import ONE

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct AP-ML-DV track from xyz_picks for a given insertion PID."
    )
    parser.add_argument(
        "pid",
        help="Alyx insertion PID (e.g., 00f51f80-11a7-4a66-93d1-50afe1d40614)"
    )
    args = parser.parse_args()

    one = ONE(base_url="https://alyx.internationalbrainlab.org")
    ba = AllenAtlas()

    try:
        insertion = one.alyx.rest("insertions", "list", id=args.pid)[0]
    except IndexError:
        print(f"PID not found: {args.pid}", file=sys.stderr)
        sys.exit(1)

    jpicks = insertion.get("json", {})
    if "xyz_picks" not in jpicks or not jpicks["xyz_picks"]:
        print(f"No 'xyz_picks' found for PID: {args.pid}", file=sys.stderr)
        sys.exit(1)

    xyz = np.array(jpicks["xyz_picks"], dtype=float) / 1e6  # meters
    ixyz = ba.bc.xyz2i(xyz)  # [ML, AP, DV]
    ixyz[:, 1] = 527 - ixyz[:, 1]  # flip AP axis (as in original)
    ap_ml_dv = np.c_[ixyz[:, 1], ixyz[:, 0], ixyz[:, 2]].astype(int)

    out_path = f"{args.pid}_reconstructed_lasgana_track.csv"
    np.savetxt(out_path, ap_ml_dv, delimiter=",", fmt="%d")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
