"""Download Franka Panda (nohand) assets from MuJoCo Menagerie.

Fetches panda_nohand.xml and all referenced OBJ mesh files from GitHub raw URLs.

Usage:
    uv run python examples/download_panda_assets.py
"""

from __future__ import annotations

import os
import urllib.request
import xml.etree.ElementTree as ET

BASE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/franka_emika_panda"
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "assets", "panda_push")
ASSETS_DIR = os.path.join(OUT_DIR, "assets")
XML_NAME = "panda_nohand.xml"


def download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  {url}\n    -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    # 1. Download the main XML
    xml_dest = os.path.join(OUT_DIR, XML_NAME)
    print(f"Downloading {XML_NAME}...")
    download(f"{BASE_URL}/{XML_NAME}", xml_dest)

    # 2. Parse the XML to find all mesh file references
    tree = ET.parse(xml_dest)
    root = tree.getroot()

    mesh_files: list[str] = []

    # <compiler meshdir="assets"> or similar — find all <mesh file="..."> elements
    for mesh in root.iter("mesh"):
        fname = mesh.get("file")
        if fname:
            mesh_files.append(fname)

    # Also check <include file="..."> elements (Menagerie sometimes splits files)
    for include in root.iter("include"):
        fname = include.get("file")
        if fname:
            # Download included XML files too
            inc_dest = os.path.join(OUT_DIR, fname)
            print(f"Downloading included file {fname}...")
            download(f"{BASE_URL}/{fname}", inc_dest)

            # Parse included file for further mesh references
            try:
                inc_tree = ET.parse(inc_dest)
                for mesh in inc_tree.getroot().iter("mesh"):
                    mf = mesh.get("file")
                    if mf:
                        mesh_files.append(mf)
            except ET.ParseError:
                pass

    # 3. Download mesh files into assets/
    if mesh_files:
        print(f"\nDownloading {len(mesh_files)} mesh file(s)...")
        for fname in mesh_files:
            dest = os.path.join(ASSETS_DIR, fname)
            download(f"{BASE_URL}/assets/{fname}", dest)
    else:
        print("No mesh files found in XML (may use primitives only).")

    # 4. Print body names so user can verify EEF link name
    print("\nBody names in compiled model:")
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(xml_dest)
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            print(f"  [{i}] {name}")
    except Exception as exc:
        print(f"  (could not load model to list bodies: {exc})")

    print("\nDone. Assets saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
