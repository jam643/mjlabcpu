"""Photorealistic renderer using Blender Cycles via a persistent subprocess.

No ``bpy`` pip package required — uses system Blender (which bundles its own
Python + bpy internally), so it works with any host Python version.

**Prerequisites**: install Blender ≥ 4.2 from https://www.blender.org/download/
(macOS: drag to /Applications; the renderer auto-detects it).

Usage::

    uv pip install -e ".[photo]"   # no-op; installs nothing — Blender is system-level

    uv run python scripts/view.py cartpole --rgb --photo --steps 500

Performance expectations (Apple Silicon, 64 samples):
    Metal backend  ~2–5 s/frame
    CPU   backend  ~8–15 s/frame
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Blender render-server script (runs inside Blender's bundled Python)
# ---------------------------------------------------------------------------

_BLENDER_SCRIPT = textwrap.dedent("""\
    \"\"\"Blender background render server.  Driven by single-line stdin commands.\"\"\"
    import sys, os, platform
    import numpy as np
    import bpy

    # ── parse CLI args ──────────────────────────────────────────────────────
    sep = sys.argv.index('--')
    args = sys.argv[sep + 1:]
    model_path, state_path, output_path = args[0], args[1], args[2]
    width, height, samples = int(args[3]), int(args[4]), int(args[5])
    device   = args[6]
    cam_pos  = (float(args[7]),  float(args[8]),  float(args[9]))
    cam_look = (float(args[10]), float(args[11]), float(args[12]))

    # ── math helpers ────────────────────────────────────────────────────────
    def _quat_wxyz_to_mat(q):
        w,x,y,z = float(q[0]),float(q[1]),float(q[2]),float(q[3])
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
            [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
        ], dtype=np.float64)

    def _mat_to_quat_wxyz(R):
        t = R[0,0]+R[1,1]+R[2,2]
        if t > 0:
            s=0.5/np.sqrt(t+1); w=0.25/s
            x=(R[2,1]-R[1,2])*s; y=(R[0,2]-R[2,0])*s; z=(R[1,0]-R[0,1])*s
        elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
            s=2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]); w=(R[2,1]-R[1,2])/s
            x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
        elif R[1,1]>R[2,2]:
            s=2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2]); w=(R[0,2]-R[2,0])/s
            x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
        else:
            s=2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1]); w=(R[1,0]-R[0,1])/s
            x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
        return np.array([w,x,y,z], dtype=np.float64)

    # ── scene setup ─────────────────────────────────────────────────────────
    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    # Compute device
    cycles_addon = bpy.context.preferences.addons.get('cycles')
    if cycles_addon:
        cp = cycles_addon.preferences
        use_metal = (device == 'metal') or (
            device == 'auto'
            and platform.machine() == 'arm64'
            and platform.system() == 'Darwin'
        )
        if use_metal:
            try:
                cp.compute_device_type = 'METAL'
                cp.refresh_devices()
                scene.cycles.device = 'GPU'
            except Exception:
                scene.cycles.device = 'CPU'
        else:
            scene.cycles.device = 'CPU'

    # Camera
    cd = bpy.data.cameras.new('Camera')
    co = bpy.data.objects.new('Camera', cd)
    scene.collection.objects.link(co); scene.camera = co
    co.location = cam_pos
    tg = bpy.data.objects.new('Target', None)
    tg.location = cam_look
    scene.collection.objects.link(tg)
    con = co.constraints.new(type='TRACK_TO')
    con.target = tg; con.track_axis = 'TRACK_NEGATIVE_Z'; con.up_axis = 'UP_Y'

    # Sun light
    sd = bpy.data.lights.new('Sun', type='SUN'); sd.energy = 3.0
    so = bpy.data.objects.new('Sun', sd)
    so.location = (5,-5,8); so.rotation_euler = (0.6,0,0.8)
    scene.collection.objects.link(so)

    # World background
    w = bpy.data.worlds.new('World'); scene.world = w; w.use_nodes = True
    bg = w.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Color'].default_value = (0.05,0.05,0.05,1)
        bg.inputs['Strength'].default_value = 0.5

    # ── load model geometry ──────────────────────────────────────────────────
    md = np.load(model_path, allow_pickle=False)
    geom_type    = md['geom_type']
    geom_size    = md['geom_size']
    geom_rgba    = md['geom_rgba']
    geom_bodyid  = md['geom_bodyid']
    geom_pos_arr = md['geom_pos']
    geom_quat_arr= md['geom_quat']
    geom_dataid  = md['geom_dataid']
    has_mesh = bool(md['has_mesh'])
    if has_mesh:
        mesh_vert    = md['mesh_vert']
        mesh_face    = md['mesh_face']
        mesh_vertadr = md['mesh_vertadr']
        mesh_vertnum = md['mesh_vertnum']
        mesh_faceadr = md['mesh_faceadr']
        mesh_facenum = md['mesh_facenum']

    PLANE,SPHERE,CAPSULE,CYLINDER,BOX,MESH = 0,2,3,5,6,7

    body_objs = {}   # body_id → [bpy_obj, ...]
    body_gids = {}   # body_id → [geom_index, ...]

    def _make_mat(rgba, idx):
        mat = bpy.data.materials.new(f'mat{idx}')
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF') or \\
               mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = (
            float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0)
        bsdf.inputs['Roughness'].default_value = 0.4
        bsdf.inputs['Metallic'].default_value  = 0.1
        return mat

    for gi in range(len(geom_type)):
        rgba = geom_rgba[gi]
        if float(rgba[3]) == 0.0:
            continue
        gtype = int(geom_type[gi])
        sz    = geom_size[gi]
        bid   = int(geom_bodyid[gi])
        mat   = _make_mat(rgba, gi)
        obj   = None

        if gtype == PLANE:
            s = float(sz[0]) if float(sz[0]) > 0 else 10.0
            bpy.ops.mesh.primitive_plane_add(size=2.0)
            obj = bpy.context.active_object; obj.scale = (s, s, 1.0)
        elif gtype == SPHERE:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=float(sz[0]))
            obj = bpy.context.active_object
        elif gtype in (CAPSULE, CYLINDER):
            bpy.ops.mesh.primitive_cylinder_add(
                radius=float(sz[0]), depth=float(sz[1]) * 2.0)
            obj = bpy.context.active_object
        elif gtype == BOX:
            bpy.ops.mesh.primitive_cube_add(size=2.0)
            obj = bpy.context.active_object
            obj.scale = (float(sz[0]), float(sz[1]), float(sz[2]))
        elif gtype == MESH and has_mesh:
            mid = int(geom_dataid[gi])
            vs, vc = int(mesh_vertadr[mid]), int(mesh_vertnum[mid])
            fs, fc = int(mesh_faceadr[mid]), int(mesh_facenum[mid])
            if vc > 0 and fc > 0:
                verts = [(float(v[0]),float(v[1]),float(v[2]))
                         for v in mesh_vert[vs:vs+vc]]
                faces = [(int(f[0]),int(f[1]),int(f[2]))
                         for f in mesh_face[fs:fs+fc]]
                mdata = bpy.data.meshes.new(f'mesh{mid}')
                mdata.from_pydata(verts, [], faces); mdata.update()
                obj = bpy.data.objects.new(f'mobj{mid}', mdata)
                scene.collection.objects.link(obj)

        if obj is None:
            continue

        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        obj.rotation_mode = 'QUATERNION'

        if bid not in body_objs:
            body_objs[bid] = []; body_gids[bid] = []
        body_objs[bid].append(obj); body_gids[bid].append(gi)

    # Signal host that scene is ready
    print('ready', flush=True)

    # ── render loop ──────────────────────────────────────────────────────────
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == 'render':
            state    = np.load(state_path, allow_pickle=False)
            xpos_all = state['xpos']                      # (nbody, 3)
            xmat_all = state['xmat'].reshape(-1, 3, 3)    # (nbody, 3, 3)

            for bid, objs in body_objs.items():
                xp = xpos_all[bid]; xm = xmat_all[bid]
                for obj, gi in zip(objs, body_gids[bid]):
                    gp = geom_pos_arr[gi]; gq = geom_quat_arr[gi]
                    wp = xp + xm @ gp
                    wm = xm @ _quat_wxyz_to_mat(gq)
                    wq = _mat_to_quat_wxyz(wm)
                    obj.location = (float(wp[0]),float(wp[1]),float(wp[2]))
                    obj.rotation_quaternion = (
                        float(wq[0]),float(wq[1]),float(wq[2]),float(wq[3]))

            bpy.ops.render.render(write_still=False)
            img = bpy.data.images.get('Render Result')
            if img:
                px = np.array(img.pixels, dtype=np.float32).reshape(height, width, 4)
                rgb = np.clip(px[::-1, :, :3] * 255.0, 0, 255).astype(np.uint8)
                np.save(output_path, rgb)
                print('done', flush=True)
            else:
                print('error', flush=True)
        elif cmd == 'quit':
            break

    sys.exit(0)
""")


# ---------------------------------------------------------------------------
# PhotoRenderer
# ---------------------------------------------------------------------------


class PhotoRenderer:
    """Photorealistic offscreen renderer using a persistent Blender Cycles subprocess.

    Finds system Blender (``/Applications/Blender.app`` or PATH), launches it
    once in background mode, and communicates via stdin/stdout + temp numpy
    files.  Works with any host Python version — no ``bpy`` pip package needed.

    Args:
        model: MuJoCo model (geometry, materials).
        width: Render width in pixels.
        height: Render height in pixels.
        samples: Cycles sample count — lower is faster.
        device: ``"auto"`` selects Metal on Apple Silicon, else CPU.
        camera_pos: Camera world position ``(x, y, z)``.
        camera_look_at: Point the camera tracks toward.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = 640,
        height: int = 480,
        samples: int = 64,
        device: str = "auto",
        camera_pos: tuple = (2.5, -2.5, 1.5),
        camera_look_at: tuple = (0.0, 0.0, 0.5),
    ) -> None:
        self._width = width
        self._height = height
        self._blender_exe = self._find_blender()

        # Temp directory for inter-process communication
        self._tmpdir = tempfile.mkdtemp(prefix="mjlab_photo_")
        model_path = os.path.join(self._tmpdir, "model.npz")
        self._state_path = os.path.join(self._tmpdir, "state.npz")
        self._output_path = os.path.join(self._tmpdir, "render.npy")
        script_path = os.path.join(self._tmpdir, "render_scene.py")

        # Serialize model geometry once
        self._save_model(model, model_path)

        # Write embedded Blender script to disk
        with open(script_path, "w") as f:
            f.write(_BLENDER_SCRIPT)

        # Launch persistent Blender background process
        cmd = [
            self._blender_exe,
            "--background",
            "--python",
            script_path,
            "--",
            model_path,
            self._state_path,
            self._output_path,
            str(width),
            str(height),
            str(samples),
            device,
            str(camera_pos[0]),
            str(camera_pos[1]),
            str(camera_pos[2]),
            str(camera_look_at[0]),
            str(camera_look_at[1]),
            str(camera_look_at[2]),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        # Wait for Blender to finish building the scene
        line = self._proc.stdout.readline().strip()
        if line != "ready":
            self._proc.kill()
            raise RuntimeError(
                f"Blender startup failed (got {line!r}). Re-run with stderr visible for details."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_blender() -> str:
        """Return path to Blender executable, or raise FileNotFoundError."""
        import glob
        import shutil

        exe = shutil.which("blender")
        if exe:
            return exe

        candidates = ["/Applications/Blender.app/Contents/MacOS/Blender"]
        candidates += glob.glob("/Applications/Blender*.app/Contents/MacOS/Blender")
        for path in candidates:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        raise FileNotFoundError(
            "Blender not found on PATH or in /Applications. "
            "Install from https://www.blender.org/download/ (≥ 4.2)."
        )

    @staticmethod
    def _save_model(model: mujoco.MjModel, path: str) -> None:
        """Serialize model geometry to an .npz file for the Blender subprocess."""
        has_mesh = model.nmesh > 0
        kwargs: dict = dict(
            geom_type=model.geom_type,
            geom_size=model.geom_size,
            geom_rgba=model.geom_rgba,
            geom_bodyid=model.geom_bodyid,
            geom_pos=model.geom_pos,
            geom_quat=model.geom_quat,
            geom_dataid=model.geom_dataid,
            has_mesh=np.array(has_mesh),
        )
        if has_mesh:
            kwargs.update(
                mesh_vert=model.mesh_vert,
                mesh_face=model.mesh_face,
                mesh_vertadr=model.mesh_vertadr,
                mesh_vertnum=model.mesh_vertnum,
                mesh_faceadr=model.mesh_faceadr,
                mesh_facenum=model.mesh_facenum,
            )
        np.savez(path, **kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, data: mujoco.MjData) -> None:
        """Write body transforms so the next ``render()`` picks them up.

        Args:
            data: MuJoCo data object (env 0) after ``mj_step`` / ``mj_forward``.
        """
        np.savez(
            self._state_path,
            xpos=data.xpos,  # (nbody, 3)
            xmat=data.xmat,  # (nbody, 9) — reshaped to (nbody,3,3) in script
        )

    def render(self) -> np.ndarray:
        """Render the current scene.

        Returns:
            RGB image as ``(height, width, 3)`` uint8 numpy array.
        """
        if self._proc.poll() is not None:
            raise RuntimeError("Blender subprocess exited unexpectedly.")

        self._proc.stdin.write("render\n")
        self._proc.stdin.flush()

        response = self._proc.stdout.readline().strip()
        if response == "error":
            raise RuntimeError("Blender render failed — Render Result image not found.")
        if response != "done":
            raise RuntimeError(f"Unexpected response from Blender: {response!r}")

        return np.load(self._output_path)

    def close(self) -> None:
        """Shut down the Blender subprocess and remove temp files."""
        import shutil

        try:
            if self._proc.poll() is None:
                self._proc.stdin.write("quit\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()
        finally:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
