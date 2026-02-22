"""Photorealistic renderer using Blender Cycles via the ``bpy`` PyPI package.

**Prerequisites**::

    uv pip install -e ".[photo]"   # installs bpy>=4.2.0 (requires Python 3.11)

Usage::

    uv run python scripts/view.py cartpole --rgb --photo --steps 500

Performance expectations (Apple Silicon, 64 samples):
    Metal backend  ~2–5 s/frame
    CPU   backend  ~8–15 s/frame
"""

from __future__ import annotations

import os
import platform
import shutil
import tempfile

import bpy
import mujoco
import numpy as np


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a wxyz quaternion."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


class PhotoRenderer:
    """Photorealistic offscreen renderer using Blender Cycles via ``bpy``.

    Initializes a Blender scene from a MuJoCo model and renders it in-process
    using Cycles.  No subprocess or system Blender installation required.

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
        self._model = model
        self._last_xpos: np.ndarray | None = None
        self._last_xmat: np.ndarray | None = None

        # Temp directory for render output files.
        # In headless bpy, write_still=True saves to filepath + ".png" (no frame suffix).
        self._tmpdir = tempfile.mkdtemp(prefix="mjlab_photo_")
        self._render_path = os.path.join(self._tmpdir, "frame.png")

        # Fresh Blender scene
        bpy.ops.wm.read_homefile(use_empty=True)
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.resolution_percentage = 100
        # Set filepath without extension — bpy appends ".png" in background mode.
        scene.render.filepath = os.path.join(self._tmpdir, "frame")
        scene.render.image_settings.file_format = "PNG"
        scene.frame_current = 1

        # GPU / Metal device
        cycles_addon = bpy.context.preferences.addons.get("cycles")
        if cycles_addon:
            cp = cycles_addon.preferences
            use_metal = (device == "metal") or (
                device == "auto" and platform.machine() == "arm64" and platform.system() == "Darwin"
            )
            if use_metal:
                try:
                    cp.compute_device_type = "METAL"
                    cp.refresh_devices()
                    scene.cycles.device = "GPU"
                except Exception:
                    scene.cycles.device = "CPU"
            else:
                scene.cycles.device = "CPU"

        # Camera
        cam_data = bpy.data.cameras.new("Camera")
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam_obj)
        scene.camera = cam_obj
        cam_obj.location = camera_pos

        target = bpy.data.objects.new("Target", None)
        target.location = camera_look_at
        scene.collection.objects.link(target)
        con = cam_obj.constraints.new(type="TRACK_TO")
        con.target = target
        con.track_axis = "TRACK_NEGATIVE_Z"
        con.up_axis = "UP_Y"

        # Sun light
        sun_data = bpy.data.lights.new("Sun", type="SUN")
        sun_data.energy = 3.0
        sun_obj = bpy.data.objects.new("Sun", sun_data)
        sun_obj.location = (5, -5, 8)
        sun_obj.rotation_euler = (0.6, 0, 0.8)
        scene.collection.objects.link(sun_obj)

        # World background
        world = bpy.data.worlds.new("World")
        scene.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1)
            bg.inputs["Strength"].default_value = 0.5

        # Geometry
        self._body_objs: dict[int, list] = {}  # body_id → [bpy objects]
        self._body_gids: dict[int, list] = {}  # body_id → [geom indices]
        self._setup_geometry(model, scene)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_material(rgba: np.ndarray, idx: int):
        mat = bpy.data.materials.new(f"mat{idx}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF") or mat.node_tree.nodes.new(
            "ShaderNodeBsdfPrincipled"
        )
        bsdf.inputs["Base Color"].default_value = (
            float(rgba[0]),
            float(rgba[1]),
            float(rgba[2]),
            1.0,
        )
        bsdf.inputs["Roughness"].default_value = 0.4
        bsdf.inputs["Metallic"].default_value = 0.1
        return mat

    def _setup_geometry(self, model: mujoco.MjModel, scene) -> None:
        PLANE, SPHERE, CAPSULE, CYLINDER, BOX, MESH = 0, 2, 3, 5, 6, 7
        has_mesh = model.nmesh > 0

        for gi in range(model.ngeom):
            rgba = model.geom_rgba[gi]
            if float(rgba[3]) == 0.0:
                continue
            gtype = int(model.geom_type[gi])
            sz = model.geom_size[gi]
            bid = int(model.geom_bodyid[gi])
            mat = self._make_material(rgba, gi)
            obj = None

            if gtype == PLANE:
                s = float(sz[0]) if float(sz[0]) > 0 else 10.0
                bpy.ops.mesh.primitive_plane_add(size=2.0)
                obj = bpy.context.active_object
                obj.scale = (s, s, 1.0)
            elif gtype == SPHERE:
                bpy.ops.mesh.primitive_uv_sphere_add(radius=float(sz[0]))
                obj = bpy.context.active_object
            elif gtype in (CAPSULE, CYLINDER):
                bpy.ops.mesh.primitive_cylinder_add(radius=float(sz[0]), depth=float(sz[1]) * 2.0)
                obj = bpy.context.active_object
            elif gtype == BOX:
                bpy.ops.mesh.primitive_cube_add(size=2.0)
                obj = bpy.context.active_object
                obj.scale = (float(sz[0]), float(sz[1]), float(sz[2]))
            elif gtype == MESH and has_mesh:
                mid = int(model.geom_dataid[gi])
                vs = int(model.mesh_vertadr[mid])
                vc = int(model.mesh_vertnum[mid])
                fs = int(model.mesh_faceadr[mid])
                fc = int(model.mesh_facenum[mid])
                if vc > 0 and fc > 0:
                    verts = [
                        (float(v[0]), float(v[1]), float(v[2]))
                        for v in model.mesh_vert[vs : vs + vc]
                    ]
                    faces = [
                        (int(f[0]), int(f[1]), int(f[2])) for f in model.mesh_face[fs : fs + fc]
                    ]
                    mdata = bpy.data.meshes.new(f"mesh{mid}")
                    mdata.from_pydata(verts, [], faces)
                    mdata.update()
                    obj = bpy.data.objects.new(f"mobj{mid}", mdata)
                    scene.collection.objects.link(obj)

            if obj is None:
                continue

            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)
            obj.rotation_mode = "QUATERNION"

            if bid not in self._body_objs:
                self._body_objs[bid] = []
                self._body_gids[bid] = []
            self._body_objs[bid].append(obj)
            self._body_gids[bid].append(gi)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, data: mujoco.MjData) -> None:
        """Store body transforms so the next ``render()`` picks them up.

        Args:
            data: MuJoCo data object (env 0) after ``mj_step`` / ``mj_forward``.
        """
        self._last_xpos = data.xpos.copy()  # (nbody, 3)
        self._last_xmat = data.xmat.reshape(-1, 3, 3).copy()  # (nbody, 3, 3)

    def render(self) -> np.ndarray:
        """Render the current scene.

        Returns:
            RGB image as ``(height, width, 3)`` uint8 numpy array.
        """
        if self._last_xpos is None or self._last_xmat is None:
            raise RuntimeError("Call update() before render().")

        xpos_all = self._last_xpos
        xmat_all = self._last_xmat
        model = self._model

        for bid, objs in self._body_objs.items():
            xp = xpos_all[bid]
            xm = xmat_all[bid]
            for obj, gi in zip(objs, self._body_gids[bid], strict=True):
                gp = model.geom_pos[gi]
                gq = model.geom_quat[gi]
                wp = xp + xm @ gp
                wm = xm @ _quat_wxyz_to_mat(gq)
                wq = _mat_to_quat_wxyz(wm)
                obj.location = (float(wp[0]), float(wp[1]), float(wp[2]))
                obj.rotation_quaternion = (
                    float(wq[0]),
                    float(wq[1]),
                    float(wq[2]),
                    float(wq[3]),
                )

        # write_still=True saves to scene.render.filepath + frame number + extension.
        # In headless bpy the in-memory 'Render Result' pixels buffer is empty;
        # writing to disk and reloading is the reliable approach.
        bpy.ops.render.render(write_still=True)
        if not os.path.exists(self._render_path):
            raise RuntimeError(
                f"Blender render failed — output file not found: {self._render_path}"
            )

        img = bpy.data.images.load(self._render_path)
        try:
            px = np.empty(self._height * self._width * 4, dtype=np.float32)
            img.pixels.foreach_get(px)
            px = px.reshape(self._height, self._width, 4)
        finally:
            bpy.data.images.remove(img)
            os.unlink(self._render_path)

        return np.clip(px[::-1, :, :3] * 255.0, 0, 255).astype(np.uint8)

    def close(self) -> None:
        """Remove temporary render directory."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)
