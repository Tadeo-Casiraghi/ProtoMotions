"""Build smpl_humanoid10bis_merged.{usda,xml}: the 8 socket boxes and the tibial pylon of
smpl_humanoid10bis become collision shapes of ONE rigid link, "socket_assembly", fixed to
prosthetic_assembly2.

Why: they were 9 separate links held by fixed joints. In PhysX's reduced-coordinate articulation a
fixed joint is already exactly rigid, but each link still costs solver work at every physics step,
so 46 links -> 38 links. prosthetic_assembly2 itself is left untouched (its mass, COM and frame),
so the shank IMU that reads its COM acceleration sees the same thing.

What is preserved exactly:
  * every collision shape (geometry, local pose, collision settings, physics material binding),
    copied from the original prims, so the skin <-> socket contact is the same;
  * the total mass, COM and inertia of the 9 bodies, combined with the parallel-axis theorem
    into an explicit mass / centerOfMass / diagonalInertia / principalAxes on the new link.

Usage (from the repo root, in the env_isaaclab conda env):
    python protomotions/data/assets/scripts/merge_socket_links.py
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

ASSETS = Path(__file__).resolve().parents[1]
ROOT = "/smpl_humanoid10"
PARENT = "prosthetic_assembly2"
NEW = "socket_assembly"
SOCKETS = [f"socket_box_{s}_{l}" for l in ("top", "bottom") for s in ("posterior", "medial", "anterior", "lateral")]
SOURCES = ["osl_tibial_pylon"] + SOCKETS


# ----------------------------------------------------------------------------------------------
# Mass properties
# ----------------------------------------------------------------------------------------------
def quat_wxyz_to_mat(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat_wxyz(R):
    q = Gf.Matrix3d(*R.T.flatten().tolist()).ExtractRotation().GetQuat()  # Gf is row-vector: pass R^T
    return (q.GetReal(), *q.GetImaginary())


def combine(parts):
    """parts: list of (mass, com[3], inertia_about_com[3x3]) in one frame -> (M, C, I_C)."""
    M = sum(m for m, _, _ in parts)
    C = sum(m * c for m, c, _ in parts) / M
    I = np.zeros((3, 3))
    for m, c, Ic in parts:
        d = c - C
        I += Ic + m * (d @ d * np.eye(3) - np.outer(d, d))
    return M, C, I


def principal(I):
    lam, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    return lam, V  # I = V diag(lam) V^T


# ----------------------------------------------------------------------------------------------
# USD
# ----------------------------------------------------------------------------------------------
def gf_to_np(m):  # Gf.Matrix4d (row-vector convention) -> (R column-vector, t)
    a = np.array([[m[i][j] for j in range(4)] for i in range(4)])
    return a[:3, :3].T, a[3, :3]


def body_mass_parts(stage, name, rel_R, rel_t):
    """Mass properties of a source body expressed in the new link's frame."""
    prim = stage.GetPrimAtPath(f"{ROOT}/bodies/{name}")
    mass = prim.GetAttribute("physics:mass")
    if mass and mass.HasAuthoredValue():  # explicit mass (the pylon)
        m = float(mass.Get())
        com = np.array(prim.GetAttribute("physics:centerOfMass").Get(), dtype=float)
        diag = np.array(prim.GetAttribute("physics:diagonalInertia").Get(), dtype=float)
        pa = prim.GetAttribute("physics:principalAxes").Get()
        Rp = quat_wxyz_to_mat(pa.GetReal(), *pa.GetImaginary())
        Ic = Rp @ np.diag(diag) @ Rp.T
    else:  # density x a single box shape (the socket boxes)
        rho = float(prim.GetAttribute("physics:density").Get())
        cubes = [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Cube) and p.HasAPI("PhysicsCollisionAPI")]
        assert len(cubes) == 1, f"{name}: expected one collision box, got {len(cubes)}"
        cube = cubes[0]
        # box full sizes in the body frame (the cube has no rotation/offset in these bodies)
        to_body = UsdGeom.Xformable(cube).ComputeLocalToWorldTransform(0) * UsdGeom.Xformable(
            prim).ComputeLocalToWorldTransform(0).GetInverse()
        Rb, tb = gf_to_np(to_body)
        size = float(UsdGeom.Cube(cube).GetSizeAttr().Get())
        full = np.linalg.norm(Rb, axis=0) * size  # column norms = per-axis scale
        assert np.allclose(tb, 0, atol=1e-7) and np.allclose(Rb / np.linalg.norm(Rb, axis=0), np.eye(3), atol=1e-6)
        a, b, c = full
        m = rho * a * b * c
        com = np.zeros(3)
        Ic = m / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
    return m, rel_R @ com + rel_t, rel_R @ Ic @ rel_R.T


def merge_usd(src: Path, dst: Path):
    stage = Usd.Stage.Open(str(src))
    layer = stage.GetRootLayer()
    bodies = f"{ROOT}/bodies"
    joints = f"{ROOT}/joints"
    parent_prim = stage.GetPrimAtPath(f"{bodies}/{PARENT}")
    M_parent = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(0)

    # --- new link: same frame as prosthetic_assembly2, API schemas of a socket box body ---
    new_path = Sdf.Path(f"{bodies}/{NEW}")
    Sdf.CopySpec(layer, Sdf.Path(f"{bodies}/{SOCKETS[0]}"), layer, new_path)
    new_spec = layer.GetPrimAtPath(new_path)
    for child in list(new_spec.nameChildren.keys()):
        del new_spec.nameChildren[child]
    for attr in ("physics:density",):
        if attr in new_spec.attributes:
            new_spec.RemoveProperty(new_spec.attributes[attr])
    parent_spec = layer.GetPrimAtPath(f"{bodies}/{PARENT}")
    for op in ("xformOp:translate", "xformOp:orient", "xformOp:scale"):
        new_spec.attributes[op].default = parent_spec.attributes[op].default

    # --- move every collision/visual prim of the sources under the new link ---
    parts = []
    for name in SOURCES:
        src_prim = stage.GetPrimAtPath(f"{bodies}/{name}")
        M_rel = UsdGeom.Xformable(src_prim).ComputeLocalToWorldTransform(0) * M_parent.GetInverse()
        R, t = gf_to_np(M_rel)
        parts.append(body_mass_parts(stage, name, R, t))
        xf = Gf.Transform(M_rel)
        for child in ("collisions", "visuals"):
            child_path = Sdf.Path(f"{bodies}/{name}/{child}")
            if not layer.GetPrimAtPath(child_path):
                continue
            dst_path = new_path.AppendChild(f"{name}_{child}")
            Sdf.CopySpec(layer, child_path, layer, dst_path)
            x = UsdGeom.Xformable(stage.GetPrimAtPath(dst_path))
            x.ClearXformOpOrder()
            x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(xf.GetTranslation()))
            x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(xf.GetRotation().GetQuat()))
            x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1, 1, 1))

    M, C, I = combine(parts)
    lam, V = principal(I)
    qw, qx, qy, qz = mat_to_quat_wxyz(V)
    new_prim = stage.GetPrimAtPath(new_path)
    new_prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(float(M))
    new_prim.CreateAttribute("physics:centerOfMass", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(*C.tolist()))
    new_prim.CreateAttribute("physics:diagonalInertia", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*lam.tolist()))
    new_prim.CreateAttribute("physics:principalAxes", Sdf.ValueTypeNames.Quatf).Set(Gf.Quatf(qw, qx, qy, qz))

    # --- fixed joint prosthetic_assembly2 -> socket_assembly (identity, same frame) ---
    new_joint = Sdf.Path(f"{joints}/{NEW}")
    Sdf.CopySpec(layer, Sdf.Path(f"{joints}/{SOCKETS[0]}"), layer, new_joint)
    j = stage.GetPrimAtPath(new_joint)
    j.GetRelationship("physics:body0").SetTargets([Sdf.Path(f"{bodies}/{PARENT}")])
    j.GetRelationship("physics:body1").SetTargets([new_path])
    j.GetAttribute("physics:localPos0").Set(Gf.Vec3f(0, 0, 0))
    j.GetAttribute("physics:localRot0").Set(Gf.Quatf(1, 0, 0, 0))
    j.GetAttribute("physics:localPos1").Set(Gf.Vec3f(0, 0, 0))
    j.GetAttribute("physics:localRot1").Set(Gf.Quatf(1, 0, 0, 0))

    # --- drop the source bodies / joints and fix every relationship that listed them ---
    removed_bodies = {Sdf.Path(f"{bodies}/{n}") for n in SOURCES}
    removed_joints = {Sdf.Path(f"{joints}/{n}") for n in SOURCES}
    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            if not any(t in removed_bodies or t in removed_joints for t in targets):
                continue
            new_targets, inserted = [], False
            for t in targets:
                if t in removed_bodies or t in removed_joints:
                    if not inserted:  # put the new link/joint where the first removed one was
                        new_targets.append(new_path if t in removed_bodies else new_joint)
                        inserted = True
                    continue
                new_targets.append(t)
            rel.SetTargets(new_targets)
    for p in sorted(removed_bodies | removed_joints):
        stage.RemovePrim(p)

    layer.Export(str(dst))
    return M, C, lam, (qw, qx, qy, qz), I


# ----------------------------------------------------------------------------------------------
# MJCF (kinematic tree used for the robot config: body names, ordering, offsets)
# ----------------------------------------------------------------------------------------------
def merge_mjcf(src: Path, dst: Path, M, C, lam, q):
    xml = src.read_text()
    start = xml.index('<body name="osl_tibial_pylon">')
    end = xml.index('<body name="osl_ankle_assembly"')
    block = xml[start:end]
    indent = xml[xml.rindex("\n", 0, start) + 1:start]

    geoms = []
    # pylon: body at the parent origin, keep its geom as is
    pylon_geom = re.search(r"<geom [^>]*/>", block[: block.index("</body>")]).group(0)
    geoms.append(pylon_geom.replace("<geom ", '<geom name="osl_tibial_pylon_geom" ', 1))
    for name in SOCKETS:
        m = re.search(
            rf'<body name="{name}" pos="([^"]+)" quat="([^"]+)">\s*(<geom [^>]*/>)', block)
        pos, quat, geom = m.group(1), m.group(2), m.group(3)
        geom = re.sub(r'\spos="[^"]*"', "", geom)  # geoms sat at their body origin
        assert "quat=" not in geom, name
        geoms.append(geom.replace("<geom ", f'<geom name="{name}_geom" pos="{pos}" quat="{quat}" ', 1))

    fmt = lambda v: " ".join(f"{x:.9g}" for x in v)  # noqa: E731
    body = (
        f'<body name="{NEW}" pos="0 0 0" quat="1 0 0 0">\n'
        f'{indent}  <!-- merged by protomotions/data/assets/scripts/merge_socket_links.py: '
        f'osl_tibial_pylon + 8 socket boxes, combined inertia -->\n'
        f'{indent}  <inertial pos="{fmt(C)}" quat="{fmt(q)}" mass="{M:.9g}" diaginertia="{fmt(lam)}"/>\n'
        + "".join(f"{indent}  {g}\n" for g in geoms)
        + f"{indent}</body>\n\n{indent}"
    )
    dst.write_text(xml[:start] + body + xml[end:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", default=str(ASSETS / "usd/smpl_humanoid10bis.usda"))
    ap.add_argument("--mjcf", default=str(ASSETS / "mjcf/smpl_humanoid10bis.xml"))
    ap.add_argument("--out-usd", default=str(ASSETS / "usd/smpl_humanoid10bis_merged.usda"))
    ap.add_argument("--out-mjcf", default=str(ASSETS / "mjcf/smpl_humanoid10bis_merged.xml"))
    a = ap.parse_args()
    M, C, lam, q, I = merge_usd(Path(a.usd), Path(a.out_usd))
    merge_mjcf(Path(a.mjcf), Path(a.out_mjcf), M, C, lam, q)
    print(f"{NEW}: mass {M:.6f} kg, COM {C}, principal inertia {lam}, axes (wxyz) {q}")
    print(f"wrote {a.out_usd}\nwrote {a.out_mjcf}")


if __name__ == "__main__":
    main()
