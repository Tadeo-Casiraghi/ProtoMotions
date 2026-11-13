"""Build smpl_humanoid10bis_tissue.{usda,xml} from smpl_humanoid10bis_merged: residual-limb soft
tissue as joints instead of compliant contacts.

Why: the skin <-> socket interface was 8 compliant (spring) contacts between links of the same
articulation. PhysX solves those iteratively and converges very slowly on this self-contact loop:
at 360 Hz / 32 iterations the socket pistoned ~5x more than the converged answer (see
scripts/socket_bench.py). Joint drives are solved implicitly (exactly), and rigid contacts converge
quickly, so the tissue compliance is moved into joints:

  * each skin box hangs from R_Knee on its own prismatic joint, "<skin box>_tissue", along the box
    local z, which is exactly the normal of the socket plate it touches. Its drive (set in the robot
    config) is the tissue spring/damper of that region: posterior / medial / anterior / lateral
    keep the stiffness ratios of the skin materials;
  * skin <-> socket contact becomes rigid and frictionless (the skin materials lose their compliant
    stiffness/damping and friction); shear, suction and liner effects live in the suspension joint
    drives (suspension_slide / _x / _y / _z);
  * the socket plates are extended down the limb axis (distal end) so that each side is one
    continuous surface over the piston range: with separate plates, skin box corners caught on the
    plate edges / the gap between the top and bottom plate and locked the socket (a modelling
    artifact). The top and bottom plate of a side are coplanar, so the bottom skin boxes ride on
    the distal extension of the top plate when the socket hangs;
  * the top plates are also extended at the proximal end: the suspension rests at +22 mm (socket
    hanging away from the limb) and reaches ~25-29 mm in swing, which moved the top skin boxes
    (1 cm, 12.5 mm below the original proximal edge at slide 0) completely off the top of their
    plates; when the socket came back up they landed on the plate edge. Medial / anterior /
    lateral: +45 mm, the whole suspension_slide range (+50 mm). Posterior: +20 mm only (fully on
    up to ~28 mm of slide), like the lower posterior trim line of a real socket: a longer posterior
    wall hits the thigh in swing (at 22 mm of slide: 81 deg knee flexion without the extension,
    64.5 deg with 20 mm, 42 deg with 45 mm; the other walls never reach the thigh);
  * the tibia capsule (R_Knee) no longer collides with socket_assembly: no rigid bone-on-pylon end stop;
  * the pylon is shortened (osl_ankle_assembly and everything below it moves up the limb axis), the
    way a prosthetist sets the length: with the softer, more realistic socket the prosthetic leg was
    ~22 mm too long when standing with half the body weight on each leg (socket_bench.py --protocol stand).

Usage (repo root, env_isaaclab):
    python protomotions/data/assets/scripts/make_tissue_socket.py
"""

import argparse
import re

import numpy as np
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ASSETS = Path(__file__).resolve().parents[1]
ROOT = "/smpl_humanoid10"
SKIN = [f"skin_box_{s}_{l}" for l in ("top", "bottom") for s in ("posterior", "medial", "anterior", "lateral")]
SOCKET_PLATES = [f"socket_box_{s}_{l}" for l in ("top", "bottom") for s in ("posterior", "medial", "anterior", "lateral")]
SKIN_MATERIALS = ["Anterior", "Lateral", "Posterior", "Medial"]


def _plate_down_sign(R_cols_y):
    """+1 if the plate local +y points down the limb (socket frame +y), else -1."""
    return 1.0 if R_cols_y[1] >= 0 else -1.0


PYLON_BODIES = ["osl_ankle_assembly", "R_Ankle_Motor", "R_Ankle_foot", "R_Ankle_toes", "R_Ankle"]


def _extensions(plate, distal, top_proximal):
    """(added length, centre shift down the limb) of a plate, in m. top_proximal: {side: m}."""
    side, level = plate.split("_")[2:]
    prox = top_proximal.get(side, 0.0) if level == "top" else 0.0
    return distal + prox, (distal - prox) / 2


def build_usd(src: Path, dst: Path, tissue_range, plate_extension, top_proximal_extension=None, pylon_shortening=0.0,
              bottom_visual_distal_extension=None):
    stage = Usd.Stage.Open(str(src))
    layer = stage.GetRootLayer()
    bodies, joints = f"{ROOT}/bodies", f"{ROOT}/joints"
    new_joints = {}

    # 1. fixed R_Knee -> skin box joints become prismatic tissue joints along the box local z
    for b in SKIN:
        old_path = Sdf.Path(f"{joints}/{b}")
        old = UsdPhysics.Joint(stage.GetPrimAtPath(old_path))
        b0, b1 = old.GetBody0Rel().GetTargets(), old.GetBody1Rel().GetTargets()
        lp0, lr0 = old.GetLocalPos0Attr().Get(), old.GetLocalRot0Attr().Get()
        lp1, lr1 = old.GetLocalPos1Attr().Get(), old.GetLocalRot1Attr().Get()
        assert lp1 == Gf.Vec3f(0, 0, 0) and lr1 == Gf.Quatf(1, 0, 0, 0), f"{b}: joint frame is not the box frame"
        stage.RemovePrim(old_path)
        path = Sdf.Path(f"{joints}/{b}_tissue")
        j = UsdPhysics.PrismaticJoint.Define(stage, path)
        j.CreateBody0Rel().SetTargets(b0)
        j.CreateBody1Rel().SetTargets(b1)
        j.CreateLocalPos0Attr(lp0)
        j.CreateLocalRot0Attr(lr0)
        j.CreateLocalPos1Attr(lp1)
        j.CreateLocalRot1Attr(lr1)
        j.CreateAxisAttr("Z")  # + = the box pushed into the limb (tissue compression)
        j.CreateLowerLimitAttr(tissue_range[0])
        j.CreateUpperLimitAttr(tissue_range[1])
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")  # gains are set by the robot config actuators
        d.CreateTypeAttr("force")
        d.CreateTargetPositionAttr(0.0)
        d.CreateStiffnessAttr(1.0)
        d.CreateDampingAttr(1.0)
        d.CreateMaxForceAttr(1.0e6)
        new_joints[old_path] = path

    # robot schema joint list: point at the renamed joints
    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            t = rel.GetTargets()
            if any(x in new_joints for x in t):
                rel.SetTargets([new_joints.get(x, x) for x in t])

    # 2. skin materials: rigid, frictionless contact
    for m in SKIN_MATERIALS:
        prim = stage.GetPrimAtPath(f"{ROOT}/Looks/{m}")
        for a in ("physxMaterial:compliantContactStiffness", "physxMaterial:compliantContactDamping"):
            if prim.GetAttribute(a):
                prim.RemoveProperty(a)
        prim.GetAttribute("physics:staticFriction").Set(0.0)
        prim.GetAttribute("physics:dynamicFriction").Set(0.0)
        prim.GetAttribute("physics:restitution").Set(0.0)

    # 3. socket plates: extend the distal end (down the limb = socket frame +y) by plate_extension,
    #    and the proximal end of the top plates by top_proximal_extension[side]. The bottom skin boxes
    #    never reach the distal part of the bottom plates, so their visuals can be extended less
    #    (bottom_visual_distal_extension; visuals have no collision and the link mass is explicit)
    sa = f"{bodies}/socket_assembly"
    for p in SOCKET_PLATES:
        for kind in ("collisions", "visuals"):
            distal = plate_extension
            if kind == "visuals" and p.endswith("_bottom") and bottom_visual_distal_extension is not None:
                distal = bottom_visual_distal_extension
            length, shift = _extensions(p, distal, top_proximal_extension or {})
            holder = stage.GetPrimAtPath(f"{sa}/{p}_{kind}")
            q = holder.GetAttribute("xformOp:orient").Get()  # plate orientation in the socket frame
            y_axis = Gf.Rotation(q).TransformDir(Gf.Vec3d(0, 1, 0))
            sgn = _plate_down_sign(y_axis)
            for g in holder.GetChildren():
                sc = g.GetAttribute("xformOp:scale")
                v = sc.Get()
                sc.Set(Gf.Vec3d(v[0], v[1] + length / 2, v[2]))  # half extent
                tr = g.GetAttribute("xformOp:translate")
                t = tr.Get()
                tr.Set(Gf.Vec3d(t[0], t[1] + sgn * shift, t[2]))

    # 4. pylon length: move the ankle subtree up the limb (prosthetic_assembly2 frame -y) by pylon_shortening
    if pylon_shortening:
        a2 = stage.GetPrimAtPath(f"{bodies}/prosthetic_assembly2")
        up = Gf.Rotation(a2.GetAttribute("xformOp:orient").Get()).TransformDir(Gf.Vec3d(0, -pylon_shortening, 0))
        for b in PYLON_BODIES:
            tr = stage.GetPrimAtPath(f"{bodies}/{b}").GetAttribute("xformOp:translate")
            tr.Set(tr.Get() + up)
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                j = UsdPhysics.Joint(prim)
                if j.GetBody1Rel().GetTargets() == [Sdf.Path(f"{bodies}/osl_ankle_assembly")]:
                    lp = j.GetLocalPos0Attr().Get()
                    j.GetLocalPos0Attr().Set(Gf.Vec3f(lp[0], lp[1] - pylon_shortening, lp[2]))

    # 5. no bone <-> socket collision (no rigid end stop)
    knee = stage.GetPrimAtPath(f"{bodies}/R_Knee")
    UsdPhysics.FilteredPairsAPI.Apply(knee).CreateFilteredPairsRel().AddTarget(Sdf.Path(sa))

    layer.Export(str(dst))


def build_mjcf(src: Path, dst: Path, tissue_range, plate_extension, top_proximal_extension=None, pylon_shortening=0.0):
    xml = src.read_text()
    if pylon_shortening:  # osl_ankle_assembly offset in the prosthetic_assembly2 frame (+y = down the limb)
        m = re.search(r'(<body name="osl_ankle_assembly" pos=")([^"]+)(")', xml)
        x, y, z = (float(v) for v in m.group(2).split())
        xml = xml[: m.start(2)] + f"{x:.8f} {y - pylon_shortening:.8f} {z:.8f}" + xml[m.end(2):]
    lo, hi = tissue_range
    for b in SKIN:  # slide joint along the box local z, first child of the skin body
        pat = re.compile(rf'(<body name="{b}"[^>]*>\s*\n)(\s*)')
        m = pat.search(xml)
        assert m, b
        joint = f'<joint name="{b}_tissue" type="slide" axis="0 0 1" range="{lo} {hi}"/>\n{m.group(2)}'
        xml = xml[: m.end()] + joint + xml[m.end():]
    for p in SOCKET_PLATES:  # extend the ends: half-length + length/2, centre moved by shift down the limb
        length, shift = _extensions(p, plate_extension, top_proximal_extension or {})
        pat = re.compile(rf'<geom name="{p}_geom" pos="([^"]+)" quat="([^"]+)"([^>]*)size="([\d.]+) ([\d.]+) ([\d.]+)"')
        m = pat.search(xml)
        assert m, p
        pos = np.array([float(x) for x in m.group(1).split()])
        w, x, y, z = (float(v) for v in m.group(2).split())
        y_axis = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])  # R[:, 1]
        y_axis /= np.linalg.norm(y_axis)
        sgn = _plate_down_sign(y_axis)
        new_pos = pos + y_axis * sgn * shift
        half = float(m.group(5)) + length / 2
        new = (f'<geom name="{p}_geom" pos="{" ".join(f"{v:.6f}" for v in new_pos)}" quat="{m.group(2)}"{m.group(3)}'
               f'size="{m.group(4)} {half:.6f} {m.group(6)}"')
        xml = xml[: m.start()] + new + xml[m.end():]
    xml = xml.replace("  </contact>", '    <exclude body1="R_Knee" body2="socket_assembly"/>\n  </contact>', 1)
    dst.write_text(xml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", default=str(ASSETS / "usd/smpl_humanoid10bis_merged.usda"))
    ap.add_argument("--mjcf", default=str(ASSETS / "mjcf/smpl_humanoid10bis_merged.xml"))
    ap.add_argument("--out-usd", default=str(ASSETS / "usd/smpl_humanoid10bis_tissue.usda"))
    ap.add_argument("--out-mjcf", default=str(ASSETS / "mjcf/smpl_humanoid10bis_tissue.xml"))
    ap.add_argument("--tissue-range", type=float, nargs=2, default=(-0.002, 0.025),
                    help="tissue joint limits (m): small extension, up to 25 mm compression")
    ap.add_argument("--plate-extension", type=float, default=0.05, help="added to the distal end of each plate (m)")
    ap.add_argument("--top-proximal-extension", type=float, default=0.045,
                    help="added to the proximal end of the medial / anterior / lateral top plates (m): keeps "
                         "the top skin boxes on the socket wall over the whole suspension_slide range")
    ap.add_argument("--posterior-proximal-extension", type=float, default=0.02,
                    help="added to the proximal end of the posterior top plate (m); shorter: it hits the thigh")
    ap.add_argument("--pylon-shortening", type=float, default=0.022,
                    help="shorten the pylon by this much (m) for standing leg-length parity")
    ap.add_argument("--bottom-visual-distal-extension", type=float, default=0.0,
                    help="distal extension of the bottom plate VISUALS only (m; collisions use --plate-extension): "
                         "the bottom skin boxes stay >= 35 mm proximal of the bottom plate centre down to -10 mm "
                         "of slide, so the extended part is never touched and only made the socket look long")
    a = ap.parse_args()
    top_prox = {"posterior": a.posterior_proximal_extension, "medial": a.top_proximal_extension,
                "anterior": a.top_proximal_extension, "lateral": a.top_proximal_extension}
    build_usd(Path(a.usd), Path(a.out_usd), a.tissue_range, a.plate_extension, top_prox, a.pylon_shortening,
              a.bottom_visual_distal_extension)
    build_mjcf(Path(a.mjcf), Path(a.out_mjcf), a.tissue_range, a.plate_extension, top_prox, a.pylon_shortening)
    print(f"wrote {a.out_usd}\nwrote {a.out_mjcf}")


if __name__ == "__main__":
    main()
