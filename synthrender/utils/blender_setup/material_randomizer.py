import os, re, random, bpy, hashlib
from collections.abc import Iterable
# --- simple helpers ----------------------------------------------------------
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".hdr", ".bmp", ".webp")

_NAME_PAT = {
    "basecolor": re.compile(r"(base(color)?|albedo|diffuse|col)(?!.*(rough|metal|normal|ao|height|disp))", re.I),
    "roughness": re.compile(r"(rough|roughness)", re.I),
    "metallic":  re.compile(r"(metal|metallic)", re.I),
    "normal":    re.compile(r"(normal|nrm)(?!.*(height|disp))", re.I),
    "normal_dx": re.compile(r"(normal.*(dx|directx))", re.I),
    "normal_gl": re.compile(r"(normal.*(gl|opengl))", re.I),
    "ao":        re.compile(r"(ao|ambientocclusion)", re.I),
    "height":    re.compile(r"(height|disp|displace)", re.I),
    "orm":       re.compile(r"(occlusion.*rough.*metal|orm)", re.I),
    "mr":        re.compile(r"(metal.*rough|mr)", re.I),
}

# ---------- resolvers ----------
def _to_bpy_objects(items):
    if items is None:
        return []
    if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
        items = [items]
    out = []
    for it in items:
        if isinstance(it, bpy.types.Object):
            out.append(it); continue
        ob = None
        if hasattr(it, "blender_obj") and isinstance(it.blender_obj, bpy.types.Object):
            ob = it.blender_obj
        elif hasattr(it, "get_blender_obj"):
            try: ob = it.get_blender_obj()
            except Exception: ob = None
        elif hasattr(it, "get_name"):
            ob = bpy.data.objects.get(it.get_name())
        elif hasattr(it, "name"):
            ob = bpy.data.objects.get(it.name)
        elif isinstance(it, (str, bytes)):
            ob = bpy.data.objects.get(str(it))
        if isinstance(ob, bpy.types.Object):
            out.append(ob)
    return out

def _mesh_descendants_inclusive(root_obj):
    meshes = set()
    if getattr(root_obj, "type", None) == "MESH":
        meshes.add(root_obj)
    for ch in getattr(root_obj, "children_recursive", []):
        if getattr(ch, "type", None) == "MESH":
            meshes.add(ch)
    return list(meshes)

import bpy

def _ensure_single_user_materials(obj: bpy.types.Object):
    """Duplicate materials so this object has unique, editable copies."""
    if obj.type != "MESH" or not obj.data.materials: return
    for i, mat in enumerate(obj.data.materials):
        if mat and mat.users > 1:
            obj.data.materials[i] = mat.copy()

def _ensure_scalar_animatable(nt, dest_socket, tag: str):
    """
    If dest_socket is unlinked: return dest_socket to animate default_value.
    If linked: insert Math(MULTIPLY) named/tagged and return its factor input.
    """
    if not dest_socket.is_linked:
        return dest_socket  # animate default_value directly

    link = dest_socket.links[0]
    prev_out = link.from_socket
    prev_node = link.from_node

    math = nt.nodes.new('ShaderNodeMath')
    math.operation = 'MULTIPLY'
    math.label = f"AR_{tag}"         # for later debug lookup
    math.name  = f"AR_{tag}"         # unique-ish name
    math.inputs[1].default_value = 1.0  # factor

    nt.links.remove(link)
    nt.links.new(math.inputs[0], prev_out)
    nt.links.new(dest_socket, math.outputs[0])

    math.location = ((prev_node.location.x + dest_socket.node.location.x) / 2,
                     dest_socket.node.location.y - 80)

    return math.inputs[1]  # animate this

def _rand_hsv(seed, ranges):
    rnd = random.Random(seed)
    # Ranges should be wide enough to be visible
    h_off = rnd.uniform(-0.45, 0.45)  # wrap handled by node
    s_mul = rnd.uniform(*ranges.get('sat', (0.6, 1.4)))
    v_mul = rnd.uniform(*ranges.get('val', (0.7, 1.3)))
    return h_off, s_mul, v_mul

def _rand_rgb(seed):
    rnd = random.Random(seed)
    # Bold tints so it pops even on metals
    return (rnd.uniform(0.25, 1.0), rnd.uniform(0.25, 1.0), rnd.uniform(0.25, 1.0), 1.0)

def animate_color_tint(mat, frame, seed, ranges=None):
    """
    Keyframe the tint shim created by ensure_color_tint_shim().
    If HSV exists -> animate Hue/Saturation/Value.
    If MixRGB+RGB exists -> animate RGB and (optionally) Fac for strength.
    """
    if not (mat and mat.use_nodes and mat.node_tree):
        return
    nt = mat.node_tree
    # Try HSV path first
    hsv = next((n for n in nt.nodes if n.type == 'HUE_SAT' and n.name.startswith('AR_HSV_')), None)
    if hsv:
        h_off, s_mul, v_mul = _rand_hsv(seed, ranges or {})
        # Blender’s Hue is 0..1; 0.5 means no shift. Add offset around 0.5.
        hue = (0.5 + h_off) % 1.0
        hsv.inputs['Hue'].default_value        = hue
        hsv.inputs['Saturation'].default_value = s_mul
        hsv.inputs['Value'].default_value      = v_mul
        hsv.inputs['Hue'].keyframe_insert('default_value',        frame=frame)
        hsv.inputs['Saturation'].keyframe_insert('default_value', frame=frame)
        hsv.inputs['Value'].keyframe_insert('default_value',      frame=frame)
        return

    # Else try MixRGB path
    mix = next((n for n in nt.nodes if n.type == 'MIX_RGB' and n.name.startswith('AR_MIX_')), None)
    rgb = next((n for n in nt.nodes if n.type == 'RGB' and n.name.startswith('AR_RGB_')), None)
    if mix and rgb:
        rgba = _rand_rgb(seed)
        rgb.outputs[0].default_value = rgba
        rgb.outputs[0].keyframe_insert('default_value', frame=frame)

        # Optionally vary strength for visibility
        fac = random.Random(seed + 101).uniform(0.55, 0.95)
        mix.inputs['Fac'].default_value = fac
        mix.inputs['Fac'].keyframe_insert('default_value', frame=frame)


def ensure_tint_node(mat):
    """Ensure a node we can animate for visible color change.
    - If Base Color has a texture chain: insert Hue/Saturation node.
    - Else: insert MixRGB (MULTIPLY) + RGB as a tint overlay.
    Returns a dict with handles to animate.
    """
    if not (mat and mat.use_nodes and mat.node_tree):
        return None

    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not bsdf:
        return None

    base = bsdf.inputs.get('Base Color')
    if base and base.is_linked and base.links:
        # Insert HSV between existing source and BSDF.Base Color
        link = base.links[0]
        src_node, src_sock = link.from_node, link.from_socket
        nt.links.remove(link)

        hsv = nt.nodes.get(f'AR_HSV_{mat.name}') or nt.nodes.new('ShaderNodeHueSaturation')
        hsv.name = f'AR_HSV_{mat.name}'
        hsv.label = hsv.name
        hsv.location = (bsdf.location.x - 220, bsdf.location.y + 120)

        nt.links.new(src_sock, hsv.inputs['Color'])
        nt.links.new(hsv.outputs['Color'], base)

        return {'mode':'hsv','bsdf':bsdf,'hsv':hsv}
    else:
        # No texture: use MixRGB MULTIPLY with an RGB tint
        mix = nt.nodes.get(f'AR_MIX_{mat.name}') or nt.nodes.new('ShaderNodeMixRGB')
        mix.name, mix.label = f'AR_MIX_{mat.name}', f'AR_MIX_{mat.name}'
        mix.blend_type = 'MULTIPLY'
        if not mix.inputs['Fac'].is_linked:
            mix.inputs['Fac'].default_value = 0.75

        rgb = nt.nodes.get(f'AR_RGB_{mat.name}') or nt.nodes.new('ShaderNodeRGB')
        rgb.name, rgb.label = f'AR_RGB_{mat.name}', f'AR_RGB_{mat.name}'

        # Original base color as Color1
        orig = nt.nodes.get(f'AR_ORIG_{mat.name}') or nt.nodes.new('ShaderNodeRGB')
        orig.name = f'AR_ORIG_{mat.name}'
        orig.label = orig.name
        if hasattr(bsdf.inputs['Base Color'], 'default_value'):
            orig.outputs[0].default_value = tuple(bsdf.inputs['Base Color'].default_value)

        # Wire: orig -> mix.C1, rgb -> mix.C2, mix -> base
        # (Clear any old link to Base Color)
        for l in list(base.links):
            nt.links.remove(l)
        nt.links.new(orig.outputs[0], mix.inputs['Color1'])
        nt.links.new(rgb.outputs[0],  mix.inputs['Color2'])
        nt.links.new(mix.outputs[0],  base)

        return {'mode':'mix','bsdf':bsdf,'mix':mix,'rgb':rgb}
    

def _find_bsdf(mat):
    if not (mat and mat.use_nodes and mat.node_tree):
        return None
    for n in mat.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            return n
    return None

def _input_src_name(sock):
    if sock is None:
        return "<None>"
    if sock.is_linked and sock.links:
        n = sock.links[0].from_node
        return f"{n.type}:{n.name}"
    return "<unlinked default>"

def debug_mat_chain(mat):
    """Print how Base Color, Roughness, Metallic, Specular are wired and their current values at the *current frame*."""
    bsdf = _find_bsdf(mat)
    if not bsdf:
        print(f"  [!] No Principled BSDF in {mat.name}")
        return

    base = bsdf.inputs.get('Base Color')
    rough = bsdf.inputs.get('Roughness')
    metal = bsdf.inputs.get('Metallic')
    spec  = bsdf.inputs.get('Specular')

    print(f"  mat={mat.name}")
    print(f"    BaseColor <- {_input_src_name(base)} ; val={tuple(getattr(base, 'default_value', [None, None, None, None]))}")
    print(f"    Roughness <- {_input_src_name(rough)} ; val={getattr(rough,'default_value',None)}")
    print(f"    Metallic  <- {_input_src_name(metal)} ; val={getattr(metal,'default_value',None)}")
    print(f"    Specular  <- {_input_src_name(spec)} ; val={getattr(spec,'default_value',None)}")

    # Look for our HSV or MixRGB tint helpers, if any:
    nt = mat.node_tree
    hsvs = [n for n in nt.nodes if n.type == 'HUE_SAT']
    mixes = [n for n in nt.nodes if n.type == 'MIX_RGB' and n.blend_type == 'MULTIPLY']
    if hsvs:
        h = hsvs[0]
        print(f"    HSV node: Hue={h.inputs['Hue'].default_value:.3f}, Sat={h.inputs['Saturation'].default_value:.3f}, Val={h.inputs['Value'].default_value:.3f}")
    if mixes:
        m = mixes[0]
        fac = m.inputs['Fac'].default_value
        print(f"    Mix(MULTIPLY) node: Fac={fac:.3f}")
        if not m.inputs['Color2'].is_linked:
            c2 = tuple(m.inputs['Color2'].default_value)
            print(f"      Color2 (tint)={c2}")

def debug_mat_keyframes(mat):
    """List FCurves on this material's node tree and print keyframe counts + first few key times/values."""
    if not (mat and mat.use_nodes and mat.node_tree and mat.node_tree.animation_data and mat.node_tree.animation_data.action):
        print(f"  mat={mat.name} has no animation_data/action.")
        return
    act = mat.node_tree.animation_data.action
    print(f"  mat={mat.name} FCurves:")
    for fc in act.fcurves:
        # data_path like 'nodes["Principled BSDF"].inputs[9].default_value'
        # or 'nodes["AR_HSV_*"].inputs[1].default_value'
        ks = fc.keyframe_points
        if len(ks) == 0: 
            continue
        samples = ", ".join(f"({int(k.co[0])}:{k.co[1]:.3f})" for k in ks[:5])
        print(f"    {fc.data_path} idx={fc.array_index}  keys={len(ks)}  {samples}{' ...' if len(ks)>5 else ''}")


def force_emissive_tint(mat, color=(5.0, 0.5, 0.5, 1.0), strength=5.0, keyframe_at=None):
    """Inject Emission tint into the BSDF and optionally keyframe it at keyframe_at."""
    bsdf = _find_bsdf(mat)
    if not bsdf: 
        print(f"  [!] No BSDF in {mat.name}")
        return
    # Set Emission and Strength directly
    if 'Emission' in bsdf.inputs and 'Emission Strength' in bsdf.inputs:
        bsdf.inputs['Emission'].default_value = color
        bsdf.inputs['Emission Strength'].default_value = strength
        if keyframe_at is not None:
            bsdf.inputs['Emission'].keyframe_insert('default_value', frame=keyframe_at)
            bsdf.inputs['Emission Strength'].keyframe_insert('default_value', frame=keyframe_at)


def _iter_mesh_children(obj):
    if getattr(obj, "type", None) == "MESH":
        yield obj
    for ch in getattr(obj, "children_recursive", []):
        if getattr(ch, "type", None) == "MESH":
            yield ch

def debug_material_snapshot_for_models(models, frame, max_objs=5, max_mats=3):
    """At a given frame, print what a few objects/mats look like, and list FCurves."""
    bpy.context.scene.frame_set(frame)
    print(f"\n=== DEBUG @ frame {frame} ===")
    objs = models if (isinstance(models, Iterable) and not isinstance(models,(str,bytes))) else [models]
    count = 0
    for parent in objs:
        bpyo = getattr(parent, "blender_obj", None) or getattr(parent, "bpy_obj", None) or parent
        for ob in _iter_mesh_children(bpyo):
            print(f"[obj] {ob.name}")
            if not ob.data or not getattr(ob.data, "materials", None):
                print("  (no materials)")
                continue
            for i, mat in enumerate(ob.data.materials[:max_mats]):
                debug_mat_chain(mat)
                debug_mat_keyframes(mat)
            count += 1
            if count >= max_objs:
                print("(truncated)")
                return
            
def add_material_keyframe(mat, frame, seed=0,
                          ranges=None,
                          key_rough=True, key_metal=True, key_spec=True, key_color=True):
    """
    Set material properties and insert keyframes at 'frame'.
    Extends scene.frame_end if needed (like add_camera_pose).
    """
    if mat is None:
        return

    scn = bpy.context.scene
    if scn.frame_end < frame + 1:
        scn.frame_end = frame + 1

    nt = getattr(mat, 'node_tree', None)
    if not (mat.use_nodes and nt):
        return

    # Ensure a tint path we can animate
    shim = ensure_tint_node(mat)
    bsdf = shim['bsdf'] if shim else next((n for n in nt.nodes if n.type=='BSDF_PRINCIPLED'), None)
    if not bsdf:
        return

    rnd = random.Random(hash(mat.name) ^ int(seed) ^ (frame * 1319))

    # 1) Roughness / Metallic / Specular keyframes
    if key_rough and 'Roughness' in bsdf.inputs:
        rmin, rmax = (0.15, 0.85) if not ranges else ranges.get('roughness', (0.15, 0.85))
        val = rnd.uniform(rmin, rmax)
        bsdf.inputs['Roughness'].default_value = val
        bsdf.inputs['Roughness'].keyframe_insert('default_value', frame=frame)

    if key_metal and 'Metallic' in bsdf.inputs:
        mmin, mmax = (0.0, 0.9) if not ranges else ranges.get('metallic', (0.0, 0.9))
        val = rnd.uniform(mmin, mmax)
        bsdf.inputs['Metallic'].default_value = val
        bsdf.inputs['Metallic'].keyframe_insert('default_value', frame=frame)

    if key_spec and 'Specular' in bsdf.inputs:
        smin, smax = (0.1, 0.5) if not ranges else ranges.get('specular', (0.1, 0.5))
        val = rnd.uniform(smin, smax)
        bsdf.inputs['Specular'].default_value = val
        bsdf.inputs['Specular'].keyframe_insert('default_value', frame=frame)

    # 2) Visible color change (tint) keyframes
    if key_color and shim:
        if shim['mode'] == 'hsv':
            hsv = shim['hsv']
            # Hue is 0..1, 0.5 ~ “no shift”; push broadly so it’s obvious
            hue = (0.5 + rnd.uniform(-0.45, 0.45)) % 1.0
            sat = rnd.uniform(*(ranges.get('sat',(0.7,1.5)) if ranges else (0.7,1.5)))
            val = rnd.uniform(*(ranges.get('val',(0.8,1.4)) if ranges else (0.8,1.4)))
            hsv.inputs['Hue'].default_value        = hue
            hsv.inputs['Saturation'].default_value = sat
            hsv.inputs['Value'].default_value      = val
            hsv.inputs['Hue'].keyframe_insert('default_value', frame=frame)
            hsv.inputs['Saturation'].keyframe_insert('default_value', frame=frame)
            hsv.inputs['Value'].keyframe_insert('default_value', frame=frame)
        else:
            mix, rgb = shim['mix'], shim['rgb']
            rgba = (rnd.uniform(0.2, 1.0), rnd.uniform(0.2, 1.0), rnd.uniform(0.2, 1.0), 1.0)
            rgb.outputs[0].default_value = rgba
            rgb.outputs[0].keyframe_insert('default_value', frame=frame)
            # Also vary the strength to make it pop
            fac = rnd.uniform(0.6, 0.95)
            mix.inputs['Fac'].default_value = fac
            mix.inputs['Fac'].keyframe_insert('default_value', frame=frame)

def add_material_keyframes_for_models(models, frame, seed, ranges=None):
    """Walk parents → mesh descendants and keyframe their materials at 'frame'."""
    from collections.abc import Iterable
    def iter_mesh_children(obj):
        if getattr(obj, "type", None) == "MESH":
            yield obj
        for ch in getattr(obj, "children_recursive", []):
            if getattr(ch, "type", None) == "MESH":
                yield ch

    objs = models if isinstance(models, Iterable) and not isinstance(models, (str, bytes)) else [models]
    for parent in objs:
        # parent could be bproc MeshObject; resolve to bpy object name if needed:
        bpyo = getattr(parent, "blender_obj", None) or getattr(parent, "bpy_obj", None) or parent
        if getattr(bpyo, "type", None) == "MESH":
            targets = [bpyo]
        else:
            targets = list(iter_mesh_children(bpyo))
        for ob in targets:
            for mat in (getattr(ob.data, "materials", None) or []):
                add_material_keyframe(mat, frame=frame, seed=(seed + hash(ob.name) % 10_000), ranges=ranges)


# def ensure_color_tint_shim(mat, strength=0.65):
#     """
#     Guarantee an animatable color/tint path into Base Color.
#     - If Base Color has a texture chain -> insert Hue/Saturation node between tex and BSDF.
#     - If Base Color is a flat color -> insert MixRGB (MULTIPLY) with a ShaderNodeRGB.
#     Returns a dict with handles you can animate later.
#     """
#     if not (mat and mat.use_nodes and mat.node_tree):
#         return None
#     nt = mat.node_tree
#     bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
#     if not bsdf:
#         return None

#     base = bsdf.inputs.get('Base Color')
#     if not base:
#         return None

#     # Case A: There is an upstream color (e.g., texture)
#     if base.is_linked and base.links:
#         from_sock = base.links[0].from_socket
#         from_node = base.links[0].from_node

#         # Insert Hue/Saturation between the existing source and BSDF Base Color
#         hsv = nt.nodes.new('ShaderNodeHueSaturation')
#         hsv.name  = f'AR_HSV_{mat.name}'
#         hsv.label = hsv.name
#         hsv.location = (bsdf.location.x - 220, bsdf.location.y + 120)

#         # Rewire: source -> HSV -> BSDF.BaseColor
#         nt.links.remove(base.links[0])
#         nt.links.new(from_sock, hsv.inputs['Color'])
#         nt.links.new(hsv.outputs['Color'], base)

#         # Wider default range so changes are obvious
#         hsv.inputs['Hue'].default_value        = 0.5   # 0.0..1.0 (0.5 == no shift in Blender terms)
#         hsv.inputs['Saturation'].default_value = 1.0
#         hsv.inputs['Value'].default_value      = 1.0

#         return {'mode': 'hsv', 'mat': mat, 'bsdf': bsdf, 'hsv': hsv}

#     # Case B: No upstream color — flat color. Make a tint overlay that’s easy to animate.
#     rgb = nt.nodes.new('ShaderNodeRGB')
#     rgb.name  = f'AR_RGB_{mat.name}'
#     rgb.label = rgb.name
#     rgb.outputs[0].default_value = (0.9, 0.9, 0.9, 1.0)

#     mix = nt.nodes.new('ShaderNodeMixRGB')
#     mix.name  = f'AR_MIX_{mat.name}'
#     mix.label = mix.name
#     mix.blend_type = 'MULTIPLY'
#     mix.inputs['Fac'].default_value = strength

#     # Wire: (original base color) -> Mix.Color1, RGB -> Mix.Color2, Mix -> BSDF.BaseColor
#     # Grab the current flat color as the "original"
#     original = bsdf.inputs['Base Color'].default_value[:]  # tuple RGBA
#     # Create a Constant node to hold original color (use an RGB node)
#     orig_rgb = nt.nodes.new('ShaderNodeRGB')
#     orig_rgb.name  = f'AR_ORIG_{mat.name}'
#     orig_rgb.label = orig_rgb.name
#     orig_rgb.outputs[0].default_value = original

#     # Connect
#     nt.links.new(orig_rgb.outputs[0], mix.inputs['Color1'])
#     nt.links.new(rgb.outputs[0],      mix.inputs['Color2'])
#     nt.links.new(mix.outputs[0],      base)

#     return {'mode': 'mix', 'mat': mat, 'bsdf': bsdf, 'rgb': rgb, 'mix': mix}


# ---------- deterministic RNG ----------
def _rng(seed, *salts):
    h = hashlib.sha256(str(seed).encode())
    for s in salts: h.update(str(s).encode())
    return random.Random(int(h.hexdigest(), 16) & ((1<<63)-1))

def _read_effective_scalar(bsdf, socket_name):
    """Return (value, source) where value may be None if not readable."""
    if not bsdf: 
        return None, 'no_bsdf'
    sock = bsdf.inputs.get(socket_name)
    if not sock:
        return None, 'no_socket'
    if sock.is_linked:
        n = sock.links[0].from_node if sock.links else None
        # our shim?
        if n and n.type == 'MATH' and ((n.name and n.name.startswith('AR_')) or (n.label and n.label.startswith('AR_'))):
            return n.inputs[1].default_value, 'shim'
        # linked to something else we can’t easily eval
        return None, 'linked_upstream'
    # unlinked -> direct default is the effective value
    return sock.default_value, 'bsdf'

def _fmt_num(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"

def debug_material_snapshot(objs, frame):
    for o in objs:
        if getattr(o, "type", None) != "MESH":
            continue
        mats = getattr(o.data, "materials", []) or []
        for m in mats:
            if not (m and m.use_nodes and m.node_tree):
                continue
            bsdf = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            r, rsrc  = _read_effective_scalar(bsdf, 'Roughness')
            me, msrc = _read_effective_scalar(bsdf, 'Metallic')
            sp, spsrc= _read_effective_scalar(bsdf, 'Specular')
            print(f"[f{frame}] {o.name} R={_fmt_num(r)}({rsrc}) M={_fmt_num(me)}({msrc}) S={_fmt_num(sp)}({spsrc})")


def animate_material_look(mat, obj_name, frame, seed, ranges=None):
    if not mat or not mat.use_nodes or not mat.node_tree: return
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not bsdf: return

    rng = _rng(seed, obj_name, frame)
    ranges = ranges or {}
    rmin, rmax = ranges.get("roughness", (0.1, 0.9))
    mmin, mmax = ranges.get("metallic",  (0.0, 1.0))
    smin, smax = ranges.get("specular",  (0.1, 0.5))

    # Roughness / Metallic / Specular
    for name, lo, hi, tag in (
        ('Roughness', rmin, rmax, 'ROUGHNESS'),
        ('Metallic',  mmin, mmax, 'METALLIC'),
        ('Specular',  smin, smax, 'SPECULAR'),
    ):
        sock = bsdf.inputs.get(name)
        if not sock: continue
        target = _ensure_scalar_animatable(nt, sock, tag)
        val = rng.uniform(lo, hi)
        target.default_value = val
        target.keyframe_insert('default_value', frame=frame)

    # (color Hue/Sat shim as before, optional)
    mat.node_tree.update_tag()



def animate_object_materials(obj: bpy.types.Object, frame: int, seed: int, ranges=None):
    if obj.type != "MESH": return
    _ensure_single_user_materials(obj)  # avoid shared datablock surprises
    for mat in obj.data.materials:
        if mat:
            animate_material_look(mat, obj_name=obj.name, frame=frame, seed=seed, ranges=ranges)


def animate_models_materials(models, frame: int, seed: int, ranges=None):
    """models can be BProc wrappers or bpy objects. Applies to ALL mesh descendants."""
    for parent in _to_bpy_objects(models):
        for child in _mesh_descendants_inclusive(parent):
            animate_object_materials(child, frame=frame, seed=seed, ranges=ranges)
            
#################################################################################################
#
# Taking PBR material from folders
#
#################################################################################################
def _stable_subfolders(path):
    """Sorted list of immediate subfolders (case-insensitive)."""
    subs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    subs.sort(key=lambda s: s.lower())
    return subs

def _rng_from(seed, *salts):
    """Deterministic RNG from seed + any extra salts (object name, frame, etc.)."""
    h = hashlib.sha256()
    h.update(str(seed).encode())
    for s in salts:
        h.update(str(s).encode())
    return random.Random(int(h.hexdigest(), 16) & ((1<<63)-1))

def pick_pbr_folder(lib_dir, seed, object_name=None, frame=None, rng=None):
    """Return a deterministic PBR subfolder based on seed (+ optional salts)."""
    folders = _stable_subfolders(lib_dir)
    if not folders:
        raise RuntimeError(f"No PBR folders in {lib_dir}")
    if rng is None:
        rng = _rng_from(seed, object_name, frame)
    return os.path.join(lib_dir, rng.choice(folders))

def _classify_files(folder: str) -> dict:
    files = [f for f in os.listdir(folder) if f.lower().endswith(_IMG_EXTS)]
    out = {}
    for f in files:
        stem = os.path.splitext(f)[0]
        for k, pat in _NAME_PAT.items():
            if pat.search(stem):
                out[k] = os.path.join(folder, f)
    # fallbacks
    if "basecolor" not in out:
        # choose any image as color if nothing matched
        if files:
            out["basecolor"] = os.path.join(folder, files[0])
    return out

def _tex(material, path, non_color=False):
    img = bpy.data.images.load(os.path.abspath(path))
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = img
    if non_color:
        node.image.colorspace_settings.name = "Non-Color"
    return node


def force_single_material(obj: bpy.types.Object, mat: bpy.types.Material):
    """Remove all material slots on obj, add `mat` as the only slot,
    and set every polygon's material_index = 0."""
    if obj.type != "MESH":
        return

    # make sure the material exists & uses nodes
    if not mat.use_nodes:
        mat.use_nodes = True

    # clear all material slots
    obj.data.materials.clear()
    # add the single material
    obj.data.materials.append(mat)

    # force every face to use slot 0
    for poly in obj.data.polygons:
        poly.material_index = 0


# --- core builder ------------------------------------------------------------
def build_pbr_material_from_folder(folder: str, name_prefix="PBR") -> bpy.types.Material:
    mat = bpy.data.materials.new(name=f"{name_prefix}_{os.path.basename(folder)}")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    # clean, keep Principled + Output
    keep = {"Principled BSDF", "Material Output"}
    for n in [n for n in nodes if n.name not in keep]:
        nodes.remove(n)
    bsdf = nodes.get("Principled BSDF")
    out  = nodes.get("Material Output")

    files = _classify_files(folder)

    # Base Color
    if "basecolor" in files:
        col = _tex(mat, files["basecolor"], non_color=False)
        links.new(bsdf.inputs["Base Color"], col.outputs["Color"])

    # Roughness
    if "roughness" in files:
        rough = _tex(mat, files["roughness"], non_color=True)
        links.new(bsdf.inputs["Roughness"], rough.outputs["Color"])

    # Metallic
    if "metallic" in files:
        met = _tex(mat, files["metallic"], non_color=True)
        links.new(bsdf.inputs["Metallic"], met.outputs["Color"])

    # Normal (DX/GL/Generic)
    if any(k in files for k in ("normal_dx", "normal_gl", "normal")):
        npath = files.get("normal_dx") or files.get("normal_gl") or files.get("normal")
        ntex  = _tex(mat, npath, non_color=True)
        nmap  = nodes.new("ShaderNodeNormalMap")
        # If you *know* it’s DirectX use strength + invert Y, but simplest: allow user to flip later
        links.new(nmap.inputs["Color"], ntex.outputs["Color"])
        links.new(bsdf.inputs["Normal"], nmap.outputs["Normal"])

    # Ambient Occlusion (multiply into base color)
    if "ao" in files and "basecolor" in files:
        ao  = _tex(mat, files["ao"], non_color=True)
        mix = nodes.new("ShaderNodeMixRGB")
        mix.blend_type = 'MULTIPLY'
        mix.inputs["Fac"].default_value = 1.0
        # rewire: BaseColor -> Mix Color1, AO -> Color2
        col = next(n for n in nodes if isinstance(n, bpy.types.ShaderNodeTexImage) and n is not ao)
        links.new(mix.inputs["Color1"], col.outputs["Color"])
        links.new(mix.inputs["Color2"], ao.outputs["Color"])
        links.new(bsdf.inputs["Base Color"], mix.outputs["Color"])

    # Packed maps: ORM (R=AO, G=Rough, B=Metal) or MR (varies by vendor; assume G=Rough, B=Metal)
    if "orm" in files or "mr" in files:
        packed = _tex(mat, files.get("orm", files.get("mr")), non_color=True)
        sep = nodes.new("ShaderNodeSeparateRGB")
        links.new(sep.inputs["Image"], packed.outputs["Color"])
        # default wiring for ORM
        ao_out, rough_out, metal_out = sep.outputs["R"], sep.outputs["G"], sep.outputs["B"]
        if "mr" in files:
            # common MR convention: G=roughness, B=metallic (AO not packed)
            rough_out, metal_out = sep.outputs["G"], sep.outputs["B"]
            ao_out = None
        links.new(bsdf.inputs["Roughness"], rough_out)
        links.new(bsdf.inputs["Metallic"], metal_out)
        if ao_out:
            # multiply AO into base if we already had it wired; else just ignore
            mix = nodes.new("ShaderNodeMixRGB")
            mix.blend_type = 'MULTIPLY'
            mix.inputs["Fac"].default_value = 1.0
            # Feed current base color
            # Create a reroute for clarity
            rer = nodes.new("NodeReroute")
            links.new(rer.inputs[0], bsdf.inputs["Base Color"].links[0].from_socket)
            links.new(mix.inputs["Color1"], rer.outputs[0])
            links.new(mix.inputs["Color2"], ao_out)
            links.new(bsdf.inputs["Base Color"], mix.outputs["Color"])

    # Height/Displacement (simple displacement)
    if "height" in files:
        htex = _tex(mat, files["height"], non_color=True)
        disp = nodes.new("ShaderNodeDisplacement")
        disp.inputs["Scale"].default_value = 0.02  # tweak as needed
        links.new(disp.inputs["Height"], htex.outputs["Color"])
        links.new(out.inputs["Displacement"], disp.outputs["Displacement"])

    return mat

# --- assigner ---------------------------------------------------------------
def assign_material_to_object(obj: bpy.types.Object, mat: bpy.types.Material):
    if obj.type != "MESH":
        return
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

# --- one-call convenience ---------------------------------------------------
def assign_random_pbr_from_library(obj: bpy.types.Object, library_dir: str, seed, frame=None, rng=None):
    """Pick a subfolder under `library_dir`, build a Principled material, assign to `obj`."""
    
    if not os.path.isdir(library_dir):
        print(f"[PBR] Invalid library: {library_dir}")
        return
    folder = pick_pbr_folder(library_dir, seed=seed, object_name=obj.name, frame=frame, rng=rng)


    mat = build_pbr_material_from_folder(folder)
    mat = mat.copy(); mat.name = f"{mat.name}__{obj.name}"  # unique per object (prevents later overwrites)

    force_single_material(obj, mat)                  # wipe slots and apply
    # assign_material_to_object(obj, mat)
    print(f"[PBR] Assigned '{mat.name}' from '{folder}' to '{obj.name}'")
