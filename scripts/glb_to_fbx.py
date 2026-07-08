import bpy
import sys
import os

# ----------------------------
# CONFIG
# ----------------------------
argv = sys.argv
argv = argv[argv.index("--") + 1:]

if len(argv) != 2:
    print("Usage: blender --background --python glb_to_fbx.py -- input.glb output.fbx")
    sys.exit(1)

input_glb = os.path.abspath(argv[0])
output_fbx = os.path.abspath(argv[1])

# ----------------------------
# CLEAN SCENE
# ----------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)

# ----------------------------
# IMPORT GLB
# ----------------------------
bpy.ops.import_scene.gltf(filepath=input_glb)

# ----------------------------
# OPTIONAL: SELECT ALL OBJECTS
# ----------------------------
bpy.ops.object.select_all(action='SELECT')

# ----------------------------
# EXPORT FBX
# ----------------------------
bpy.ops.export_scene.fbx(
    filepath=output_fbx,
    use_selection=True,
    apply_unit_scale=True,
    bake_space_transform=False,
    object_types={'ARMATURE', 'MESH'},
    bake_anim=True,
    bake_anim_use_all_actions=False,
    bake_anim_force_startend_keying=True,
    bake_anim_simplify_factor=0.0,
    add_leaf_bones=False,
    primary_bone_axis='Y',
    secondary_bone_axis='X',
    armature_nodetype='NULL',
    use_armature_deform_only=True,
    path_mode='AUTO'
)

print(f"✅ Successfully converted:\n{input_glb} → {output_fbx}")


# /Applications/Blender.app/Contents/MacOS/Blender \
#   --background \
#   --python glb_to_fbx.py \
#   -- m1_brown_4.glb m1_brown_6.fbx
