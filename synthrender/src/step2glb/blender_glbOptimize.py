import blenderproc 
import bpy, sys

if __name__ == "__main__":

    if len(sys.argv) != 3:
        exit()

    infile = sys.argv[-2]
    outfile = sys.argv[-1]

    # 1) clear default scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2) import
    bpy.ops.import_scene.gltf(filepath=infile)

    # 3) export (with Draco compression)
    bpy.ops.export_scene    
    bpy.ops.export_scene.gltf(
        filepath=outfile,
        export_format='GLB',
        export_draco_mesh_compression_enable=True,
        export_draco_mesh_compression_level=6
    )
