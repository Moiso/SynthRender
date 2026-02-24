import blenderproc as bproc
from blenderproc.python.writer import CocoWriterUtility

import shutil
import bpy

from synthrender.utils.bproc_utils import custom_blenderproc

# Monkey-patch:
CocoWriterUtility._CocoWriterUtility.binary_mask_to_polygon = custom_blenderproc.custom_binary_mask_to_polygon
bproc.writer.write_coco_annotations = custom_blenderproc.modified_write_coco_annotations

class Coco_Annotator:

    def __init__(self, output_dir:str):
        self.output_path = output_dir
    
        shutil.rmtree(output_dir, ignore_errors=True) # Removing old files:

    def annotate_data(self, batch, hdf5):
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = batch

        bproc.writer.write_coco_annotations(
            output_dir=self.output_path,
            instance_segmaps=hdf5["instance_segmaps"],
            instance_attribute_maps=hdf5["instance_attribute_maps"],
            colors=hdf5["colors"],
            color_file_format="PNG",
            indent=4,
            append_to_existing_output=True,
            mask_encoding_format="polygon"
        )