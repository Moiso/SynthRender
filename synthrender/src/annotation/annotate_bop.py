import blenderproc as bproc

import os
import shutil
import bpy

from synthrender.utils import misc_utils
from synthrender.utils.bproc_utils import bproc_utils
from synthrender.utils import misc_utils

from synthrender.src.simulation import KeyframeGenerator

class Bop_Annotator:

    def __init__(self, output_dir:str):
        self.output_path = output_dir
    
        shutil.rmtree(output_dir, ignore_errors=True) # Removing old files:

    def set_up_scene(self, config, keyframes):
        keyframer = KeyframeGenerator(config)
        keyframer.set_up_keyframes(keyframes)
        print("[#]: Scene loaded!\n")

    def get_target_elements(self):
        all_elements = list(set([*bproc_utils.get_all_parents(), *bproc_utils.get_entities()]))
        target_elements = [element for element in all_elements if element.has_cp('category_id')]

        return target_elements

    def annotate_data(self, target_elements, start, batch, fixed_hdf5):
        bpy.context.scene.frame_start = start
        bpy.context.scene.frame_end = start + batch
        
        bproc.writer.write_bop(
            output_dir=self.output_path,
            target_objects=target_elements, #all_elements,
            depths=fixed_hdf5["depth"],
            colors=fixed_hdf5["colors"],
            calc_mask_info_coco=True,
            append_to_existing_output=True,
            frames_per_chunk=batch,
            annotation_unit='mm'
        )
        train_pbr = os.path.join(self.output_path, "train_pbr")
        if os.path.isdir(train_pbr):
            batches = os.listdir(train_pbr)
            last_id = len(batches) -1

            if os.path.isdir(os.path.join(train_pbr, "{last_id:06d}/scene_gt.json")):
                misc_utils.prettier_json(os.path.join(train_pbr, "{last_id:06d}/scene_gt.json"))
                misc_utils.fix_coco_json(os.path.join(train_pbr, "{last_id:06d}/scene_gt_coco.json"), target_elements)
                misc_utils.prettier_json(os.path.join(train_pbr, "{last_id:06d}/scene_gt_coco.json"))