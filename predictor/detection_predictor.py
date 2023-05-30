# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from .speed_estimator import twoline_speed, twopoint_speed
from .draw_boxes import *



class DetectionPredictor_V2(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_width, example=str(self.model.names))

    def postprocess(self, preds, img, orig_imgs):
    
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
        
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def write_results_v2(self, idx, tracker_outputs, results, batch, speed_method):
      
        p, im, im0 = batch
        # all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
 
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)


        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        outputs = tracker_outputs[idx]
        result = results[idx]
 
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -3]
            object_id = outputs[:, -1]
            
            img = draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)

        if self.args.save or self.args.show:  # Add bbox to image
            self.plotted_img = img
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops', file_name=self.data_path.stem)


        return log_string

