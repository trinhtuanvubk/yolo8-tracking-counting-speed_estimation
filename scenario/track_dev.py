# https://github.com/ultralytics/ultralytics/issues/1429#issuecomment-1519239409

from pathlib import Path
import torch
import argparse
import numpy as np
import cv2

from trackers import create_tracker
# import trackers
from predictor.detection_predictor import DetectionPredictor_V2


from ultralytics.yolo.engine.model import YOLO, TASK_MAP
from ultralytics.yolo.utils import LOGGER, SETTINGS, colorstr, ops, is_git_dir, IterableSimpleNamespace
from ultralytics.yolo.utils.checks import check_imgsz, print_args
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.engine.results import Boxes
from ultralytics.yolo.data.utils import VID_FORMATS

from .multi_yolo_backend import MultiYolo
# from super_gradients.training import models
# from super_gradients.common.object_names import Models
# from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model


WEIGHTS = Path(SETTINGS['weights_dir'])
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root dir
WEIGHTS = ROOT / 'weights'


def tracker_details(tracker_outputs, im0, plate_model, plate_predictor, detail_tracker_outputs):
    h, w, _ = im0.shape
    bbox_xyxy = tracker_outputs[:, :4]
    identities = tracker_outputs[:, -3]
    print(identities)
    object_id = tracker_outputs[:, -1]
    for i, box in enumerate(bbox_xyxy):

        x1, y1, x2, y2 = [int(i) for i in box]
        _h = x2-x1
        _w = y2-y1
        cropped_img = [im0[x1:x2, y1:y2, :]]
        print(cropped_img[0].shape)
        # im = plate_predictor.preprocess(cropped_img)
        preds = plate_model.predict(cropped_img)
        print(preds[0].boxes.data)
        plate_box = preds[0].boxes.data
        if plate_box.shape[0]:
            plate_box = plate_box.squeeze().tolist()
            plate_box = plate_box[:4]
            plate_box[0] = plate_box[0] / _h * h
            plate_box[2] = plate_box[2] / _h * h
            plate_box[1] = plate_box[1] / _w * w
            plate_box[3] = plate_box[3] / _w * w
        else:
            plate_box = None
        detail_tracker_outputs[i]["tracker_id"] = identities[i]
        detail_tracker_outputs[i]["tracker_bboxes"] = box
        detail_tracker_outputs[i]["object_id"] = object_id[i]
        detail_tracker_outputs[i]["plate_box"] = plate_box
 


        # results = plate_predictor.model.postprocess(path, preds, im, im0s, plate_predictor)
        # print(results)

    # return cropped_img


def on_predict_start(predictor):
    predictor.trackers = []
    predictor.tracker_outputs = [None] * predictor.dataset.bs
    predictor.detail_tracker_outputs = [{}] * predictor.dataset.bs
    predictor.args.tracking_config = \
        Path('trackers') /\
        predictor.args.tracking_method /\
        'configs' /\
        (predictor.args.tracking_method + '.yaml')
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.args.tracking_method,
            predictor.args.tracking_config,
            predictor.args.reid_model,
            predictor.args.device,
            predictor.args.half,
            
        )
        predictor.trackers.append(tracker)
        # if hasattr(predictor.trackers[i], 'model'):
        #     if hasattr(predictor.trackers[i].model, 'warmup'):
        #         predictor.trackers[i].model.warmup()
                
                
def write_MOT_results(txt_path, results, frame_idx, i):
    nr_dets = len(results.boxes)
    frame_idx = torch.full((1, 1), frame_idx + 1)
    frame_idx = frame_idx.repeat(nr_dets, 1)
    dont_care = torch.full((nr_dets, 3), -1)
    i = torch.full((nr_dets, 1), i)
    mot = torch.cat([
        frame_idx,
        results.boxes.id.unsqueeze(1).to('cpu'),
        ops.xyxy2ltwh(results.boxes.xyxy).to('cpu'),
        dont_care,
        i
    ], dim=1)

    with open(str(txt_path) + '.txt', 'ab') as f:  # append binary mode
        np.savetxt(f, mot.numpy(), fmt='%d')  # save as ints instead of scientific notation


@torch.no_grad()
def run(args):
    # ----------------------------
    plate_model = YOLO("weights/yolov8n_plate_dec.pt")
    # print(model)
    overrides = plate_model.overrides.copy()
    plate_model.plate_predictor = TASK_MAP[plate_model.task][3](overrides=overrides, _callbacks=plate_model.callbacks)
    plate_predictor = DetectionPredictor_V2()
    # combine default plate_predictor args with custom, preferring custom
    combined_args = {**plate_predictor.args.__dict__, **args}
    # overwrite default args
    plate_predictor.args = IterableSimpleNamespace(**combined_args)

    plate_predictor.write_MOT_results = write_MOT_results

    if not plate_predictor.model:
        plate_predictor.setup_model(model=plate_model.model, verbose=False)
    # ----------------------------

    model = YOLO(args['yolo_model'] if 'v8' in str(args['yolo_model']) else 'yolov8n')
    # print(model)
    overrides = model.overrides.copy()
    model.predictor = TASK_MAP[model.task][3](overrides=overrides, _callbacks=model.callbacks)

    predictor = DetectionPredictor_V2()

    
    # combine default predictor args with custom, preferring custom
    combined_args = {**predictor.args.__dict__, **args}
    # overwrite default args
    predictor.args = IterableSimpleNamespace(**combined_args)

    predictor.write_MOT_results = write_MOT_results

    if not predictor.model:
        predictor.setup_model(model=model.model, verbose=False)

    predictor.setup_source(predictor.args.source)
    
    predictor.args.imgsz = check_imgsz(predictor.args.imgsz, stride=model.model.stride, min_dim=2)  # check image size
    predictor.save_dir = increment_path(Path(predictor.args.project) / predictor.args.name, exist_ok=predictor.args.exist_ok)
    
    # Check if save_dir/ label file exists
    if predictor.args.save or predictor.args.save_txt:
        (predictor.save_dir / 'labels' if predictor.args.save_txt else predictor.save_dir).mkdir(parents=True, exist_ok=True)
    # Warmup model
    if not predictor.done_warmup:
        predictor.model.warmup(imgsz=(1 if predictor.model.pt or predictor.model.triton else predictor.dataset.bs, 3, *predictor.imgsz))
        predictor.done_warmup = True
    predictor.seen, predictor.windows, predictor.batch, predictor.profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile())
    predictor.add_callback('on_predict_start', on_predict_start)
    predictor.run_callbacks('on_predict_start')

    model = MultiYolo(
        model=predictor.model if 'v8' in str(args['yolo_model']) else args['yolo_model'],
        device=predictor.device,
        args=predictor.args
    )
    for frame_idx, batch in enumerate(predictor.dataset):
        predictor.run_callbacks('on_predict_batch_start')
        predictor.batch = batch
        path, im0s, vid_cap, s = batch
        visualize = increment_path(save_dir / Path(path[0]).stem, exist_ok=True, mkdir=True) if predictor.args.visualize and (not predictor.dataset.source_type.tensor) else False

        n = len(im0s)
        predictor.results = [None] * n

        # Preprocess
        with predictor.profilers[0]:
            im = predictor.preprocess(im0s)
            print(len(im0s))
            print("im ori shape:{}".format(im0s[0].shape))
            # print("im shape: {}".format(im.shape))

        # Inference
        with predictor.profilers[1]:
            preds = model(im, im0s)
            # print(preds)
        # Postprocess moved to MultiYolo
        with predictor.profilers[2]:
            predictor.results = model.postprocess(path, preds, im, im0s, predictor)
            # print(predictor.results)
        predictor.run_callbacks('on_predict_postprocess_end')
        
        # Visualize, save, write results
        n = len(im0s)
        for i in range(n):
            
            if predictor.dataset.source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                continue
            p, im0 = path[i], im0s[i].copy()
            p = Path(p)
            
            with predictor.profilers[3]:
                # get raw bboxes tensor
                dets = predictor.results[i].boxes.data
                print("-----------------------------")
                print("box:{}".format(dets))
                # get predictions
                predictor.tracker_outputs[i] = predictor.trackers[i].update(dets.cpu().detach(), im0)
                print(predictor.tracker_outputs[i])
               
                tracker_details(predictor.tracker_outputs[i], im0, plate_model, plate_predictor, predictor.detail_tracker_outputs)
                # print("plate: {}".format(plate_boxes))
                print(predictor.detail_tracker_outputs)
                print("------------------------------")
            predictor.results[i].speed = {
                'preprocess': predictor.profilers[0].dt * 1E3 / n,
                'inference': predictor.profilers[1].dt * 1E3 / n,
                'postprocess': predictor.profilers[2].dt * 1E3 / n,
                'tracking': predictor.profilers[3].dt * 1E3 / n
            
            }
            # filter boxes masks and pose results by tracking results
            model.filter_results(i, predictor)
            # overwrite bbox results with tracker predictions
            model.overwrite_results(i, im0.shape[:2], predictor)
            # write inference results to a file or directory   
            if predictor.args.verbose or predictor.args.save or predictor.args.save_txt or predictor.args.show:
                if predictor.args.only_track:
                    s+= predictor.write_results(i, predictor.results, (p, im, im0))
                else:
                    s += predictor.write_results_v2(i, predictor.tracker_outputs, predictor.results, (p, im, im0))
                
                predictor.txt_path = Path(predictor.txt_path)
                
                # write MOT specific results
                if predictor.args.source.endswith(VID_FORMATS):
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.stem
                else:
                    # append folder name containing current img
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.parent.name
                    
                if predictor.tracker_outputs[i].size != 0:
                    # print(predictor.tracker_outputs[i])
                    write_MOT_results(
                        predictor.MOT_txt_path,
                        predictor.results[i],
                        frame_idx,
                        i,
                    )
            # display an image in a window using OpenCV imshow()
            if predictor.args.show and predictor.plotted_img is not None:
                predictor.show(p)

            # save video predictions
            if predictor.args.save and predictor.plotted_img is not None:
                predictor.save_preds(vid_cap, i, str(predictor.save_dir / p.name))

        predictor.run_callbacks('on_predict_batch_end')
        # print time (inference-only)
        if predictor.args.verbose:
            LOGGER.info(f'{s}YOLO {predictor.profilers[1].dt * 1E3:.1f}ms, TRACKING {predictor.profilers[3].dt * 1E3:.1f}ms')

    # Release assets
    if isinstance(predictor.vid_writer[-1], cv2.VideoWriter):
        predictor.vid_writer[-1].release()  # release final video writer

    # Print results
    if predictor.args.verbose and predictor.seen:
        t = tuple(x.t / predictor.seen * 1E3 for x in predictor.profilers)  # speeds per image
        LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess, %.1fms tracking per image at shape '
                    f'{(1, 3, *predictor.args.imgsz)}' % t)
    if predictor.args.save or predictor.args.save_txt or predictor.args.save_crop:
        nl = len(list(predictor.save_dir.glob('labels/*.txt')))  # number of labels
        s = f"\n{nl} label{'s' * (nl > 1)} saved to {predictor.save_dir / 'labels'}" if predictor.args.save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', predictor.save_dir)}{s}")

    predictor.run_callbacks('on_predict_end')
    return predictor.save_dir
    

def track(args):
    save_dir = run(vars(args))
    # if args['multi-tasks']:
    #     args['previous_save_dir'] = save_dir

