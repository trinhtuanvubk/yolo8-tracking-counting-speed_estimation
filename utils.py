import argparse
import torch
from pathlib import Path




def get_args():
    # create parser args
    parser = argparse.ArgumentParser()

    # senario
    parser.add_argument('--scenario', type=str, default='main')
    parser.add_argument('--yolo-model', type=Path, default='weights/yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--obj-yolo-model', type=Path, default='weights/yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--plate-yolo-model', type=Path, default='weights/yolov8n_plate_dec.pt', help='model.pt path(s)')
    parser.add_argument('--reid-model', type=Path, default='weights/mobilenetv2_x1_4_dukemtmcreid.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true', help='display tracking video results')
    parser.add_argument('--save', action='store_true', help='save video tracking results')
    # # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default= './runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exists-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--speed-method', type=str, default='3dtransform', help='speed estimation method')
    parser.add_argument('--save-txt', action='store_true', help='save tracking results in a txt file')

    parser.add_argument('--only-track', action='store_true', help='only save tracking results')
    parser.add_argument('--multi-tasks', action='store_true', help="multi models for multi tasks")
    # val 
    # parser.add_argument('--yolo-model', type=str, default='weights/yolov8n.pt', help='model.pt path(s)')
    # parser.add_argument('--reid-model', type=str, default='weights/mobilenetv2_x1_4_dukemtmcreid.pt')
    # parser.add_argument('--tracking-method', type=str, default='deepocsort', help='strongsort, ocsort')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--project', default='./runs/val', help='save results to project/name')
    # parser.add_argument('--exists-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--benchmark', type=str, default='MOT17-mini', help='MOT16, MOT17, MOT20')
    # parser.add_argument('--split', type=str, default='train', help='existing project/name ok, do not increment')
    # parser.add_argument('--eval-existing', type=str, default='', help='evaluate existing tracker results under mot_callenge/MOTXX-YY/...')
    # parser.add_argument('--conf', type=float, default=0.45, help='confidence threshold')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--processes-per-device', type=int, default=2,
    #                     help='how many subprocesses can be invoked per GPU (to manage memory consumption)')

    # evolve
    # parser.add_argument('--yolo-model', type=str, default='weights/yolov8n.pt', help='model.pt path(s)')
    # parser.add_argument('--reid-model', type=str, default='weights/lmbn_n_cuhk03_d.pt')
    # parser.add_argument('--tracking-method', type=str, default='deepocsort', help='strongsort, ocsort')
    # parser.add_argument('--tracking-config', type=Path, default=None)
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--project', default='./runs/evolve', help='save results to project/name')
    # parser.add_argument('--exists-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--benchmark', type=str,  default='MOT17', help='MOT16, MOT17, MOT20')
    # parser.add_argument('--split', type=str,  default='train', help='existing project/name ok, do not increment')
    # parser.add_argument('--eval-existing', type=str, default='', help='evaluate existing tracker results under mot_callenge/MOTXX-YY/...')
    # parser.add_argument('--conf', type=float, default=0.45, help='confidence threshold')
    # parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--n-trials', type=int, default=10, help='nr of trials for evolution')
    # parser.add_argument('--resume', action='store_true', help='resume hparam search')
    # parser.add_argument('--processes-per-device', type=int, default=2, help='how many subprocesses can be invoked per GPU (to manage memory consumption)')
    # parser.add_argument('--objectives', type=str, default='HOTA,MOTA,IDF1', help='set of objective metrics: HOTA,MOTA,IDF1')
    
    args = parser.parse_args()
    return args

def print_args(args):
    pass