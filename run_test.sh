python track.py --yolo-model weights/yolov8n.pt \
--tracking-method bytetrack \
--name test1_line \
--source test_data/test1.mp4 \
--classes 0 1 2 3 5 6 7 \
--save \


# python3 track.py --yolo-model weights/yolov8n.pt \
# --reid-model weights/osnet_x0_25_msmt17.pt \
# --source test_images/out.avi \
# --save

# python track.py --yolo-model weights/yolov8n.pt \
# --tracking-method bytetrack \
# --source https://www.youtube.com/watch?v=bHWgc5MPnPA \
# --save


# best: seg + deepocsort


# python3 track.py --source road_trafifc.mp4

