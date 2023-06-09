python3 main.py --scenario track_plate \
--multi-tasks \
--yolo-model weights/yolov8n.pt \
--tracking-method bytetrack \
--speed-method transform_3d \
--source test_data/plate_test4.mp4 \
--classes 0 1 2 3 5 6 7 \
--name plate_test4 \
--save \
--save-txt \
# --only-track
