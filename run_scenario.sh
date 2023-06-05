python3 main.py --scenario track \
--yolo-model weights/yolov8n.pt \
--tracking-method bytetrack \
--speed-method transform_3d \
--source test_data/test1.mp4 \
--classes 0 1 2 3 5 6 7 \
--name test1_output \
--save \
--save-txt \