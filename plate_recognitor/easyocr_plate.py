import easyocr
import cv2
reader = easyocr.Reader(['en'], gpu=False)


def easyocr_plate_rec(cropped_obj_img, plate_box):
    x1, y1, x2, y2 = [int(i) for i in plate_box]
    plate_img = cropped_obj_img[y1:y2,x1:x2]
    print(plate_img.shape)
    gray = cv2.cvtColor(plate_img , cv2.COLOR_RGB2GRAY)
    #gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    result = reader.readtext(gray)
    text = ""

    for res in result:
        if len(result) == 1:
            text = res[1]
        if len(result) >1 and len(res[1])>6 and res[2]> 0.2:
            text = res[1]
    #     text += res[1] + " "
    
    return str(text)