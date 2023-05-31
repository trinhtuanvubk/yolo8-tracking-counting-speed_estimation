import math
from .transform_3d import * 


def twopoints_speed(Location1, Location2):
    #Euclidean Distance Formula
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    # defining thr pixels per meter
    ppm = 8
    d_meters = d_pixel/ppm
    # fps: 15
    time_constant = 15*3.6
    #distance = speed/time
    speed = d_meters * time_constant
    return int(speed)

def twolines_speed(num_frames, line1, line2):

    ppm = 8 
    d_pixel = line1[0][1] - line2[0][1]
    d_meters = d_pixel/ppm
    time_constant = num_frames/30
    #distance = speed/time
    speed = d_meters / time_constant *3.6

    return int(speed)


def birdeyes_speed(location1, location2):
    #Euclidean Distance Formula
    d_pixel = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # defining thr pixels per meter
    ppm = 8
    d_meters = d_pixel/ppm
    # fps: 15
    time_constant = 1/(30*3.6)
    #distance = speed/time
    speed = d_meters / time_constant
    return int(speed)


def transform_3d_speed(location1, location2):
    height, width = 740, 1280

    # Khởi tạo các tham số
    pi = 3.14
    # Góc quay camera
    p = [-pi / 7.4, 0.11, 0.0]  # [góc quay theo trục X, quay theo trục Y, quay theo trục Z]

    H = 5.0  # Chiều cao của camera so với mặt đất
    f = 0.010  # focal_length của camera
    # Độ rộng 1 pixel
    pu = 0.000025

    # Ma trận hướng của camera, Rx tương ứng là ma trận xoay theo trục X
    Rx = np.array([[1, 0, 0, 0],
                   [0, cos(-p[0]), sin(-p[0]), 0],
                   [0, -sin(-p[0]), cos(-p[0]), 0],
                   [0, 0, 0, 1]])

    Ry = np.array([[cos(-p[1]), 0, -sin(-p[1]), 0],
                   [0, 1, 0, 0],
                   [sin(-p[1]), 0, cos(-p[1]), 0],
                   [0, 0, 0, 1]])

    Rz = np.array([[cos(-p[2]), sin(-p[2]), 0, 0],
                   [-sin(-p[2]), cos(-p[2]), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Ma trận vị trí của camera
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, -H],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    R = Rx @ Ry @ Rz

    # Ma trận chiếu
    K = np.array([[-f, 0, 0],
                  [0, -f, 0],
                  [0, 0, 1]])

    cx = width / 2.0
    cy = height / 2.0
    pv = pu / width * height

    # Ma trận pixel
    P = np.array([[1 / pu, 0, cx],
                  [0, 1 / pv, cy],
                  [0, 0, 1]])

    fps = 30
    _location1 = invert(location1)
    inverted_location1 = np.array([_location1[0], _location1[2]])
    _location2 = invert(location2)
    inverted_location2 = np.array([_location2[0], _location2[2]])

    d_pixel = math.sqrt(math.pow(inverted_location2[0] - inverted_location1[0], 2) + math.pow(inverted_location2[1] - inverted_location1[1], 2))

    speed = d_pixel * 30 * 3.6

    return speed




