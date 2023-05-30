import numpy as np
import cv2
from math import cos, sin


def convert(X):
    """Chuyển đổi từ tọa độ điểm trong thế giới thực sang tọa độ pixel"""
    # Tọa độ thực tế theo camera
    Yc = R @ T @ X
    # Tọa độ khi chiếu qua camera
    Yz = (K @ Yc[:3]) / Yc[2]
    # Tọa độ pixel
    Yp = P @ Yz
    x, y = int(Yp[0]), int(Yp[1])
    return x, y


def invert(Y):
    """Chuyển đổi điểm từ tọa độ pixel sang tọa độ trong thế giới thực"""
    Y = np.concatenate([Y[:2], [1]], axis=-1)
    Yz = np.linalg.inv(P) @ Y
    # phương trình đường thẳng d đi qua Y và tâm
    dv = np.concatenate([Yz[:2], [-f]], axis=-1)  # direction vector
    x = np.array([0, 0, 0])

    # tính phương trình mặt phẳng M với tọa độ gốc là camera biến đổi từ mặt phẳng y = 0
    n0 = np.array([0, 1, 0, 1])  # vector pháp tuyến trước khi thực hiện phép biến đổi
    x0 = [0, 0, 0, 1]  # 1 điểm nằm trên mặt phẳng y=0
    nc = R @ T @ n0  # vector pháp tuyến sau khi thực hiện phép biến đổi
    xc = R @ T @ x0  # 1 điểm nằm trên mặt phẳng y=0 sau khi biến đổi
    pec = plane_equation(xc, nc)  # tính phương trình mặt phẳng với vector pháp tuyến và điểm

    #  tính điểm giao giữa d và M
    t = (-pec[3] - (pec[:3] @ x)) / (pec[:3] @ dv)
    Yc = dv * t
    Yc = np.concatenate([Yc, [1]], axis=-1)
    X = np.linalg.inv(R @ T) @ Yc
    return X


def plane_equation(x, v):
    """Phương trình đường thẳng đi qua điểm x và nhận vector v là vector pháp tuyến
        aX + bY + cZ + d = 0"""
    a, b, c = v[0], v[1], v[2]
    d = -(a * x[0] + b * x[1] + c * x[2])
    return np.array([a, b, c, d])


if __name__ == '__main__':
    img_path = './img_test.png'
    img = cv2.imread(img_path)
    # Kích thước ảnh
    height, width, _ = img.shape

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

    """ đoạn code sau tạo 1 lưới các điểm trên mặt đường (y=0),
    ánh xạ nó lên vị trí pixel tương ứng và hiển thị"""
    for x in range(-20, 20):
        for z in range(1, 200):
            y = 0
            # Tọa độ điểm 3D trong thực tế sang tọa độ pixel
            point_in_real_world = np.array([x, y, z, 1])
            point_in_pixel = convert(point_in_real_world)
            print(point_in_pixel)
            cv2.circle(img, point_in_pixel, 2, (255, 0, 0), 2)

            # chuyển đổi từ tọa độ pixel sang tọa độ thế giới thực
            p_inverted = invert(point_in_pixel)
            print(p_inverted)

    # cv2.imshow('test', img)
    cv2.imwrite("test.jpg", img)
    # key = cv2.waitKey(0)
