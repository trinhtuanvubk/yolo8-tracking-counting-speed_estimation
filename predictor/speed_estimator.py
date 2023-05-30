import math



def twopoint_speed(Location1, Location2):
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

def twoline_speed(num_frames, line1, line2):

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


def transform_3d_speed():
    pass



