import cv2
import numpy as np
import time
import threading
import math
import signal
import copy

# from CameraCalibration.CalibrationConfig import * # Importing the camera calibration if needed

ret = False
debug = True
calib = False
r_w = 640
r_h = 480




############################### calibration #######################
####load calibration parameters####

# param_data = np.load(calibration_param_path + '.npz')

#get the parameters
# dim = tuple(param_data['dim_array'])
# k = np.array(param_data['k_array'].tolist())
# d = np.array(param_data['d_array'].tolist())

# print('parameters loaded:')
# print('dim:\n', dim)
# print('k:\n', k)
# print('d:\n', d)

####crop and scale, 1 means no crop####

# scale = 1
# p = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(k, d, dim ,None)
# Knew = p.copy()
# if scale:#change fov
#     Knew[(0,1), (0,1)] = scale * Knew[(0,1), (0,1)]
# map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), Knew, dim, cv2.CV_16SC2)

############################### calibration done #######################

cap = cv2.VideoCapture(-1)


################################################ image reading thread #################################################
def get_img():
    global cap, org_img, debug, ret
    while True:
        if cap.isOpened():
            ret, org_img_fish = cap.read()
            #org_img = cv2.remap(org_img_fish.copy(), map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
            if ret:
                if calib:
                    org_img = cv2.remap(org_img_fish.copy(), map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                else:
                    org_img = org_img_fish.copy()
                if debug:
                    cv2.imshow('orginal frame', org_img)
#                     time.sleep(0.5)
                    time.sleep(0.6)
                    key = cv2.waitKey(3)
                    
            else:
                time.sleep(0.01)
        else:
            time.sleep(0.01)

# open the thread
th1 = threading.Thread(target=get_img)
th1.setDaemon(True)  # set as background thread, default is False, set as True, the main thread does not need to wait for the child thread
th1.start()
################################################ image reading thread done #################################################


# def draw_image(image):
#     # your code here
#     pass

def draw_text(image, text):
    # your code here
    pass

def inference(image):
    # your code here
    pass


while True:
    if org_img is not None and ret:
        image_used = org_img.copy()
        print('image read')
        # your code here
        
    else:
        time.sleep(0.01)
        print('waiting for image')