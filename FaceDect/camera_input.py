import cv2
import paddlehub as hub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pylab
import numpy as np

face_detector = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320")
# Define the face detection function and return the path of the recognized image or video
def face_det(img):
    result = face_detector.face_detection(images=[img], visualization='false', output_dir='face_detection_result')
    if(result[0]['data']):
        print("1")
    else:
        print("0")
    # return result[0]['save_path']


# Define the function to rotate the picture
def RotateClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 1)
    return new_img

# Define video processing function
def video_pro(visualization):
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 1
    cnt = 1
    timeF  = fps*2
    cv2.namedWindow("Image")
    while cap.isOpened():
        ret, frame = cap.read()
        if visualization:
            cv2.imshow("1", frame)
            c = cv2.waitKey(30) & 0xff
            if c == 27:
                cap.release()
                return
        if(cnt %timeF != 1):
            continue
        else:
            cnt = 0
        cnt+=1
        if ret:
            face_det(frame)
            #result_path =face_det(frame)
            #old_img = cv2.imread(result_path)
            # cv2.imshow("Image", old_img)
            # cv2.waitKey(1)

        if count == frame_count:
            break
        else:
            count += 1
    cv2.destroyAllWindows()
    cap.release()
    # out.release()

if __name__ == '__main__':
    video_pro(1)#
    print("处理完毕")