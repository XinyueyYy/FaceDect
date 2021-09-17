import cv2
import paddlehub as hub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pylab
import numpy as np

face_detector = hub.Module(name="pyramidbox_lite_server")
# 定义检测人脸函数，返回识别的图片路径
def face_det(img):
    result = face_detector.face_detection(images=[img], visualization='true', output_dir='face_detection_result')
    print(result)
    if(result[0]['data']):
        w = result[0]['data'][0]['right'] - result[0]['data'][0]['left']
        h = result[0]['data'][0]['bottom'] - result[0]['data'][0]['top']
        if (h / w < 1.4):
            print("1")
        else:
            print('0')
    else:
        print('0')
    return result[0]['path']


# 定义旋转图片函数
def RotateClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 1)
    return new_img


# 定义视频处理函数
def video_pro(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('帧率：%d' % fps)
    print('帧数：%d' % frame_count)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    out = cv2.VideoWriter(save_path, fourcc, fps, size)
    count = 1
    cnt = 1
    timeF  = fps*2
    cv2.namedWindow("Image")
    while cap.isOpened():
        ret, frame = cap.read()
        if(cnt %timeF != 1):
            continue
        else:
            cnt = 0
        cnt+=1
        if ret:
            face_det(frame)
            result_path = 'face_detection_result' + '/' + face_det(frame).split('.')[0] + '.jpg'
            old_img = cv2.imread(result_path)
            cv2.imshow("Image", old_img)
            cv2.waitKey(1)
            new_img = np.rot90(old_img)
            new_img = np.rot90(new_img)
            new_img = np.rot90(new_img)
            out.write(new_img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif count == frame_count:
                break
            else:
                count += 1
        else:
            break
    cv2.destroyAllWindows()
    cap.release()
    out.release()

if __name__ == '__main__':
    video_pro('Test1.mp4', "Result1.mp4")
    print("处理完毕")