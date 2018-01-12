import argparse
import time
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2
import numpy as np
import pickle
import grpc
from utils import draw_result

import object_detection_pb2
import object_detection_pb2_grpc


def webcam(vs, mirror=False):
    while True:
        img = vs.read()
        if mirror: 
            img = cv2.flip(img, 1)
        # crop image to square as YOLO input
        if img.shape[0] < img.shape[1]:
            pad = (img.shape[1]-img.shape[0])//2
            img = img[:, pad: pad+img.shape[0]]
        else:
            pad = (img.shape[0]-img.shape[1])//2
            img = img[pad: pad+img.shape[1], :]
        yield img


def run(args, size=224):
    print('[INFO] starting...')
    channel = grpc.insecure_channel(args.server)
    stub = object_detection_pb2_grpc.DetectorStub(channel)
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    fps = FPS().start()
    try:
        for img in webcam(vs, mirror=True):
            # compress frame
            resized_img = cv2.resize(img, (size, size))
            jpg = cv2.imencode('.jpg', resized_img)[1]
            # send to server for object detection
            response = stub.detect(object_detection_pb2.Image(jpeg_data=pickle.dumps(jpg)))
            # parse detection result and draw on the frame
            result = pickle.loads(response.data)
            display = draw_result(img, result, scale=float(img.shape[0])/size)
            cv2.imshow('Object Detection', display)
            cv2.waitKey(1)
            fps.update()
    except grpc._channel._Rendezvous as err:
        print(err)
    except KeyboardInterrupt:
        fps.stop()
        print('[INFO] elapsed time: {:.2f}'.format(fps.elapsed()))
        print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
        cv2.destroyAllWindows()
        vs.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection Client')
    parser.add_argument('--server', default='next-gpu3.d2.comp.nus.edu.sg:50051',
            help='Server url:port')
    args = parser.parse_args()
    run(args)