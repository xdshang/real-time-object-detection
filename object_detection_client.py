import argparse
import time
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2
import numpy as np
import cPickle
import grpc
from utils import draw_result

import object_detection_pb2
import object_detection_pb2_grpc

_image = None


def webcam(vs, scale=0.7, mirror=False):
    global _image
    while True:
        img = vs.read()
        img = imutils.resize(img, height=320)
        if mirror: 
            img = cv2.flip(img, 1)
        _image = img
        jpg = cv2.imencode('.jpg', _image)[1]
        yield object_detection_pb2.Image(jpeg_data=cPickle.dumps(jpg))


def run(args):
    global _image
    channel = grpc.insecure_channel(args.server)
    stub = object_detection_pb2_grpc.DetectorStub(channel)
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()
    try:
        for out in stub.detect(webcam(vs, mirror=True)):
            result = cPickle.loads(out.data)
            display = draw_result(_image, result)
            cv2.imshow('Object Detection', display)
            cv2.waitKey(20)
            fps.update()
    except grpc._channel._Rendezvous as err:
        print(err)
    except KeyboardInterrupt:
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cv2.destroyAllWindows()
        vs.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection Client')
    parser.add_argument('--server', default='next-gpu2.d2.comp.nus.edu.sg:50051',
            help='Server url:port')
    args = parser.parse_args()
    run(args)