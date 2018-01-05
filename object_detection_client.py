import time
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2
import numpy as np
import cPickle
import grpc

import object_detection_pb2
import object_detection_pb2_grpc


def webcam(vs, scale=0.7, mirror=False):
    while True:
        img = vs.read()
        img = imutils.resize(img, height=320)
        if mirror: 
            img = cv2.flip(img, 1)
        jpg = cv2.imencode('.jpg', img)[1]
        yield object_detection_pb2.Image(jpeg_data=cPickle.dumps(jpg))


def run():
    channel = grpc.insecure_channel('xindi-optiplex-9020.d1.comp.nus.edu.sg:50051')
    stub = object_detection_pb2_grpc.DetectorStub(channel)
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()
    try:
        for out in stub.detect(webcam(vs, mirror=True)):
            jpg = cPickle.loads(out.jpeg_data)
            img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            cv2.imshow('Object Detection', img)
            cv2.waitKey(1)
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
    run()