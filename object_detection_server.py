from concurrent import futures
import time
import numpy as np
import cv2
import cPickle

import grpc

import object_detection_pb2
import object_detection_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Detector(object_detection_pb2_grpc.DetectorServicer):

    def __init__(self, detector=None):
        super(Detector, self).__init__()
        self.detector = detector
    
    def detect(self, request_iterator, context):
        for request in request_iterator:
            jpg = cPickle.loads(request.jpeg_data)
            img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            if self.detector:
                result = self.detector.detect(img)
            else:
                result = 'Debug Info'
            yield object_detection_pb2.BBoxes(data=cPickle.dumps(result))


def serve(detector):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    object_detection_pb2_grpc.add_DetectorServicer_to_server(detector, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
             time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
     serve(Detector())