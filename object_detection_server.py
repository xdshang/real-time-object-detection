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
            bboxes = object_detection_pb2.BBoxes()
            if self.detector:
                results = self.detector.detect(img)
                for res in results:
                    bbox = bboxes.bboxes.add()
                    bbox.category = res[0]
                    bbox.center_x = res[1]
                    bbox.center_y = res[2]
                    bbox.width = res[3]
                    bbox.height = res[4]
                    bbox.confidence = res[5]
            yield bboxes


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