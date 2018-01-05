# Real-Time Object Detection

### Prerequisites
- tensorflow-gpu, opencv-contrib-python, imutils, grpcio, cPickle
- [YOLO_small_weights](https://drive.google.com/file/d/0B2JbaJSrWLpza08yS2FSUnV2dlE/view?usp=sharing)

### Quick Start
1. On GPU server side, execute the code under "Run Server" in the jupyter notebook (the service listens on port 50051).
2. On client side with a webcam, execute `python object_detection_client.py --server [server_ip/hostname:50051]` to start demo.
3. Press `ctrl+C` to quit the programs.
