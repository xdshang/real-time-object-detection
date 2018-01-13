# Real-Time Object Detection

### Prerequisites
- Server: Python3, tensorflow-gpu, opencv-contrib-python, grpcio-tools
- Client: Python3, opencv-contrib-python, imutils, grpcio
- [YOLO_small_weights](https://drive.google.com/file/d/0B2JbaJSrWLpza08yS2FSUnV2dlE/view?usp=sharing)

### Quick Start
1. On GPU server side, run the jupyter notebook and execute the code under the title"Run Server" (the service listens on port 50051).
2. On client side with a webcam, execute `python object_detection_client.py --server [server_ip/hostname:50051]` to start demo.
3. Press `ctrl+C` to quit the programs.
