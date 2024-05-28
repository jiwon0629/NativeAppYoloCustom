import sys
import cv2
import torch
import time

# 여기서 사용한 Pyside6는 pyQt5와 완전 호환됩니다. 
# GPT에 pyQt5와 PySide6는 뭐가 다르냐?  물어 보세요 
from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer

# 작업폴더에 Yolov5를 클로닝하세요 
# git clone https://github.com/ultralytics/yolov5.git

# window_yolo_ui.ui 파일을 window_yolo_ui.py 파일로 변환 하여 import 시키는 방법
from window_yolo_ui import Ui_MainWindow
# 터미널에서 pyside6-uic window_yolo.ui -o window_yolo_ui.py
import serial
import time

# 시리얼 포트 설정
SERIAL_PORT = 'COM4'  # 시리얼 포트 이름 (로봇이 연결된 포트에 맞게 수정)
SERIAL_BAUDRATE = 112500  # 통신 속도 (로봇과의 통신 속도에 맞게 수정)

# 시리얼 통신 설정
ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE)

def robot_action(no):
    # 로봇을 움직이는 프로토콜에 따라 명령을 전달하는 것을 구현
    exe_cmd = bytearray([0xff, 0xff, 0x4c, 0x53, 0x00, 
                         0x00, 0x00, 0x00, 0x30, 0x0c, 0x03,  # 0x0c=동작 실행 명령, 0x03=파라메터 수 
                         0x01, 0x00, 100, 0x00])  # Index 01 명령어 샘플
    """
    |      |    |    `---Check sum
    |      |    `--------속도      파라1
    |      `-------------지연시간  파라2
    `--------------------Index    파라3
    """
    exe_cmd[11] = no
    exe_cmd[14] = 0x00  # checksum
    
    for i in range(6, 14):
        exe_cmd[14] += exe_cmd[i]
    
    ser.write(exe_cmd)
    
    time.sleep(0.05)  # 50ms 딜레이

# 예시로 robot_action 함수를 호출하여 로봇을 제어할 수 있습니다.
# robot_action(1)  # 예시로 명령 번호 1을 전달하여 로봇을 제어


# class name을 갖고 있은 배열을 이용하기 
# COCO 클래스 이름
cls_names = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
    'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # 메인 윈도우 제목 설정
        self.setWindowTitle("Custom Model을 이용한 YOLO 웹캠")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 자신이 만든(학습시킨) pt파일을(대부분은 best.py라고 쓰고 있겠지만) 이름을 바꿔서 
        # 내 폴더에 놓고 사용하는 방법
        self.model = torch.hub.load("./yolov5", model="custom",
                                    path="./myModels/yolov5s.pt", source="local")
        # 폴더 
        # 여기는 반드시 model = "custom"
        # 그리고 path 에 있는 것이 실제 본인의 모델임 여기서는 편의상 yolov5s.pt를 사용했음.(본인것으로 바꾸세요)
        # 맨뒤에 source = "local"

        self.model.to(device=self.device)

        # 이전 동작 저장 변수 초기화
        self.previous_action = None

        # 타이머를 이용해서 20mSec 마다 1개의 프레임을 읽어 오는 것 
        # 이것은 Thread를 쓰는 것 보다 비교적 간단한 방법으로 화면 딜레이를 줄일수 있음.,
        # 즉 모델이 Predict 하는 시간을 벌어 주는 역할을 함.         
        self.timer = QTimer()  # 타이머
        self.timer.setInterval(500)  # 타이머 간격을 20ms로 설정, 즉 매 20ms마다 신호를 보내 한 프레임 추론
                                    # 실제 처리 시간이 167 정도 소요 되므로 인터벌을 길게 잡아도 될듯 
        
        self.video = cv2.VideoCapture(0)  # 웹캠 열기

        self.timer.timeout.connect(self.video_pred) # 매 타이머 마다 이 함수를 실행하도록 예약 
        self.timer.start()

    def convert2QImage(self, img):  # 배열을 QImage로 변환하여 표시
        height, width, channel = img.shape
        return QImage(img, width, height, width * channel, QImage.Format_RGB888)

    def video_pred(self):  # 비디오 감지
        ret, frame = self.video.read()  # ret은 프레임을 읽었는지 여부, frame은 읽은 프레임
        # 매번 read() 시 다음 프레임을 읽음
        if not ret:
            self.timer.stop()
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.input.setPixmap(QPixmap.fromImage(self.convert2QImage(frame)))
            start = time.perf_counter()
            results = self.model(frame)

            # 감지된 객체를 confidence 값으로 정렬하여 상위 1개 출력
            detections = results.xyxy[0].cpu().numpy()
            if len(detections) > 0:
                sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)[:1]  # confidence 값 기준으로 정렬 후 상위 1개 선택
                for i, det in enumerate(sorted_detections):
                    xyxy = det[:4]
                    conf = det[4]
                    cls = int(det[5])
                    class_name = cls_names[cls] if cls < len(cls_names) else f"클래스 {cls}"
                    if i == 0:
                        self.label.setText(f"클래스:{cls}-{class_name}, confidence: {conf:.2f}")
                    elif i == 1:
                        self.label_2.setText(f"클래스:{cls}-{class_name}, confidence: {conf:.2f}")
                    
                    # 이전 동작과 현재 동작이 다른 경우에만 로봇 제어 동작 실행
                    if self.previous_action != class_name:
                        if class_name == cls_names[0]:  # person이라면
                            robot_action(19)  # 로봇 제어 명령 호출
                        elif class_name == 'cell phone': #cell phone이라면
                            robot_action(17)  # 로봇 제어 명령 호출
                        elif class_name == 'bottle': #bottle이라면
                            robot_action(20)  # 로봇 제어 명령 호출
                        elif class_name == 'stop sign': #stop sign이라면
                            robot_action(16)  # 로봇 제어 명령 호출
                        elif class_name == 'cup': #cup이라면
                            robot_action(21)  # 로봇 제어 명령 호출
                        elif class_name == 'fork': #fork이라면
                            robot_action(18)  # 로봇 제어 명령 호출
                        elif class_name == 'knife': #knife이라면
                            robot_action(30)  # 로봇 제어 명령 호출
                        elif class_name == 'spoon': #spoon이라면
                            robot_action(31)  # 로봇 제어 명령 호출
                        elif class_name == 'book': #book이라면
                            robot_action(32)  # 로봇 제어 명령 호출
                        elif class_name == 'scissors': #scissors이라면
                            robot_action(33)  # 로봇 제어 명령 호출
                            
                        # 이전 동작 업데이트
                        self.previous_action = class_name

            end = time.perf_counter()
            self.label_3.setText(f'판독시간:{round((end - start) * 1000,4)} ms')
            
            image = results.render()[0]
            self.output.setPixmap(
                QPixmap.fromImage(self.convert2QImage(image)))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())