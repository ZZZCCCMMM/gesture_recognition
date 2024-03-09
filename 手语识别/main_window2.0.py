"""
In this example, we demonstrate how to create simple face detection using Opencv3 and PyQt5

Author: Berrouba.A
Last edited: 23 Feb 2018
"""

# import system module
from detect import detectWindow
import sqlite3
from PyQt5 import QtSql
from PyQt5.QtSql import QSqlDatabase,QSqlQuery
import sys  # 系统参数操作
from PyQt5.QtWidgets import *  # 模块包含创造经典桌面风格的用户界面提供了一套UI元素的类
from PyQt5.QtCore import *  # 此模块用于处理时间、文件和目录、各种数据类型、流、URL、MIME类型、线程或进程
from PyQt5.QtGui import *  # 含类窗口系统集成、事件处理、二维图形、基本成像、字体和文本
# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

#Holistic 人体检测模型（手、脸、姿）
#Drawing 连接检测到的关键点，可视化检测信息
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

from ui_main_window import *

class MainWindow(QWidget):
    # class constructor
    def __init__(self, *args, **kwargs):
        '''
        构造函数，初始化参数属性
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.setWindowTitle('主功能页面')
        self.setFixedWidth(600)
        self.setFixedHeight(600)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.detectFaces)
        # set control_bt callback clicked  function
        self.ui.control_bt.clicked.connect(self.controlTimer)

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.control_bt.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.control_bt.setText("Start")
    def detectFaces(self):

        # 1. New detection variables
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5
        model = load_model('action.h5')
        actions = np.array(['hello', 'thanks', 'iloveyou'])
        label_map = {label: num for num, label in enumerate(actions)}


        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():
                # Read feed
                ret, frame = self.cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                print(str(len(sequence)))
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))

                    # 3. Viz logic
                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:

                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    # Viz probabilities
                    image = prob_viz(res, actions, image, colors)
                # convert frame to RGB format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # get frame infos
                height, width, channel = image.shape
                step = channel * width
                # create QImage from RGB frame
                qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
                # show frame in img_label
                self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
                self.ui.result_label.setText(str(sentence))
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            self.cap.release()
            cv2.destroyAllWindows()

            

class LoginDialog(QDialog):
    def __init__(self, *args, **kwargs):
        '''
        构造函数，初始化登录对话框的内容
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self._detect = None
        self.setWindowTitle('欢迎登录')  # 设置标题
        self.resize(200, 200)  # 设置宽、高
        self.setFixedSize(self.width(), self.height())
        self.setWindowFlags(Qt.WindowCloseButtonHint)  # 设置隐藏关闭X的按钮

        '''
        定义界面控件设置
        '''
        self.frame = QFrame(self)  # 初始化 Frame对象
        self.verticalLayout = QVBoxLayout(self.frame)  # 设置横向布局


        self.login_id = QLineEdit()  # 定义用户名输入框
        self.login_id.setPlaceholderText("请输入登录账号")  # 设置默认显示的提示语
        self.verticalLayout.addWidget(self.login_id)  # 将该登录账户设置添加到页面控件

        self.passwd = QLineEdit()  # 定义密码输入框
        self.passwd.setPlaceholderText("请输入登录密码")  # 设置默认显示的提示语
        self.verticalLayout.addWidget(self.passwd)  # 将该登录密码设置添加到页面控件

        self.button_enter = QPushButton()  # 定义登录按钮
        self.button_enter.setText("登录")  # 按钮显示值为登录
        self.verticalLayout.addWidget(self.button_enter)  # 将按钮添加到页面控件

        self.button_quit = QPushButton()  # 定义返回按钮
        self.button_quit.setText("返回")  # 按钮显示值为返回
        self.verticalLayout.addWidget(self.button_quit)  # 将按钮添加到页面控件

        self.button_detect = QPushButton()
        self.button_detect.setText("管理员界面")
        font1 = QtGui.QFont("Times", 16, QtGui.QFont.Bold)
        self.button_detect.setFont(font1)
        self.button_detect.setGeometry(450, 350, 200, 50)
        self.button_detect.setStyleSheet("QPushButton { background-color : gray;color :black ; }")
        self.verticalLayout.addWidget(self.button_detect)



        # 绑定按钮事件
        self.button_enter.clicked.connect(self.det)
        self.button_quit.clicked.connect(
            QCoreApplication.instance().quit)  # 返回按钮绑定到退出
        self.button_detect.clicked.connect(self.create_detect_window)
    def create_detect_window(self):
        # Function for opening Attendance window

        try:
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM student WHERE name = "+str(self.login_id.text())+" AND pwd = "+str(self.passwd.text())+" AND Id = 0")
            rows = cursor.fetchall()
            if bool(rows):
                self._detect = detectWindow(self)
                self._detect.show()
            conn.close()
        except Exception:
            QMessageBox.warning(QMessageBox(), 'Error', '密码或者用户名错误，请重新输入密码')


    def det(self):
        try:
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM student WHERE name = "+str(self.login_id.text()+" AND pwd = "+str(self.passwd.text())))
            rows = cursor.fetchall()
            if bool(rows):
                self.accept()
            conn.close()
        except Exception:
            QMessageBox.warning(QMessageBox(), 'Error', '密码或者用户名错误，请重新输入密码')






def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BRG2RGB
    image.flags.writeable = False  # 调整为禁止写入模式
    results = model.process(image)  # 进行预测
    image.flags.writeable = True  # 调整为可写入模式
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 重新转换为BGR
    return image, results


# 连接各部位关键点函数
def draw_styled_landmarks(image, results):
    # Draw the face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw the pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=4)
                              )
    # Draw the left_hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw the right_hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


# 提取检测到的姿态关键点信息（手、脸）保存成Pose、Face、LeftHand、RightHand列表中，Numpy数组
# 如果没检测到就置0
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


# 可视化
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame




if __name__ == '__main__':
    window_application = QApplication(sys.argv)
    # 设置登录窗口
    login_ui = LoginDialog()
    # 校验是否验证通过
    if login_ui.exec_() == QDialog.Accepted:
        # 初始化主功能窗口
        main_window = MainWindow()
        # 展示窗口
        main_window.show()
        # 设置应用退出
        sys.exit(window_application.exec_())