from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
from PyQt5.QtCore import QTimer
import sys
import main_ui
import torch
import torchvision.transforms as transforms
from PIL import Image

#导入自定义模型
from models.resnet18 import ResNet34_CBAM
from models.safe_distance import DrivingSafetySystem
class UI_main_1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = main_ui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.img_path = None
        self.model = self.load_model()
        self.class_names = ['dry', 'wet', 'icy', 'fully_snowy', 'snow_blowing', 'snow_melting']
        # 设置初始窗口尺寸（宽度, 高度）
        self.setMinimumSize(1000, 700)  # 强制设置窗口尺寸
        font = self.ui.label.font()
        font.setPointSize(8)  # 设置字体大小为10
        self.ui.label.setFont(font)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton.clicked.connect(lambda: self.openimage())
        self.ui.pushButton_2.clicked.connect(lambda: self.inferimage())
        self.ui.pushButton_3.clicked.connect(lambda: self.exit())
        self.ui.pushButton_4.clicked.connect(lambda: self.calculate_safety())
        self.safety_system = DrivingSafetySystem() #实例化安全系统
        self.show()

        # 车速输入框设置 (km/h)
        self.ui.doubleSpinBox.setRange(0, 55.56)  # 200km/h ≈ 55.56m/s
        self.ui.doubleSpinBox.setDecimals(2)  # 1位小数
        self.ui.doubleSpinBox.setValue(20)  # 默认20 m/s
        self.ui.doubleSpinBox.setSuffix(" m/s")

        # 前车距离输入框设置 (m)
        self.ui.doubleSpinBox_2.setRange(0, 1000)  # 0-1000米
        self.ui.doubleSpinBox_2.setDecimals(1)  # 1位小数
        self.ui.doubleSpinBox_2.setValue(100)  # 默认50米
        self.ui.doubleSpinBox_2.setSuffix(" m")
    # 关闭程序
    def exit(self):
        self.close()

    #打开图片
    def openimage(self):
        """打开图片"""
        imgName, _ = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            "",
            "*.jpg;;*.png;;All Files(*)"
        )

        if not imgName:
            return

        # 显示原始图片
        pixmap = QPixmap(imgName).scaled(
            self.ui.label_origin.width(),
            self.ui.label_origin.height()
        )
        self.ui.label_origin.setPixmap(pixmap)
        self.img_path = imgName
        self.ui.label_result.clear()  # 清除之前的推理结果

    def load_model(self):
        # 加载训练好的模型
        model = ResNet34_CBAM()  # 请根据实际类别数调整
        model.load_state_dict(torch.load(r'C:\Users\闫博乔\PycharmProjects\road_detection Project\scripts\saved_models\ResNet34_CBAM.pth'))
        model.eval()
        return model

    def preprocess_image(self, image_path):
        """图像预处理"""
        try:
            image = Image.open(image_path).convert('RGB')

            # 预处理管道（根据实际训练参数调整）
            transform = transforms.Compose([
                transforms.Resize((256, 256)),  # 根据模型输入尺寸调整
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            return transform(image).unsqueeze(0)
        except Exception as e:
            print(f"预处理错误: {str(e)}")
            return None

    def calculate_safety(self):
        """执行安全距离计算"""
        # 获取输入值
        try:
            speed = self.ui.doubleSpinBox.value()  # 直接获取浮点数值（单位：m/s）
            distance = self.ui.doubleSpinBox_2.value()  # 单位：米
            road_condition = self.ui.label_result.text()
            # 2. 验证输入范围
            if speed <= 0 or speed > 55.56:  # 对应200km/h的m/s值
                raise ValueError("车速需在0-55.56 m/s范围内")
            if distance <= 0 or distance > 1000:
                raise ValueError("前车距离需在0-1000米范围内")

            # 4. 计算安全距离
            warning = self.safety_system.get_warning_level(
                actual_distance=distance,
                speed=speed,
                road_condition=road_condition  # 使用英文状态标识
            )

            # 5. 显示结果
            self.display_warning_result(warning)
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", str(e))
        except Exception as e:
            QMessageBox.critical(self, "系统错误", f"未知错误: {str(e)}")


    def display_warning_result(self, warning):
        """显示预警结果"""
        color_mapping = {
            "安全驾驶": "yellow",
            "一级预警": "orange",
            "二级预警": "red"
        }
        color = color_mapping.get(warning.split("（")[0], "black")

        self.ui.label_warning.setText(warning)
        self.ui.label_warning.setStyleSheet(f"""
            color: {color};
            font-weight: bold;
            font-size: 14px;
            padding: 10px;
            border: 1px solid {color};
            border-radius: 5px;
        """)

    #推理功能
    def inferimage(self):
        """执行推理"""
        if not self.img_path:
            QtWidgets.QMessageBox.warning(
                self,
                "警告",
                "请先选择图片！"
            )
            return

        input_tensor = self.preprocess_image(self.img_path)
        if input_tensor is None:
            self.ui.label_result.setText("图片处理失败")
            self.ui.label_result.setStyleSheet("color: red;")
            return

        if self.model is None:
            self.ui.label_result.setText("模型未加载")
            self.ui.label_result.setStyleSheet("color: red;")
            return

        with torch.no_grad():
            try:
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                conf, preds = torch.max(probabilities, 1)

                result_text = (
                    f"预测结果: {self.class_names[preds.item()]}\n"
                    f"置信度: {conf.item():.2%}"
                )
                self.ui.label_result.setText(result_text)
                self.ui.label_result.setStyleSheet(
                    "color: green;"
                    "font-size: 16px;"
                )
            except Exception as e:
                self.ui.label_result.setText(f"推理失败: {str(e)}")
                self.ui.label_result.setStyleSheet("color: red;")

        self.current_road_condition = self.class_names[preds.item()]
        road_condition_cn = {
            'dry': '干燥',
            'wet': '湿滑',
            'icy': '结冰',
            'fully_snowy': '积雪',
            'snow_blowing': '吹雪',
            'snow_melting': '融雪'
        }
        self.ui.label_result.setText(
            road_condition_cn.get(self.current_road_condition, "未知")
        )
        self.ui.label_result.setStyleSheet("color: blue;")

    # 拖动
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图标



if __name__ =="__main__":
    app =QApplication(sys.argv)
    win = UI_main_1()
    sys.exit(app.exec_())