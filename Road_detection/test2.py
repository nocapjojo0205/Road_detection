from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
import sys, os, cv2, torch
from models.resnet18 import ResNet34_CBAM  # 替换为实际模型路径与类
import video_ui  # 导入 Qt Designer 生成的界面类

class VideoProcessor(QThread):
    frame_updated = pyqtSignal(QImage, dict)
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, model_path, video_path):
        super().__init__()
        self.video_path = video_path
        self._is_running = False
        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResNet34_CBAM(num_classes=6)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device).eval()

    def preprocess_frame(self, frame):
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit()
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        class_names = ['干燥','湿滑','结冻','积雪','吹雪','融雪']
        self._is_running = True
        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                break
            tensor = self.preprocess_frame(frame)
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                idx = torch.argmax(probs, dim=1).item()
            result = {
                'class': class_names[idx],
                'confidence': f"{probs[0][idx].item():.2%}",
                'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            }
            # 转QImage
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w,ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            # 发信号
            self.frame_updated.emit(qimg, result)
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.progress_updated.emit(int(pos/total*100))
            self.msleep(30)
        cap.release()
        self.finished.emit()

    def stop(self):
        self._is_running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = video_ui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.processor = None
        self.model_path = r'C:\Users\闫博乔\PycharmProjects\road_detection Project\scripts\saved_models\ResNet34_CBAM.pth'# 填写模型权重绝对路径

        # 初始禁用按钮
        self.ui.btn_start.setEnabled(False)
        self.ui.btn_play.setEnabled(False)
        self.ui.btn_stop.setEnabled(False)

        # 信号
        self.ui.btn_open.clicked.connect(self.open_video)
        self.ui.btn_play.clicked.connect(self.play)
        self.ui.btn_start.clicked.connect(self.play)
        self.ui.btn_stop.clicked.connect(self.stop)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择视频', '', '视频 (*.mp4 *.avi *.mov)')
        if path:
            self.video_path = path
            self.ui.video_path.setText(os.path.basename(path))
            self.ui.btn_play.setEnabled(True)
            self.ui.btn_start.setEnabled(True)
            self.ui.result_label.clear()

    def play(self):
        if not hasattr(self, 'video_path'):
            QMessageBox.warning(self, '提示', '请先选择视频')
            return
        # 如果已存在线程，先停止
        if self.processor:
            self.processor.stop()
        # 创建新线程
        self.processor = VideoProcessor(self.model_path, self.video_path)
        self.processor.frame_updated.connect(self.update_frame)
        self.processor.progress_updated.connect(self.update_progress)
        self.processor.finished.connect(self.on_finished)
        self.ui.btn_play.setEnabled(False)
        self.ui.btn_start.setEnabled(False)
        self.ui.btn_stop.setEnabled(True)
        self.processor.start()

    def stop(self):
        if self.processor:
            self.processor.stop()
        self.ui.btn_play.setEnabled(True)
        self.ui.btn_start.setEnabled(True)
        self.ui.btn_stop.setEnabled(False)

    def update_frame(self, img, result):
        self.ui.video_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.ui.video_label.size(), Qt.KeepAspectRatio))
        self.ui.result_label.setText(f"状态: {result['class']}\n置信度: {result['confidence']}\n时间: {result['timestamp']:.1f}s")

    def update_progress(self, val):
        self.ui.slider.setValue(val)

    def on_finished(self):
        self.ui.btn_stop.setEnabled(False)
        self.ui.btn_play.setEnabled(True)
        self.ui.btn_start.setEnabled(True)
        QMessageBox.information(self, '提示', '分析完成')

    def closeEvent(self, event):
        if self.processor:
            self.processor.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
