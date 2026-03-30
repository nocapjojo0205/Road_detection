import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


def plot_training_metrics(train_loss, test_loss, train_acc, test_acc, save_path):
    """绘制训练指标曲线"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Test')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    # 添加分类指标
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(0.5, -0.2,
             f'Accuracy: {accuracy:.2%}\nTotal Samples: {len(y_true)}',
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path, dpi=300)
    plt.close()




