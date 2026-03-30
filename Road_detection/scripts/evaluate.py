import torch
import numpy as np
from sklearn.metrics import classification_report
from data.dataloader import get_data_loaders
from models.resnet_se import RoadClassifier
import yaml

def evaluate(config_path="configs/base_config.yaml"):
    # 加载配置
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 准备数据
    _, test_loader, classes = get_data_loaders(config)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadClassifier(num_classes=config['num_classes']).to(device)
    model.load_state_dict(torch.load(config['model_save_path']))
    model.eval()

    # 收集预测结果
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 生成分类报告
    print(classification_report(
        y_true, y_pred,
        target_names=classes,
        digits=4
    ))


if __name__ == "__main__":
    evaluate()