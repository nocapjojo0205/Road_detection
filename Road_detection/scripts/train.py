import torch
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from data.dataloader import get_data_loaders
from models.resnet18 import ResNet18, MobileNetv2, ResNet34_CBAM
from models.resnet_se import RoadClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tabulate import tabulate
from sklearn.metrics import classification_report
import os
sns.set_theme()
import matplotlib.font_manager as fm
def load_config(config_path="../configs/base_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
report_dict = {}
global_metrics = {}
# === 图表与分析部分 ===
# 设置中文字体，推荐使用 SimHei 或 Microsoft YaHei，具体根据系统可用字体选择
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
def train(model, model_name):
    config = load_config()
    train_loader, test_loader, classes = get_data_loaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])

    best_acc = 0.0
    global global_metrics
    global_metrics[model_name] = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'best_acc': 0.0,
        'best_epoch': 0
    }
    metrics = global_metrics[model_name]

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        metrics['train_loss'].append(epoch_loss / len(train_loader))
        metrics['train_acc'].append(correct / total)

        # Validation
        model.eval()
        test_loss = 0.0
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算每个类别的指标
        cls_report = classification_report(
            all_labels, all_preds,
            target_names=classes,
            output_dict=True
        )

        # 存储每个类别的指标
        class_metrics = []
        for cls in classes:
            class_metrics.append({
                'class': cls,
                'precision': cls_report[cls]['precision'],
                'recall': cls_report[cls]['recall'],
                'f1': cls_report[cls]['f1-score']
            })
        metrics['class_metrics'] = class_metrics
        # 计算验证指标
        # 计算验证指标
        test_acc = correct / total
        metrics['test_loss'].append(test_loss / len(test_loader))
        metrics['test_acc'].append(test_acc)
        metrics['precision'].append(precision_score(all_labels, all_preds, average='macro'))
        metrics['recall'].append(recall_score(all_labels, all_preds, average='macro'))
        metrics['f1'].append(f1_score(all_labels, all_preds, average='macro'))

        if test_acc > best_acc:
            best_acc = test_acc
            metrics['best_acc'] = best_acc
            metrics['best_epoch'] = epoch + 1
        # 保存模型和指标
        torch.save(model.state_dict(), f"saved_models/{model_name}.pth")
        print(f"✅ Best model updated at epoch {epoch + 1} with acc {best_acc:.2%}")
        with open(f"metrics/{model_name}_metrics.json", "w") as f:
            json.dump(metrics, f)

    return metrics

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
def plot_compare_metrics(global_metrics, save_path="model_compare.png"):
    plt.figure(figsize=(12, 5))

    ax = plt.subplot(1, 1, 1)
    for model_name, metrics in global_metrics.items():
        smoothed = np.convolve(metrics['train_loss'], np.ones(5)/5, mode='valid')
        ax.plot(smoothed, label=f"{model_name} ")

    ax.set_title("训练损失对比", fontsize=14)
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("损失值")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_comparison_table(global_metrics):
    table_data = []
    for model_name, m in global_metrics.items():
        row = [
            model_name,
            f"{m['train_loss'][-1]:.4f}",
            f"{m['test_acc'][-1]:.2%}",
            f"{m['precision'][-1]:.2%}",
            f"{m['recall'][-1]:.2%}",
            f"{m['f1'][-1]:.2%}",
            m['best_epoch']
        ]
        table_data.append(row)

    headers = ["Model", "Final Train Loss", "Final Accuracy", "Precision", "Recall", "F1 Score", "Best Epoch"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    print(table)
    with open("model_metrics_table.txt", "w") as f:
        f.write(table)

def plot_conf_matrix(all_labels, all_preds, class_names, save_path='conf_matrix.png'):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_metrics(model_names):
    global_metrics = {}
    for model_name in model_names:
        with open(f"metrics_{model_name}.json", "r") as f:
            global_metrics[model_name] = json.load(f)
    return global_metrics

# 去除 state_dict 中的 'backbone.' 前缀
def remove_backbone_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_k = k.replace("backbone.", "")
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

def convert_weights(weights):
    new_weights = {}
    for k, v in weights.items():
        if k.startswith('backbone.'):
            new_k = k.replace('backbone.', 'features.')
            # 可能需要更复杂的替换规则
            new_weights[new_k] = v
        else:
            new_weights[k] = v
    return new_weights


# 3. 可视化函数
def plot_class_metrics(all_metrics, classes):
    # 准备数据
    models = list(all_metrics.keys())
    metrics = ['precision', 'recall', 'f1']

    # 创建三个图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        # 收集每个模型每个类别的指标
        data = []
        for model_name in models:
            model_metrics = all_metrics[model_name]['class_metrics']
            data.append([m[metric] for m in model_metrics])

        # 转换为numpy数组便于处理
        data = np.array(data)

        # 绘制堆叠柱状图
        x = np.arange(len(classes))
        width = 0.8 / len(models)

        for j, model_name in enumerate(models):
            ax.bar(x + j * width, data[j], width, label=model_name)

        ax.set_title(metric.capitalize())
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_ylim(0, 1.1)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_metrics.png', dpi=300)
    plt.show()

# === 主流程 ===
if __name__ == "__main__":
    # 初始化
    # os.makedirs("saved_models", exist_ok=True)
    # os.makedirs("metrics", exist_ok=True)
    # os.makedirs("results", exist_ok=True)
    # models_to_train = {
    #     "ResNet18": ResNet18(num_classes=6),
    #     "MobileNetV2": MobileNetv2(num_classes=6),
    #     "ResNet34_CBAM": ResNet34_CBAM(num_classes=6),
    #     "ResNet_SE": RoadClassifier(num_classes=6)
    # }
    # config = load_config()
    # train_loader, test_loader, classes = get_data_loaders(config)
    # all_metrics = {}
    # for model_name, model in models_to_train.items():
    #     all_metrics[model_name] = train(model, model_name)
    # # 可视化结果
    # plot_class_metrics(all_metrics, classes)
    #
    # print("所有模型训练完成！")
    # 保存高清图像
    # 定义激活函数
    def relu(x):
        return np.maximum(0, x)


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    # 定义输入范围
    x = np.linspace(-10, 10, 1000)

    # 计算激活函数输出
    y_relu = relu(x)
    y_sigmoid = sigmoid(x)

    # 创建图像
    plt.figure(figsize=(10, 5))

    # 绘制 ReLU 函数
    plt.subplot(1, 2, 1)
    plt.plot(x, y_relu, label="ReLU", color="blue")
    plt.title("ReLU Activation Function")
    plt.xlabel("x")
    plt.ylabel("ReLU(x)")
    plt.grid(True)
    plt.legend()

    # 绘制 Sigmoid 函数
    plt.subplot(1, 2, 2)
    plt.plot(x, y_sigmoid, label="Sigmoid", color="green")
    plt.title("Sigmoid Activation Function")
    plt.xlabel("x")
    plt.ylabel("Sigmoid(x)")
    plt.grid(True)
    plt.legend()

    # 显示图像
    plt.tight_layout()
    plt.show()

    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    # plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # global_metrics = load_metrics(trained_models)
    #
    # plot_compare_metrics(global_metrics)
    # generate_comparison_table(global_metrics)



