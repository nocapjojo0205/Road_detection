from torchvision import transforms, datasets
from torch.utils.data import DataLoader



def get_data_loaders(config):
    """创建训练和测试数据加载器"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),  # 随机旋转
            transforms.RandomResizedCrop(224),  # 随机裁剪并缩放
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机改变亮度、对比度等
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    # 打印路径
    train_dir = r'C:\Users\闫博乔\PycharmProjects\road_detection Project\data\train'
    test_dir = r'C:\Users\闫博乔\PycharmProjects\road_detection Project\data\test'

    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")

    train_dataset = datasets.ImageFolder(
        train_dir,
        data_transforms['train']
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        data_transforms['test']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    return train_loader, test_loader, train_dataset.classes
