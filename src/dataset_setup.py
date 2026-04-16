import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

def get_dataloaders(batch_size=128, size=224):
    data_flag = 'octmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    data_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Caricamento Dataset
    train_dataset = DataClass(split='train', transform=data_transform, download=True, size=size)
    test_dataset = DataClass(split='test', transform=data_transform, download=True, size=size)
    val_dataset = DataClass(split='val', transform=data_transform, download=True, size=size)
    
    # Creazione Loader
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # Shuffle True per il train!
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader