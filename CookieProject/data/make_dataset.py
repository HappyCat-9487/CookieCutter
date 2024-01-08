import torch
import os
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

def load_corruptmnist():
    data_datasets = []
    
    for i in range(6):
        train_images = torch.load(f'/Users/luchengliang/MLO/CookieProject/data/raw/train_images_{i}.pt')
        train_targets = torch.load(f'/Users/luchengliang/MLO/CookieProject/data/raw/train_target_{i}.pt')
        data_datasets.append(TensorDataset(train_images, train_targets))
        
    test_images = torch.load(f'/Users/luchengliang/MLO/CookieProject/data/raw/test_images.pt')
    test_targets= torch.load(f'/Users/luchengliang/MLO/CookieProject/data/raw/test_target.pt')
    data_datasets.append(TensorDataset(test_images, test_targets))
    
    combined_data = ConcatDataset(data_datasets)
    
    return combined_data

def normalize_data(data):
    all_data = torch.cat([sample[0] for sample in data], dim=0)
    mean = torch.mean(all_data)
    std = torch.std(all_data)
    
    normalize_data = (all_data - mean) / std
    
    return normalize_data

def save_processed_data(normalized_data, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    torch.save(normalized_data, os.path.join(output_folder, 'processed_corruptmnist.pt'))

def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    train_datasets = []
    test_datasets = []
    
    for i in range(6):
        train_images = torch.load(f'/Users/luchengliang/MLO/CookieProject/data/raw/train_images_{i}.pt')
        train_targets = torch.load(f'/Users/luchengliang/MLO/CookieProject/data/raw/train_target_{i}.pt')
        train_datasets.append(TensorDataset(train_images, train_targets))
        
    test_images = torch.load(f'/Users/luchengliang/MLO/CookieProject/data/raw/test_images.pt')
    test_targets= torch.load(f'//Users/luchengliang/MLO/CookieProject/data/raw/test_target.pt')
    test_datasets.append(TensorDataset(test_images, test_targets))
    
    train_dataset = ConcatDataset(train_datasets)
    test_datasets = ConcatDataset(test_datasets)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_datasets,batch_size=64, shuffle=False)
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Get the data and process it
    corruptmnist_data = load_corruptmnist()
    normalized_data = normalize_data(corruptmnist_data)
    save_processed_data(normalized_data, '/Users/luchengliang/MLO/CookieProject/data/processed')