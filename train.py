import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from tqdm import tqdm
import wandb

from cifar100_dataloader import CIFAR100Dataset
from vgg16 import VGG16

torch.manual_seed(123)
            
if __name__ == "__main__":
    print("Beginning training")
        
    # parameters
    epochs = 100
    batch_size = 32
    learning_rate = 0.01
    device = "cuda"
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    dataset_path = "cifar-100-python"
    train_dataset = CIFAR100Dataset(dataset_path, is_train=True)
    test_dataset = CIFAR100Dataset(dataset_path, is_train=False)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=32, 
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=batch_size, 
                                 shuffle=True, 
                                 num_workers=32, 
                                 pin_memory=True)
    dataset_meta = test_dataloader.dataset.meta
    
    class_n = len(dataset_meta)
          
    model = VGG16(class_n).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, 
                                weight_decay=0.0005)
    
    steps = len(train_dataloader)
    
    best_acc = 0.
    
    wandb.init(project="VGG16-CIFAR100")
    for epoch in range(epochs):
        model.train()
        
        total = 0
        correct = 0
        loss_accum = 0
        for step, (images, labels_gt) in enumerate(train_dataloader):
            images = images.to(device)
            labels_gt = labels_gt.to(device)
            
            output = model(images)
            loss = criterion(output, labels_gt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            labels_pred = torch.argmax(output.data, dim=1)
            total += labels_pred.size(0)
            correct += (labels_pred == labels_gt).sum().item()
                                   
            if step % 100 == 0:                       
                print ('Epoch [{}/{}], Steps [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, epochs, step, steps, loss.item()))
            
            wandb.log({"train/loss":loss.item(), "train/acc": 100 * correct / total})
            loss_accum += loss.item()
            
        wandb.log({"train/loss_avg": loss_accum / len(train_dataloader), "train/acc_avg": 100 * correct / total})
        
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            debug_images = []
            for images, labels_gt in test_dataloader:
                image_debugging = test_dataloader.dataset.tensor_to_image(images[0].clone())
                                
                images = images.to(device)
                labels_gt = labels_gt.to(device)
                
                output = model(images)
                
                labels_pred = torch.argmax(output.data, dim=1)
                
                total += labels_pred.size(0)
                correct += (labels_pred == labels_gt).sum().item()
                
                if len(debug_images) < 16:
                    gt_idx = labels_gt[0]
                    gt = dataset_meta[gt_idx].decode("utf-8")
                    
                    pred_idx = labels_pred[0]
                    pred = dataset_meta[pred_idx].decode("utf-8")
                    
                    debug_images.append(wandb.Image(image_debugging, caption=f"Pred:{pred}, Label:{gt}"))
                
            acc = 100 * correct / total
                    
            print("Accuracy of the network on the {} validation images: {} %"
                .format(len(test_dataset), acc)) 
            
            wandb.log({"test/acc_avg": acc, "test/examples": debug_images})
            
            if acc > best_acc:
                save_path = os.path.join(weights_dir, "best.pt")
                torch.save(model.state_dict(), save_path)
                best_acc = acc
                    
            save_path = os.path.join(weights_dir, "last.pt")
            torch.save(model.state_dict(), save_path)

            
