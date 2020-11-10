import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
from CT_dataset import *
from UNet3D import *
from losses_and_metrics import *
import utils
import pandas as pd

if __name__ == '__main__':
    
    torch.cuda.set_device(utils.get_avail_gpu()) # assign which gpu will be used (only linux works)
    use_visdom = True
    
    train_list = './train_list.csv'
    val_list = './val_list.csv'
    
    model_path = './models'
    model_name = 'ALV_unet3d_patch64x64x64_1500_3labels_30samples' #remember to include the project title (e.g., ALV)
    checkpoint_name = 'latest_checkpoint.tar'
    
    num_classes = 3
    num_channels = 1
    num_epochs = 50 
    num_workers = 6
    train_batch_size = 6
    val_batch_size = 1
    num_batches_to_print = 200
    
    if use_visdom:
        # set plotter
        global plotter
        plotter = utils.VisdomLinePlotter(env_name=model_name)
    
    # mkdir 'models'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    
    # set dataset
    training_dataset = CT_Dataset(train_list)
    val_dataset = CT_Dataset(val_list)
    
    train_loader = DataLoader(dataset=training_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    
    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_channels=num_channels, out_channels=num_classes).to(device, dtype=torch.float)
    opt = optim.Adam(model.parameters(), lr=0.0001, amsgrad=True)
    #scheduler = StepLR(opt, step_size=2, gamma=0.8)
    
    losses, mdsc = [], []
    val_losses, val_mdsc = [], []
    
    best_val_dsc = 0.0
    
    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
        
    print('Training model...')
    class_weights = torch.Tensor([0.05, 1.0, 2.0]).to(device, dtype=torch.float)
    for epoch in range(num_epochs):

        # training
        model.train()
        running_loss = 0.0
        running_dsc = 0.0
        loss_epoch = 0.0
        dsc_epoch = 0.0        
        for i_batch, batched_sample in enumerate(train_loader):

            # send mini-batch to device            
            inputs, labels = batched_sample['image'].to(device, dtype=torch.float), batched_sample['label'].to(device, dtype=torch.long)
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :, :, :], num_classes=num_classes)
            one_hot_labels = one_hot_labels.permute(0, 4, 1, 2, 3)
            
            # zero the parameter gradients
            opt.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            loss.backward()
            opt.step()
            
            # print statistics
            running_loss += loss.item()
            running_dsc += dsc.item()
            loss_epoch += loss.item()
            dsc_epoch += dsc.item()
            if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                print('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print, running_dsc/num_batches_to_print))
                if use_visdom:
                    plotter.plot('loss', 'train', 'Loss', epoch+(i_batch+1)/len(train_loader), running_loss/num_batches_to_print)
                    plotter.plot('DSC', 'train', 'DSC', epoch+(i_batch+1)/len(train_loader), running_dsc/num_batches_to_print)
                running_loss = 0.0
                running_dsc = 0.0

        # record losses and dsc
        losses.append(loss_epoch/len(train_loader))
        mdsc.append(dsc_epoch/len(train_loader))
        
        #reset
        loss_epoch = 0.0
        dsc_epoch = 0.0
                
        # validation
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            running_val_dsc = 0.0
            val_loss_epoch = 0.0
            val_dsc_epoch = 0.0
            for i_batch, batched_val_sample in enumerate(val_loader):
                
                # send mini-batch to device
                inputs, labels = batched_val_sample['image'].to(device, dtype=torch.float), batched_val_sample['label'].to(device, dtype=torch.long)
                one_hot_labels = nn.functional.one_hot(labels[:, 0, :, :, :], num_classes=num_classes)
                one_hot_labels = one_hot_labels.permute(0, 4, 1, 2, 3)
                
                outputs = model(inputs)
                loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
                dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
                
                running_val_loss += loss.item()
                running_val_dsc += dsc.item()
                val_loss_epoch += loss.item()
                val_dsc_epoch += dsc.item()
                
                if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                    print('[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}, val_dsc: {5}'.format(epoch+1, num_epochs, i_batch+1, len(val_loader), running_val_loss/num_batches_to_print, running_val_dsc/num_batches_to_print))
                    running_val_loss = 0.0
                    running_val_dsc = 0.0
            
            # record losses and dsc
            val_losses.append(val_loss_epoch/len(val_loader))
            val_mdsc.append(val_dsc_epoch/len(val_loader))
            
            # reset
            val_loss_epoch = 0.0
            val_dsc_epoch = 0.0
            
            # output current status
            print('*****\nEpoch: {0}/{1}, loss: {2}, dsc: {3}\n         val_loss: {4}, val_dsc: {5}\n*****'.format(epoch+1, num_epochs, losses[-1], mdsc[-1], val_losses[-1], val_mdsc[-1]))
            if use_visdom:
                plotter.plot('loss', 'train', 'Loss', epoch+1, losses[-1])
                plotter.plot('DSC', 'train', 'DSC', epoch+1, mdsc[-1])
                plotter.plot('loss', 'val', 'Loss', epoch+1, val_losses[-1])
                plotter.plot('DSC', 'val', 'DSC', epoch+1, val_mdsc[-1])
            
        # save the checkpoint
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'losses': losses,
                    'mdsc': mdsc,
                    'val_losses': val_losses,
                    'val_mdsc': val_mdsc},
                    os.path.join(model_path, checkpoint_name))
        
        # save the best model
        if best_val_dsc < val_mdsc[-1]:
            best_val_dsc = val_mdsc[-1]
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'losses': losses,
                        'mdsc': mdsc,
                        'val_losses': val_losses,
                        'val_mdsc': val_mdsc},
                        os.path.join(model_path, '{}_best.tar'.format(model_name)))
            
        # save all losses and dsc data
        pd_dict = {'loss': losses, 'DSC': mdsc, 'val_loss': val_losses, 'val_DSC': val_mdsc}
        stat = pd.DataFrame(pd_dict)
        stat.to_csv('losses_dsc_vs_epoch.csv')
            
        # decay learning rate
        #scheduler.step()
