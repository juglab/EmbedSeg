import os
import shutil

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from EmbedSeg.criterions import get_loss
from EmbedSeg.datasets import get_dataset
from EmbedSeg.models import get_model
from EmbedSeg.utils.utils import AverageMeter, Cluster, Cluster_3d, Logger, Visualizer, prepare_embedding_for_train_image

torch.backends.cudnn.benchmark = True
from matplotlib.colors import ListedColormap
import numpy as np


# https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
def train(virtual_batch_multiplier, one_hot, n_sigma, args):
    # define meters
    loss_meter = AverageMeter()
    # put model into training mode
    model.train()
    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    optimizer.zero_grad()  # Reset gradients tensors
    for i, sample in enumerate(tqdm(train_dataset_it)):
        im = sample['image']
        instances = sample['instance'].squeeze(1)
        class_labels = sample['label'].squeeze(1)
        center_images = sample['center_image'].squeeze(1)
        output = model(im)  # Forward pass
        loss = criterion(output, instances, class_labels, center_images, **args)
        loss = loss / virtual_batch_multiplier  # Normalize our loss (if averaged)
        loss = loss.mean()
        loss.backward()  # Backward pass
        if (i + 1) % virtual_batch_multiplier == 0:  # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step
            optimizer.zero_grad()  # Reset gradients tensors
        loss_meter.update(loss.item())
    return loss_meter.avg * virtual_batch_multiplier


def train_vanilla(display, display_embedding, display_it, one_hot, grid_x, grid_y, pixel_x, pixel_y, n_sigma,
                  args):  # this is without virtual batches!

    # define meters
    loss_meter = AverageMeter()
    # put model into training mode
    model.train()

    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))
    for i, sample in enumerate(tqdm(train_dataset_it)):

        im = sample['image']
        instances = sample['instance'].squeeze(1)  # 1YX (not one-hot) or 1DYX (one-hot)
        class_labels = sample['label'].squeeze(1)  # 1YX
        center_images = sample['center_image'].squeeze(1)  # 1YX
        output = model(im)  # B 5 Y X
        loss = criterion(output, instances, class_labels, center_images, **args)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        if display and i % display_it == 0:
            with torch.no_grad():
                visualizer.display(im[0], key='image', title='Image')
                predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=n_sigma)
                if one_hot:
                    instance = invert_one_hot(instances[0].cpu().detach().numpy())
                    visualizer.display(instance, key='groundtruth', title='Ground Truth')  # TODO
                    instance_ids = np.arange(instances.size(1)) # instances[0] --> DYX
                else:
                    visualizer.display(instances[0].cpu(), key='groundtruth', title='Ground Truth')  # TODO
                    instance_ids = instances[0].unique()
                    instance_ids = instance_ids[instance_ids != 0]

                if display_embedding:
                    center_x, center_y, samples_x, samples_y, sample_spatial_embedding_x, \
                    sample_spatial_embedding_y, sigma_x, sigma_y, color_sample_dic, color_embedding_dic = \
                        prepare_embedding_for_train_image(one_hot= one_hot, grid_x=grid_x, grid_y=grid_y, pixel_x=pixel_x, pixel_y=pixel_y,
                                                          predictions=predictions, instance_ids=instance_ids,
                                                          center_images=center_images,
                                                          output=output, instances=instances, n_sigma=n_sigma)
                    if one_hot:
                        visualizer.display(torch.max(instances[0], dim =0)[0], key='center', title='Center', center_x=center_x,
                                           center_y=center_y,
                                           samples_x=samples_x, samples_y=samples_y,
                                           sample_spatial_embedding_x=sample_spatial_embedding_x,
                                           sample_spatial_embedding_y=sample_spatial_embedding_y,
                                           sigma_x=sigma_x, sigma_y=sigma_y,
                                           color_sample=color_sample_dic, color_embedding=color_embedding_dic)
                    else:
                        visualizer.display(instances[0] > 0, key='center', title='Center', center_x=center_x,
                                           center_y=center_y,
                                           samples_x=samples_x, samples_y=samples_y,
                                           sample_spatial_embedding_x=sample_spatial_embedding_x,
                                           sample_spatial_embedding_y=sample_spatial_embedding_y,
                                           sigma_x=sigma_x, sigma_y=sigma_y,
                                           color_sample=color_sample_dic, color_embedding=color_embedding_dic)
                visualizer.display(predictions.cpu(), key='prediction', title='Prediction')  # TODO

    return loss_meter.avg



# https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
def train_3d(virtual_batch_multiplier, one_hot, n_sigma, args):
    # define meters
    loss_meter = AverageMeter()
    # put model into training mode
    model.train()
    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    optimizer.zero_grad()  # Reset gradients tensors
    for i, sample in enumerate(tqdm(train_dataset_it)):
        im = sample['image']
        instances = sample['instance'].squeeze(1)
        class_labels = sample['label'].squeeze(1)
        center_images = sample['center_image'].squeeze(1)
        output = model(im)  # Forward pass


        loss = criterion(output, instances, class_labels, center_images, **args)
        loss = loss / virtual_batch_multiplier  # Normalize our loss (if averaged)
        loss = loss.mean()
        loss.backward()  # Backward pass
        if (i + 1) % virtual_batch_multiplier == 0:  # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step
            optimizer.zero_grad()  # Reset gradients tensors
        loss_meter.update(loss.item())
    return loss_meter.avg * virtual_batch_multiplier


def train_vanilla_3d(display, display_embedding, display_it, one_hot, grid_x, grid_y, grid_z, pixel_x, pixel_y, pixel_z, n_sigma,
                  args):  # this is without virtual batches!

    # define meters
    loss_meter = AverageMeter()
    # put model into training mode
    model.train()

    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))


    for i, sample in enumerate(tqdm(train_dataset_it)):

        im = sample['image'] # BCZYX
        instances = sample['instance'].squeeze(1)  # BZYX
        class_labels = sample['label'].squeeze(1)  # BZYX
        center_images = sample['center_image'].squeeze(1)  # BZYX
        output = model(im)  # B 7 Z Y X
        loss = criterion(output, instances, class_labels, center_images, **args)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        if display and i % display_it == 0:
            with torch.no_grad():
                visualizer.display(im[0], key='image', title='Image')
                predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=n_sigma)
                if one_hot:
                    instance = invert_one_hot(instances[0].cpu().detach().numpy())
                    visualizer.display(instance, key='groundtruth', title='Ground Truth')  # TODO
                    instance_ids = np.arange(instances.size(1))  # instances[0] --> DYX
                else:
                    visualizer.display(instances[0].cpu(), key='groundtruth', title='Ground Truth')  # TODO
                    instance_ids = instances[0].unique()
                    instance_ids = instance_ids[instance_ids != 0]

                if display_embedding:
                    center_x, center_y, samples_x, samples_y, sample_spatial_embedding_x, \
                    sample_spatial_embedding_y, sigma_x, sigma_y, color_sample_dic, color_embedding_dic = \
                        prepare_embedding_for_train_image(one_hot=one_hot, grid_x=grid_x, grid_y=grid_y,
                                                          pixel_x=pixel_x, pixel_y=pixel_y,
                                                          predictions=predictions, instance_ids=instance_ids,
                                                          center_images=center_images,
                                                          output=output, instances=instances, n_sigma=n_sigma)
                    if one_hot:
                        visualizer.display(torch.max(instances[0], dim=0)[0], key='center', title='Center',
                                           center_x=center_x,
                                           center_y=center_y,
                                           samples_x=samples_x, samples_y=samples_y,
                                           sample_spatial_embedding_x=sample_spatial_embedding_x,
                                           sample_spatial_embedding_y=sample_spatial_embedding_y,
                                           sigma_x=sigma_x, sigma_y=sigma_y,
                                           color_sample=color_sample_dic, color_embedding=color_embedding_dic)
                    else:
                        visualizer.display(instances[0] > 0, key='center', title='Center', center_x=center_x,
                                           center_y=center_y,
                                           samples_x=samples_x, samples_y=samples_y,
                                           sample_spatial_embedding_x=sample_spatial_embedding_x,
                                           sample_spatial_embedding_y=sample_spatial_embedding_y,
                                           sigma_x=sigma_x, sigma_y=sigma_y,
                                           color_sample=color_sample_dic, color_embedding=color_embedding_dic)
                visualizer.display(predictions.cpu(), key='prediction', title='Prediction')  # TODO

    return loss_meter.avg





def val(virtual_batch_multiplier, one_hot, n_sigma, args):
    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    # put model into eval mode
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):
            im = sample['image']
            instances = sample['instance'].squeeze(1)
            class_labels = sample['label'].squeeze(1)
            center_images = sample['center_image'].squeeze(1)
            output = model(im)
            loss = criterion(output, instances, class_labels, center_images, **args, iou=True, iou_meter=iou_meter)
            loss = loss.mean()
            loss = loss / virtual_batch_multiplier
            loss_meter.update(loss.item())

    return loss_meter.avg * virtual_batch_multiplier, iou_meter.avg


def val_vanilla(display, display_embedding, display_it, one_hot, grid_x, grid_y, pixel_x, pixel_y, n_sigma, args):
    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    # put model into eval mode
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):
            im = sample['image']
            instances = sample['instance'].squeeze(1)
            class_labels = sample['label'].squeeze(1)
            center_images = sample['center_image'].squeeze(1)
            output = model(im)
            loss = criterion(output, instances, class_labels, center_images, **args, iou=True, iou_meter=iou_meter)
            loss = loss.mean()
            if display and i % display_it == 0:
                with torch.no_grad():
                    visualizer.display(im[0], key='image', title='Image')
                    predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=n_sigma)
                    if one_hot:
                        instance = invert_one_hot(instances[0].cpu().detach().numpy())
                        visualizer.display(instance, key='groundtruth', title='Ground Truth')  # TODO
                        instance_ids = np.arange(instances[0].size(1))
                    else:
                        visualizer.display(instances[0].cpu(), key='groundtruth', title='Ground Truth')  # TODO
                        instance_ids = instances[0].unique()
                        instance_ids = instance_ids[instance_ids != 0]
                    if (display_embedding):
                        center_x, center_y, samples_x, samples_y, sample_spatial_embedding_x, \
                        sample_spatial_embedding_y, sigma_x, sigma_y, color_sample_dic, color_embedding_dic = \
                            prepare_embedding_for_train_image(one_hot = one_hot, grid_x=grid_x, grid_y=grid_y, pixel_x=pixel_x, pixel_y=pixel_y,
                                                              predictions=predictions, instance_ids=instance_ids,
                                                              center_images=center_images,
                                                              output=output, instances=instances, n_sigma=n_sigma)
                        if one_hot:
                            visualizer.display(torch.max(instances[0], dim=0)[0].cpu(), key='center', title='Center', # torch.max returns a tuple
                                               center_x=center_x,
                                               center_y=center_y,
                                               samples_x=samples_x, samples_y=samples_y,
                                               sample_spatial_embedding_x=sample_spatial_embedding_x,
                                               sample_spatial_embedding_y=sample_spatial_embedding_y,
                                               sigma_x=sigma_x, sigma_y=sigma_y,
                                               color_sample=color_sample_dic, color_embedding=color_embedding_dic)
                        else:
                            visualizer.display(instances[0] > 0, key='center', title='Center', center_x=center_x,
                                               center_y=center_y,
                                               samples_x=samples_x, samples_y=samples_y,
                                               sample_spatial_embedding_x=sample_spatial_embedding_x,
                                               sample_spatial_embedding_y=sample_spatial_embedding_y,
                                               sigma_x=sigma_x, sigma_y=sigma_y,
                                               color_sample=color_sample_dic, color_embedding=color_embedding_dic)

                    visualizer.display(predictions.cpu(), key='prediction', title='Prediction')  # TODO

            loss_meter.update(loss.item())

    return loss_meter.avg, iou_meter.avg



def val_3d(virtual_batch_multiplier, one_hot, n_sigma, args):
    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    # put model into eval mode
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):
            im = sample['image']
            instances = sample['instance'].squeeze(1)
            class_labels = sample['label'].squeeze(1)
            center_images = sample['center_image'].squeeze(1)
            output = model(im)
            loss = criterion(output, instances, class_labels, center_images, **args, iou=True, iou_meter=iou_meter)
            loss = loss.mean()
            loss = loss / virtual_batch_multiplier
            loss_meter.update(loss.item())

    return loss_meter.avg * virtual_batch_multiplier, iou_meter.avg


def val_vanilla_3d(display, display_embedding, display_it, one_hot, grid_x, grid_y, grid_z, pixel_x, pixel_y, pixel_z, n_sigma, args):
    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    # put model into eval mode
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):
            im = sample['image'] # BCZYX
            instances = sample['instance'].squeeze(1) # BZYX
            class_labels = sample['label'].squeeze(1) # BZYX
            center_images = sample['center_image'].squeeze(1) # BZYX
            output = model(im)
            loss = criterion(output, instances, class_labels, center_images, **args, iou=True, iou_meter=iou_meter)
            loss = loss.mean()
            if display and i % display_it == 0:
                with torch.no_grad():
                    visualizer.display(im[0], key='image', title='Image')
                    predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=n_sigma)
                    if one_hot:
                        instance = invert_one_hot(instances[0].cpu().detach().numpy())
                        visualizer.display(instance, key='groundtruth', title='Ground Truth')  # TODO
                        instance_ids = np.arange(instances[0].size(1))
                    else:
                        visualizer.display(instances[0].cpu(), key='groundtruth', title='Ground Truth')  # TODO
                        instance_ids = instances[0].unique()
                        instance_ids = instance_ids[instance_ids != 0]
                    if (display_embedding):
                        center_x, center_y, samples_x, samples_y, sample_spatial_embedding_x, \
                        sample_spatial_embedding_y, sigma_x, sigma_y, color_sample_dic, color_embedding_dic = \
                            prepare_embedding_for_train_image(one_hot=one_hot, grid_x=grid_x, grid_y=grid_y,
                                                              pixel_x=pixel_x, pixel_y=pixel_y,
                                                              predictions=predictions, instance_ids=instance_ids,
                                                              center_images=center_images,
                                                              output=output, instances=instances, n_sigma=n_sigma)
                        if one_hot:
                            visualizer.display(torch.max(instances[0], dim=0)[0].cpu(), key='center', title='Center',
                                               # torch.max returns a tuple
                                               center_x=center_x,
                                               center_y=center_y,
                                               samples_x=samples_x, samples_y=samples_y,
                                               sample_spatial_embedding_x=sample_spatial_embedding_x,
                                               sample_spatial_embedding_y=sample_spatial_embedding_y,
                                               sigma_x=sigma_x, sigma_y=sigma_y,
                                               color_sample=color_sample_dic, color_embedding=color_embedding_dic)
                        else:
                            visualizer.display(instances[0] > 0, key='center', title='Center', center_x=center_x,
                                               center_y=center_y,
                                               samples_x=samples_x, samples_y=samples_y,
                                               sample_spatial_embedding_x=sample_spatial_embedding_x,
                                               sample_spatial_embedding_y=sample_spatial_embedding_y,
                                               sigma_x=sigma_x, sigma_y=sigma_y,
                                               color_sample=color_sample_dic, color_embedding=color_embedding_dic)

                    visualizer.display(predictions.cpu(), key='prediction', title='Prediction')  # TODO

            loss_meter.update(loss.item())

    return loss_meter.avg, iou_meter.avg




def invert_one_hot(image):
    instance = np.zeros((image.shape[1], image.shape[2]), dtype="uint16")
    for z in range(image.shape[0]):
        instance = np.where(image[z] > 0, instance + z + 1, instance)  # TODO - not completely accurate!
    return instance


def save_checkpoint(state, is_best, epoch, save_dir, name='checkpoint.pth'):
    print('=> saving checkpoint')
    file_name = os.path.join(save_dir, name)
    torch.save(state, file_name)
    if (epoch % 10 == 0):
        file_name2 = os.path.join(save_dir, str(epoch) + "_" + name)
        torch.save(state, file_name2)
    if is_best:
        shutil.copyfile(file_name, os.path.join(
            save_dir, 'best_iou_model.pth'))


def begin_training(train_dataset_dict, val_dataset_dict, model_dict, loss_dict, configs, color_map='magma'):


    if configs['save']:
        if not os.path.exists(configs['save_dir']):
            os.makedirs(configs['save_dir'])

    if configs['display']:
        plt.ion()
    else:
        plt.ioff()
        plt.switch_backend("agg")


    # set device
    device = torch.device("cuda:0" if configs['cuda'] else "cpu")

    # define global variables
    global train_dataset_it, val_dataset_it, model, criterion, optimizer, visualizer, cluster

    # train dataloader


    train_dataset = get_dataset(train_dataset_dict['name'], train_dataset_dict['kwargs'])
    train_dataset_it = torch.utils.data.DataLoader(train_dataset, batch_size=train_dataset_dict['batch_size'],
                                                   shuffle=True, drop_last=True,
                                                   num_workers=train_dataset_dict['workers'],
                                                   pin_memory=True if configs['cuda'] else False)

    # val dataloader
    val_dataset = get_dataset(val_dataset_dict['name'], val_dataset_dict['kwargs'])
    val_dataset_it = torch.utils.data.DataLoader(val_dataset, batch_size=val_dataset_dict['batch_size'], shuffle=True,
                                                 drop_last=True, num_workers=val_dataset_dict['workers'],
                                                 pin_memory=True if configs['cuda'] else False)

    # set model
    model = get_model(model_dict['name'], model_dict['kwargs'])
    model.init_output(loss_dict['lossOpts']['n_sigma'])
    model = torch.nn.DataParallel(model).to(device)


    if (configs['grid_z'] is None):
        criterion = get_loss(grid_z= None, grid_y = configs['grid_y'], grid_x = configs['grid_x'], pixel_z=None,
                             pixel_y = configs['pixel_y'], pixel_x = configs['pixel_x'],
                             one_hot = configs['one_hot'], loss_opts= loss_dict['lossOpts'])
    else:
        criterion = get_loss(configs['grid_z'], configs['grid_y'], configs['grid_x'],
                             configs['pixel_z'], configs['pixel_y'], configs['pixel_x'],
                             configs['one_hot'], loss_dict['lossOpts'])
    criterion = torch.nn.DataParallel(criterion).to(device)


    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['train_lr'], weight_decay=1e-4)



    def lambda_(epoch):
        return pow((1 - ((epoch) / 200)), 0.9)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_, )

    if(configs['grid_z'] is None):
        # clustering
        cluster = Cluster(configs['grid_y'], configs['grid_x'], configs['pixel_y'], configs['pixel_x'],
                          configs['one_hot'])
    else:
        # clustering
        cluster = Cluster_3d(configs['grid_z'], configs['grid_y'], configs['grid_x'], configs['pixel_z'], configs['pixel_y'],
                          configs['pixel_x'], configs['one_hot'])

    # Visualizer

    
    visualizer = Visualizer(('image', 'groundtruth', 'prediction', 'center'), color_map)  # 5 keys

    # Logger
    logger = Logger(('train', 'val', 'iou'), 'loss')



    # resume
    start_epoch = 0
    best_iou = 0
    if configs['resume_path'] is not None and os.path.exists(configs['resume_path']):
        print('Resuming model from {}'.format(configs['resume_path']))
        state = torch.load(configs['resume_path'])
        start_epoch = state['epoch'] + 1
        best_iou = state['best_iou']
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optim_state_dict'])
        logger.data = state['logger_data']


    for epoch in range(start_epoch, configs['n_epochs']):

        print('Starting epoch {}'.format(epoch))
        scheduler.step(epoch)

        if (configs['grid_z'] is None):
            if (train_dataset_dict['virtual_batch_multiplier'] > 1):
                train_loss = train(virtual_batch_multiplier=train_dataset_dict['virtual_batch_multiplier'],
                                   one_hot=configs['one_hot'],
                                   n_sigma=loss_dict['lossOpts']['n_sigma'], args=loss_dict['lossW'])
            elif (train_dataset_dict['virtual_batch_multiplier'] == 1):
                train_loss = train_vanilla(display=configs['display'], display_embedding=configs['display_embedding'],
                                           display_it=configs['display_it'], one_hot=configs['one_hot'],
                                           n_sigma=loss_dict['lossOpts']['n_sigma'], grid_x=configs['grid_x'],
                                           grid_y=configs['grid_y'],
                                           pixel_x=configs['pixel_x'], pixel_y=configs['pixel_y'],
                                           args=loss_dict['lossW'])

            if (val_dataset_dict['virtual_batch_multiplier'] > 1):
                val_loss, val_iou = val(virtual_batch_multiplier=val_dataset_dict['virtual_batch_multiplier'],
                                        one_hot=configs['one_hot'],
                                        n_sigma=loss_dict['lossOpts']['n_sigma'], args=loss_dict['lossW'])
            elif (val_dataset_dict['virtual_batch_multiplier'] == 1):
                val_loss, val_iou = val_vanilla(display=configs['display'], display_embedding=configs['display_embedding'],
                                                display_it=configs['display_it'], one_hot=configs['one_hot'],
                                                n_sigma=loss_dict['lossOpts']['n_sigma'], grid_x=configs['grid_x'],
                                                grid_y=configs['grid_y'], pixel_x=configs['pixel_x'],
                                                pixel_y=configs['pixel_y'],
                                                args=loss_dict['lossW'])
        else:
            if (train_dataset_dict['virtual_batch_multiplier'] > 1):
                train_loss = train_3d(virtual_batch_multiplier=train_dataset_dict['virtual_batch_multiplier'],
                                      one_hot=configs['one_hot'],
                                      n_sigma=loss_dict['lossOpts']['n_sigma'], args=loss_dict['lossW'])
            elif (train_dataset_dict['virtual_batch_multiplier'] == 1):
                train_loss = train_vanilla_3d(display=configs['display'],
                                              display_embedding=configs['display_embedding'],
                                              display_it=configs['display_it'], one_hot=configs['one_hot'],
                                              n_sigma=loss_dict['lossOpts']['n_sigma'], grid_x=configs['grid_x'],
                                              grid_y=configs['grid_y'], grid_z=configs['grid_z'],
                                              pixel_x=configs['pixel_x'], pixel_y=configs['pixel_y'],
                                              pixel_z=configs['pixel_z'], args=loss_dict['lossW'])

            if (val_dataset_dict['virtual_batch_multiplier'] > 1):
                val_loss, val_iou = val_3d(virtual_batch_multiplier=val_dataset_dict['virtual_batch_multiplier'],
                                           one_hot=configs['one_hot'],
                                           n_sigma=loss_dict['lossOpts']['n_sigma'], args=loss_dict['lossW'])
            elif (val_dataset_dict['virtual_batch_multiplier'] == 1):
                val_loss, val_iou = val_vanilla_3d(display=configs['display'],
                                                   display_embedding=configs['display_embedding'],
                                                   display_it=configs['display_it'], one_hot=configs['one_hot'],
                                                   n_sigma=loss_dict['lossOpts']['n_sigma'], grid_x=configs['grid_x'],
                                                   grid_y=configs['grid_y'], grid_z=configs['grid_z'],
                                                   pixel_x=configs['pixel_x'], pixel_y=configs['pixel_y'], pixel_z=configs['pixel_z'],
                                                   args=loss_dict['lossW'])



        print('===> train loss: {:.2f}'.format(train_loss))
        print('===> val loss: {:.2f}, val iou: {:.2f}'.format(val_loss, val_iou))

        logger.add('train', train_loss)
        logger.add('val', val_loss)
        logger.add('iou', val_iou)
        logger.plot(save=configs['save'], save_dir=configs['save_dir'])  # TODO

        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)

        if configs['save']:
            state = {
                'epoch': epoch,
                'best_iou': best_iou,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'logger_data': logger.data,
            }
        save_checkpoint(state, is_best, epoch, save_dir=configs['save_dir'])


