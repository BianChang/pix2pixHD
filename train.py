import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
from math import gcd

def lcm(a, b): return abs(a * b) // gcd(a, b) if a and b else 0


from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import csv
from tqdm import tqdm
from post_process import validation_train
from copy import deepcopy
from data.aligned_dataset import AlignedDataset

if __name__ == '__main__':
    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    opt.print_freq = lcm(opt.print_freq, opt.batch_size)
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10
    train_opt = deepcopy(opt)

    '''
    # Now, update opt for the validation phase without affecting train_opt
    opt.phase = 'val'
    opt.batchSize = 1  # Set the batch size to 1 for validation
    val_data_loader = CreateDataLoader(opt)
    val_dataset = val_data_loader.load_data()
    val_dataset_size = len(val_data_loader)
    print('#validation images = %d' % val_dataset_size)

    # Restore the original training configuration before continuing with training
    opt = train_opt
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    '''
    dataset_train = AlignedDataset(opt)
    n_train = len(dataset_train)
    dataset_size = len(dataset_train)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # Update opt for the validation phase
    opt.phase = 'val'
    dataset_validation = AlignedDataset(opt)
    n_val = len(dataset_validation)
    print('The number of validation images = %d' % n_val)

    dataset = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads))
    dataset_val = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=1,
        shuffle=False,
        num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = create_model(opt)
    model.to(device)

    visualizer = Visualizer(opt)
    if opt.fp16:
        from apex import amp
        model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    else:
        # optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
        # Directly access the optimizers from the model instance
        optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D

    total_steps = (start_epoch-1) * dataset_size + epoch_iter

    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    f = open('./checkpoints/' + '%s/' % opt.name + 'validation_train.csv', 'w', encoding='utf-8',
             newline='')  # record validation result
    csv_writer = csv.writer(f)
    csv_writer.writerow(['epoch', 'dapi', 'cd3', 'panck', 'average'])

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in tqdm(enumerate(dataset), total=len(dataset), desc="Training Epoch %d" % epoch):
            print('TRAINING')
            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            ############## Forward Pass ######################
            losses, generated = model(data['A'].to(device), data['inst'].to(device),
                                      data['B'].to(device), data['feat'].to(device), infer=save_fake)


            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            #loss_dict = dict(zip(model.module.loss_names, losses))
            loss_dict = dict(zip(model.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

            ############### Backward Pass ####################
            # update generator weights
            optimizer_G.zero_grad()
            if opt.fp16:
                with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()
            else:
                loss_G.backward()
            optimizer_G.step()

            # update discriminator weights
            optimizer_D.zero_grad()
            if opt.fp16:
                with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()
            else:
                loss_D.backward()
            optimizer_D.step()

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)
                #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

            ### display output images
            if save_fake:
                visuals = OrderedDict([('input_label', util.tensor2label(data['A'][0], opt.label_nc)),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(data['B'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                #model.module.save('latest')
                model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break

        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            #model.module.save('latest')
            #model.module.save(epoch)
            model.save('latest')
            model.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        if epoch % opt.val_freq == 0:
            if epoch % opt.val_freq == 0:  # run validation on the validation set
                dapi = 0
                cd3 = 0
                panck = 0
                average = 0
                with torch.no_grad():
                    for i, data_val in tqdm(enumerate(dataset_val), total=len(dataset_val),
                                            desc="Validating Epoch %d" % epoch):
                        print('VALIDATION')
                        imgs = data_val['A'].to(device)
                        truemasks = data_val['B'].to(device)
                        net = getattr(model, 'net' + 'G')
                        maskpred = net(imgs)
                        # maskpred = maskpred.cpu().numpy()
                        # truemasks = truemasks.cpu().numpy()
                        dapi_score, cd3_score, panck_score, average_score = validation_train(truemasks, maskpred)
                        dapi += dapi_score
                        cd3 += cd3_score
                        panck += panck_score
                        average += average_score
                    csv_writer.writerow([epoch, dapi / n_val, cd3 / n_val,
                                         panck / n_val, average / n_val])
            print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))


        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            #model.module.update_fixed_params()
            model.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            #model.module.update_learning_rate()
            model.update_learning_rate()
    f.close()