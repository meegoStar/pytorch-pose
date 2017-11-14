import os
import time
from tqdm import tqdm
import pandas as pd
import shutil
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from utils.datasets import jhmdb
from utils import pickle_tools
from utils.average_meter import AverageMeter
from utils.accuracy import accuracy


class TrainingFlow():
    def __init__(self, epochs, batch_size, lr, num_classes, model, num_workers,
                 train_dict_path='', test_dict_path='', jhmdb_rgb_root='',
                 checkpoint_name='', best_model_name='',
                 train_csv_name='', test_csv_name='', args=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = num_classes
        self.model = model
        self.num_workers = num_workers
        self.train_dict_path = train_dict_path
        self.test_dict_path = test_dict_path
        self.jhmdb_rgb_root = jhmdb_rgb_root
        self.checkpoint_name = checkpoint_name
        self.best_model_name = best_model_name
        self.train_csv_name = train_csv_name
        self.test_csv_name = test_csv_name
        self.args = args

        self.best_prec1 = 0
        self.start_epoch = 0
        self.video_top1 = 0
        self.video_top5 = 0

        self.prepare_training()

    def load_dicts(self):
        self.train_dict = pickle_tools.load_pickle(self.train_dict_path)
        self.test_dict = pickle_tools.load_pickle(self.test_dict_path)

    def prepare_datasets(self):
        self.train_set = jhmdb.JhmdbRgbData(self.train_dict,
                                            self.jhmdb_rgb_root,
                                            transform=transforms.Compose([
                                                transforms.Scale([256, 256]),
                                                transforms.ToTensor()]))

        self.test_set = jhmdb.JhmdbRgbData(self.test_dict,
                                           self.jhmdb_rgb_root,
                                           train=False,
                                           transform=transforms.Compose([
                                               transforms.Scale([256, 256]),
                                               transforms.ToTensor()]))

    def prepare_dataloaders(self):
        self.train_loader = DataLoader(self.train_set,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=self.num_workers)

        self.test_loader = DataLoader(self.test_set,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers)

    def set_loss_function(self):
        self.criterion = nn.CrossEntropyLoss().cuda()

    def set_optimizer(self):
        if self.model.module.freeze_pose_feature_net:
            params_to_optimize = self.model.module.pose_resnet.parameters()
        else:
            params_to_optimize = self.model.parameters()

        self.optimizer = Adam(params_to_optimize, lr=self.lr)
        #self.optimizer = SGD(params_to_optimize, lr=self.lr, momentum=0.9)

    def set_scheduler(self):
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=0, verbose=True)

    def resume(self):
        args = self.args
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, self.start_epoch))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

    def prepare_training(self):
        self.load_dicts()
        self.prepare_datasets()
        self.prepare_dataloaders()
        self.set_loss_function()
        self.set_optimizer()
        self.set_scheduler()
        self.resume()

    def display_epoch_info(self, train=True):
        print('****' * 40)

        if train:
            print('Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.epochs))
        else:
            print('Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.epochs))

        print('****' * 40)

    def initialize_statistic(self, period_scale=50):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

        self.batch_start_time = time.time()
        self.progress = tqdm(self.current_data_loader)

        self.batch_info_period = len(self.current_data_loader) / period_scale

    def initialize_train_epoch(self):
        self.display_epoch_info(train=True)
        self.current_data_loader = self.train_loader
        self.initialize_statistic()
        self.model.train() # switch to train mode

    def initialize_val_epoch(self):
        self.display_epoch_info(train=False)
        self.current_data_loader = self.test_loader
        self.initialize_statistic()

        self.video_scores = {}  # container for video level accuracy computation

        self.model.eval() # switch to evaluate mode

    def measure_data_time(self):
        self.data_time.update(time.time() - self.batch_start_time)

    def update_video_predictions(self, video_name_batch, output_batch, label_batch):
        for i in range(0, len(label_batch)):
            video_name = video_name_batch[i]
            score = output_batch.data[i, :]
            label = label_batch[i]

            if video_name not in self.video_scores:
                self.video_scores[video_name] = {}
                self.video_scores[video_name]['label'] = label

            if 'score' not in self.video_scores[video_name]:
                self.video_scores[video_name]['score'] = score
            else:
                self.video_scores[video_name]['score'] += score

    def compute_accuracy(self, data, labels, output):
        prec1, prec5 = accuracy(output.data, labels, topk=(1, 5))
        self.top1.update(prec1[0], data.size(0))
        self.top5.update(prec5[0], data.size(0))

    def record_losses(self, data, loss):
        self.losses.update(loss.data[0], data.size(0))

    def measure_batch_time(self):
        self.batch_time.update(time.time() - self.batch_start_time)
        self.batch_start_time = time.time()

    def print_batch_info(self, train=True):
        if train:
            phase_str = 'Training'
            data_time_info = 'Data loading {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(data_time=self.data_time)
        else:
            phase_str = 'Testing'
            data_time_info = ''

        core_info = ('Epoch: [{0}], {1}[{2}/{3}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(self.epoch,
                                                                     phase_str,
                                                                     self.iteration_count,
                                                                     len(self.current_data_loader),
                                                                     loss=self.losses,
                                                                     top1=self.top1,
                                                                     top5=self.top5))

        batch_time_info = 'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=self.batch_time)

        result_info = '\n'.join([core_info, batch_time_info, data_time_info])
        print result_info
        print '----' * 40

    def save_batch_info(self, train=True):
        prog = ' '.join([str(round((float(self.iteration_count) / float(len(self.current_data_loader))), 2) * 100), '%'])
        column_names = ['Epoch',
                        'Progress',
                        'Loss',
                        'Prec@1',
                        'Prec@5',
                        'Batch Time']
        info_dict = {'Epoch': [self.epoch],
                     'Progress': [prog],
                     'Loss': [self.losses.avg],
                     'Prec@1': [self.top1.avg],
                     'Prec@5': [self.top5.avg],
                     'Batch Time': [round(self.batch_time.avg, 3)]}

        if train:
            target_csv_name = self.train_csv_name

            column_names.append('Data Time')
            info_dict['Data Time'] = [round(self.data_time.avg, 3)]
        else:
            target_csv_name = self.test_csv_name

            column_names.append('Video prec@1')
            info_dict['Video prec@1'] = [round(self.video_top1, 3)]

            column_names.append('Video prec@5')
            info_dict['Video prec@5'] = [round(self.video_top5, 3)]

        df = pd.DataFrame.from_dict(info_dict)

        if not os.path.isfile(target_csv_name):
            df.to_csv(target_csv_name, index=False, columns=column_names)
        else: # else it exists so append without writing the header
            df.to_csv(target_csv_name, mode='a', header=False, index=False, columns=column_names)

    def present_batch_info(self, train=True):
        if (self.iteration_count + 1) % self.batch_info_period == 0:
            self.print_batch_info(train=train)
            self.save_batch_info(train=train) # save the information to training.csv file

    def train_one_epoch(self):
        self.initialize_train_epoch()

        for self.iteration_count, (data_batch, label_batch) in enumerate(self.progress):
            self.measure_data_time() # measure batch data loading time

            label_batch = label_batch.cuda(async=True)
            data_batch_var = Variable(data_batch).cuda()
            label_batch_var = Variable(label_batch).cuda()

            # compute output batch
            output_batch = self.model(data_batch_var)
            loss = self.criterion(output_batch, label_batch_var)

            # measure accuracy and record loss
            self.compute_accuracy(data_batch, label_batch, output_batch)
            self.record_losses(data_batch, loss)

            # compute gradient and let optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.measure_batch_time() # measure elapsed time
            self.present_batch_info(train=True) # display and record batch info

    def validate_one_epoch(self):
        self.initialize_val_epoch()

        for self.iteration_count, (video_name_batch, data_batch, label_batch) in enumerate(self.progress):
            label_batch = label_batch.cuda(async=True)
            data_batch_var = Variable(data_batch, volatile=True).cuda(async=True)
            label_batch_var = Variable(label_batch, volatile=True).cuda(async=True)

            # compute output batch
            output_batch = self.model(data_batch_var)
            loss = self.criterion(output_batch, label_batch_var)

            # measure accuracy and record loss
            self.compute_accuracy(data_batch, label_batch, output_batch)
            self.record_losses(data_batch, loss)

            self.measure_batch_time() # measure elapsed time
            self.present_batch_info(train=False) # display and record batch info

            # update video level predictions
            self.update_video_predictions(video_name_batch, output_batch, label_batch)

        print(' * Prec@1: {top1.avg:.3f} Prec@5: {top5.avg:.3f} Loss: {loss.avg:.4f} '.format(top1=self.top1,
                                                                                              top5=self.top5,
                                                                                              loss=self.losses))
        return self.top1.avg, self.losses.avg

    def compute_video_accuracy(self):
        correct_count_top1 = 0.
        correct_count_top5 = 0.
        for video_name in self.video_scores.keys():
            video_label = self.video_scores[video_name]['label']
            video_score = self.video_scores[video_name]['score']
            video_score_numpy = video_score.cpu().numpy()

            prediction_top1 = list(np.argsort(video_score_numpy)[-1:])
            if video_label in prediction_top1:
                correct_count_top1 += 1

            prediction_top5 = list(np.argsort(video_score_numpy)[-5:])
            if video_label in prediction_top5:
                correct_count_top5 += 1

        self.video_top1 = correct_count_top1 / len(self.video_scores)
        self.video_top5 = correct_count_top5 / len(self.video_scores)

        print '-' * 60
        print(' * Video-prec@1: {top1:.3f} Video-prec@5: {top5:.3f} '.format(top1=self.video_top1,
                                                                             top5=self.video_top5))
        print '-' * 60

    def save_checkpoint(self):
        checkpoint_name = self.checkpoint_name
        state = {'epoch': self.epoch,
                 'arch': 'pose-feature + resnet18',
                 'state_dict': self.model.state_dict(),
                 'best_prec1': self.best_prec1,
                 'frame_prec5': self.top5.avg,
                 'video_prec1': self.video_top1,
                 'video_prec5': self.video_top5,
                 'optimizer' : self.optimizer.state_dict()}
        torch.save(state, checkpoint_name)

        if self.is_best:
            shutil.copyfile(checkpoint_name, self.best_model_name)

    def train(self):
        epochs = self.epochs
        for self.epoch in range(self.start_epoch + 1, epochs + 1):
            self.train_one_epoch() # train for one epoch
            val_prec1, val_loss = self.validate_one_epoch() # evaluate on validation set

            # compute video level accuracy
            self.compute_video_accuracy()

            self.scheduler.step(val_prec1) # call lr_scheduler

            # record best prec@1 and save checkpoint
            self.is_best = val_prec1 > self.best_prec1
            self.best_prec1 = max(val_prec1, self.best_prec1)
            self.save_checkpoint()
