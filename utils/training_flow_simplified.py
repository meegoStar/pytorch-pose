from __future__ import print_function

from tqdm import tqdm
import shutil
import pandas as pd
import os
import numpy as np

import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.models as models


class TrainingFlowSimplified():
    def __init__(self, model=None, params_to_optimize=None, loss_function=None, compute_batch_accuracy=None,    
                 optimizer=None, train_loader=None, test_loader=None,
                 epochs=200, saturate_patience=20, reduce_patience=5, freeze_pose_feature_net=None,
                 csv_log_name='', checkpoint_name='', best_model_name='', arch='', args=None):
        self.model = model
        self.params_to_optimize = params_to_optimize
        self.criterion = loss_function
        self.compute_batch_accuracy = compute_batch_accuracy

        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.epochs = epochs
        self.saturate_patience = saturate_patience
        self.reduce_patience = reduce_patience
        self.freeze_pose_feature_net = freeze_pose_feature_net

        self.csv_log_name = csv_log_name
        self.checkpoint_name = checkpoint_name
        self.best_model_name = best_model_name
        self.arch = arch
        self.args = args

        self.start_epoch = 1
        self.best_val_acc = 0.0
        self.best_video_val_acc = 0.0
        self.saturate_count = 0
        self.video_top1 = 0
        self.video_top5 = 0

        self.prepare_training()

    def set_scheduler(self):
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=self.reduce_patience, verbose=True)

    def resume(self):
        args = self.args
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_val_acc = checkpoint['best_val_acc']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, self.start_epoch))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

    def prepare_training(self):
        self.set_scheduler()
        self.resume()

    def initialize_epoch(self):
        self.progress = tqdm(self.current_data_loader)

    def initialize_train_epoch(self):
        self.current_data_loader = self.train_loader
        self.train_epoch_acc = 0.0
        self.running_loss = 0.0
        self.train_epoch_loss = 0.0
        self.initialize_epoch()
        self.model.train()  # switch to train mode

        print('-' * 80, '\n', '-' * 80, '\n', '-' * 80)
        print('Training stage, epoch:', self.epoch)
        print('-' * 80, '\n', '-' * 80, '\n', '-' * 80)

    def initialize_val_epoch(self):
        self.current_data_loader = self.test_loader
        self.initialize_epoch()
        self.video_scores = {}  # container for video level accuracy computation
        self.model.eval()  # switch to evaluate mode

        print('-' * 80, '\n', '-' * 80, '\n', '-' * 80)
        print('Validation stage, epoch:', self.epoch)
        print('-' * 80, '\n', '-' * 80, '\n', '-' * 80)

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

    def print_train_batch_statistics(self):
        self.running_loss += self.loss.data[0]
        if self.iteration_count % self.print_steps == self.print_steps - 1:  # print every print_steps mini-batches
            print('[%d, %5d] loss: %.3f' % (self.epoch, self.iteration_count + 1, self.running_loss / self.print_steps))
            self.running_loss = 0.0

    def print_train_epoch_statistics(self):
        print('*' * 60, '\n', '*' * 60)
        print('Training accuracy of this epoch: %.1f %%' % self.train_epoch_acc)
        print('Training loss of this epoch: %.3f' % self.train_epoch_loss)
        print('*' * 60, '\n', '*' * 60, '\n')

    def print_val_statistics(self):
        print('*' * 60, '\n', '*' * 60)
        print('Validation accuracy of this epoch: %.1f %%' % self.val_acc)
        print('*' * 60, '\n', '*' * 60, '\n')

    def train_one_epoch(self):
        self.initialize_train_epoch()
        self.print_steps = len(self.train_loader) / 10
        for self.iteration_count, data in enumerate(self.progress, 0):
            inputs, labels = data  # get the inputs
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()  # wrap them in Variable and move to GPU

            self.optimizer.zero_grad()  # zero the parameter gradients

            # forward + backward + optimize
            outputs = self.model(inputs)
            self.loss = self.criterion(outputs, labels)
            self.loss.backward()
            self.optimizer.step()

            # statistics
            _, train_batch_acc = self.compute_batch_accuracy(outputs, labels)
            self.train_epoch_acc += train_batch_acc
            self.train_epoch_loss += self.loss.data[0]
            self.print_train_batch_statistics()

        iterations = self.iteration_count + 1
        self.train_epoch_acc = 100 * self.train_epoch_acc / iterations
        self.train_epoch_loss = self.train_epoch_loss / iterations
        self.print_train_epoch_statistics()

    def validate_one_epoch(self):
        self.initialize_val_epoch()

        correct = 0.
        total = 0.
        for self.iteration_count, data in enumerate(self.progress, 0):
            video_names, images, labels = data
            images, labels = Variable(images).cuda(), Variable(labels).cuda()  # wrap them in Variable and move to GPU
            outputs = self.model(images)

            batch_correct, _ = self.compute_batch_accuracy(outputs, labels)
            correct += batch_correct
            total += labels.size(0)

            self.update_video_predictions(video_names, outputs, labels)  # update video level predictions

        self.val_acc = 100 * correct / total
        self.print_val_statistics()

    def compute_video_accuracy(self):
        correct_count_top1 = 0.
        correct_count_top5 = 0.
        for video_name in self.video_scores.keys():
            video_label = self.video_scores[video_name]['label']
            video_label_numpy = video_label.data[0]
            
            video_score = self.video_scores[video_name]['score']
            video_score_numpy = video_score.cpu().numpy()

            prediction_top1 = list(np.argsort(video_score_numpy)[-1:])
            if video_label_numpy in prediction_top1:
                correct_count_top1 += 1

            prediction_top5 = list(np.argsort(video_score_numpy)[-5:])
            if video_label_numpy in prediction_top5:
                correct_count_top5 += 1

        self.video_top1 = correct_count_top1 / len(self.video_scores) * 100
        self.video_top5 = correct_count_top5 / len(self.video_scores) * 100

        print('-' * 60)
        print(' * Video-prec@1: {top1:.3f} Video-prec@5: {top5:.3f} '.format(top1=self.video_top1,
                                                                             top5=self.video_top5))
        print('-' * 60)

    def write_csv_logs(self):
        column_names = ['Data', 'Epoch', 'Arch', 'Freeze-pose-feature-net', 'Optimizer-type', 'Learning-rate', 'Batch-size', 'Saturate-patience', 'Train-Loss', 'Train-Acc', 'Val-Acc', 'Val-Video-Acc-Top1', 'Val-Video-Acc-Top5']
        info_dict = {column_names[0]: [str(type(self.train_loader.dataset))],
                     column_names[1]: [self.epoch],
                     column_names[2]: [self.arch],
                     column_names[3]: [str(self.freeze_pose_feature_net)],
                     column_names[4]: [str(type(self.optimizer))],
                     column_names[5]: [self.optimizer.param_groups[0]['lr']],
                     column_names[6]: [self.train_loader.batch_size],
                     column_names[7]: [self.saturate_patience],
                     column_names[8]: [round(self.train_epoch_loss, 3)],
                     column_names[9]: [round(self.train_epoch_acc, 3)],
                     column_names[10]: [round(self.val_acc, 3)],
                     column_names[11]: [round(self.video_top1, 3)],
                     column_names[12]: [round(self.video_top5, 3)]}

        csv_log_name = self.csv_log_name
        data_frame = pd.DataFrame.from_dict(info_dict)
        if not os.path.isfile(csv_log_name):
            data_frame.to_csv(csv_log_name, index=False, columns=column_names)
        else: # else it exists so append without writing the header
            data_frame.to_csv(csv_log_name, mode='a', header=False, index=False, columns=column_names)

    def save_checkpoints(self):
        checkpoint_name = self.checkpoint_name
        state = {'epoch': self.epoch,
                 'arch': self.arch,
                 'freeze_pose_feature_net': self.freeze_pose_feature_net,
                 'dataset': str(type(self.train_loader.dataset)),
                 'state_dict': self.model.state_dict(),
                 'val_acc': self.val_acc,
                 'best_val_acc': self.best_val_acc,
                 'val_video_acc_top1': self.video_top1,
                 'val_video_acc_top5': self.video_top5,
                 'best_video_val_acc': self.best_video_val_acc,
                 'optimizer' : self.optimizer.state_dict()}
        torch.save(state, checkpoint_name)

        if self.is_best:
            shutil.copyfile(checkpoint_name, self.best_model_name)

    def check_saturate(self):
        is_saturate = False
        if self.is_best:
            self.best_val_acc = self.val_acc
            self.best_video_val_acc = self.video_top1
            self.saturate_count = 0
        else:
            self.saturate_count += 1
            if self.saturate_count >= self.saturate_patience:
                is_saturate = True
        self.is_saturate = is_saturate

    def train(self):
        for self.epoch in range(self.start_epoch, self.epochs + 1):
            self.train_one_epoch()
            self.validate_one_epoch()
            self.scheduler.step(self.val_acc)  # call lr_scheduler

            self.compute_video_accuracy()  # compute video level accuracy
            self.write_csv_logs()

            #self.is_best = self.val_acc > self.best_val_acc
            self.is_best = self.video_top1 > self.best_video_val_acc
            self.check_saturate()
            self.save_checkpoints()
            if self.is_saturate:
                print('Validation accuracy is saturate!')
                break

        print('Finished Training')
