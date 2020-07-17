import torch
import torchvision.utils as vutils
import torch.utils.data
import torchvision.transforms as transforms
from torch import optim

import argparse
import os
import time
import errno
from tensorboardX import SummaryWriter

import models
import utils
import datasets

import ae_grad_reg


parser = argparse.ArgumentParser(description='Training GradCon')
parser.add_argument('-e', '--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-pf', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--write-freq', '-wf', default=5, type=int,
                    metavar='N', help='write frequency (default: 5)')
parser.add_argument('-r', '--resume', default='', type=str, help='Resume training from a checkpoint')
parser.add_argument('--dataset', default='', type=str, help='Dataset to be used for training (e.g. cifar-10, mnist)')
parser.add_argument('--dataset_dir', default='./datasets', type=str, help='Path for the dataset')
parser.add_argument('--save_dir', default='./save', type=str, help='Path to save the data')
parser.add_argument('--save_name', default='GradConCAE', type=str, help='Save name')
parser.add_argument('--grad-loss-weight', '-gw', default=3e-2, type=float, help='gradient loss weight')


def main():

    args = parser.parse_args()

    if args.dataset not in ['cifar-10', 'mnist']:
        raise ValueError('Dataset should be one of the followings: cifar-10, mnist')

    dataset = args.dataset
    grad_loss_weight = args.grad_loss_weight

    dataset_dir = os.path.join(args.dataset_dir, dataset, 'splits')
    in_channel = 3 if dataset == 'cifar-10' else 1  # cifar-10: RGB, mnist, fminst: Graysacle
    batch_size = 64
    num_decoder_layers = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for in_cls in range(0, 10):
        print('Training with inlier class: %d' % in_cls)

        save_dir = os.path.join(args.save_dir, dataset, args.save_name + '_inlier-%d' % in_cls)
        log_dir = os.path.join(save_dir, 'logs')
        try:
            os.makedirs(save_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        writer = SummaryWriter(log_dir=log_dir)
        print('Save directory: %s' % save_dir)

        # Define an autoencoder model
        ae = models.GradConCAE(in_channel=in_channel)
        ae = torch.nn.DataParallel(ae).to(device)
        best_loss = 1e20
        optimizer = optim.Adam(ae.parameters(), lr=1e-3)

        # Keep track of traning gradients to calculate the gradient loss
        ref_grad = []
        for i in range(num_decoder_layers):
            layer_grad = utils.AverageMeter()
            layer_grad.avg = torch.zeros(ae.module.up[2 * i].weight.shape).to(device)
            ref_grad.append(layer_grad)

        # Resume training from a checkpoint
        if args.resume:
            ae_resume_ckpt = os.path.join(args.resume, 'model_best.pth.tar')
            if os.path.isfile(ae_resume_ckpt):
                print("=> loading checkpoint '{}'".format(ae_resume_ckpt))
                checkpoint = torch.load(ae_resume_ckpt)
                ae.load_state_dict(checkpoint['state_dict'])
                ref_grad = checkpoint['ref_grad']
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {}, best_loss {})"
                      .format(ae_resume_ckpt, checkpoint['epoch'], checkpoint['best_loss']))
            else:
                print("=> no checkpoint found at '{}'".format(ae_resume_ckpt))
                return

        # Dataloader for training and validation
        in_train_loader = torch.utils.data.DataLoader(
            datasets.AnomalyDataset(dataset_dir, split='train', in_channel=in_channel,
                                 transform=transforms.ToTensor(),
                                 target_transform=transforms.ToTensor(),
                                 cls=in_cls),
            batch_size=batch_size, shuffle=True)

        in_val_loader = torch.utils.data.DataLoader(
            datasets.AnomalyDataset(dataset_dir, split='val', in_channel=in_channel,
                                 transform=transforms.ToTensor(),
                                 target_transform=transforms.ToTensor(),
                                 cls=in_cls),
            batch_size=batch_size, shuffle=True)

        # Start training
        timestart = time.time()
        for epoch in range(args.start_epoch, args.epochs):

            print('\n*** Start Training *** Epoch: [%d/%d]\n' % (epoch + 1, args.epochs))
            ae_grad_reg.train(ae, device, in_train_loader, optimizer, epoch + 1, args.print_freq,
                              grad_loss_weight, ref_grad, num_decoder_layers)

            print('\n*** Start Testing *** Epoch: [%d/%d]\n' % (epoch + 1, args.epochs))
            loss, recon_loss, grad_loss, input_img, recon_img, target_img = ae_grad_reg.test(
                ae, device, in_val_loader, epoch + 1, args.print_freq, grad_loss_weight,
                ref_grad, num_decoder_layers)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if is_best:
                best_epoch = epoch + 1

            if (epoch % args.write_freq == 0) or (epoch == args.epochs - 1):
                writer.add_scalar('loss', loss, epoch + 1)
                writer.add_scalar('recon_loss', recon_loss, epoch + 1)
                writer.add_scalar('grad_loss', grad_loss, epoch + 1)

                writer.add_image('input_img', vutils.make_grid(input_img, nrow=3), epoch + 1)
                writer.add_image('recon_img', vutils.make_grid(recon_img, nrow=3), epoch + 1)
                writer.add_image('target_img', vutils.make_grid(target_img, nrow=3), epoch + 1)

                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': ae.state_dict(),
                    'best_loss': best_loss,
                    'last_loss': loss,
                    'optimizer': optimizer.state_dict(),
                    'ref_grad': ref_grad,
                }, is_best, save_dir)

        writer.close()

        print('Best test loss: %.3f at epoch %d' % (best_loss, best_epoch))
        print('Best epoch: ', best_epoch)
        print('Total processing time: %.4f' % (time.time() - timestart))


if __name__ == '__main__':
    main()
