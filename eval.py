import argparse
import os
import errno

import numpy as np
from sklearn.metrics import roc_curve, auc
import torch
import torch.utils.data
import torchvision.transforms as transforms

import models
import ae_grad_reg
import datasets


parser = argparse.ArgumentParser(description='Evaluation of GradCon')
parser.add_argument('--print-freq', '-pf', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--dataset', default='', type=str, help='Dataset to be used for training '
                                                            '(e.g. cifar-10, mnist, fmnist)')
parser.add_argument('--dataset_dir', default='./datasets', type=str, help='Path for the dataset')
parser.add_argument('--ckpt_dir', default='./save', type=str, help='Path to the folder that contains saved models')
parser.add_argument('--ckpt_name', default='GradConCAE', type=str, help='Checkpoint name')
parser.add_argument('--output_dir', default='./results', type=str, help='Path to save the result file')
parser.add_argument('--grad-loss-weight', '-gw', default=0.12, type=float,
                    metavar='N', help='gradient loss weight for the anomaly score')


def main():

    args = parser.parse_args()

    if args.dataset not in ['cifar-10', 'mnist', 'fmnist']:
        raise ValueError('Dataset should be one of the followings: cifar-10, mnist, fmnist')

    dataset = args.dataset
    grad_loss_weight = args.grad_loss_weight

    dataset_dir = os.path.join(args.dataset_dir, dataset, 'splits')
    in_channel = 3 if dataset == 'cifar-10' else 1  # cifar-10: RGB, mnist, fminst: Graysacle
    batch_size = 1
    num_decoder_layers = 4

    auroc_results = np.zeros([1, 11])  # 10 inlier classes + average
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for in_cls in range(10):

        ae_ckpt = os.path.join(args.ckpt_dir, dataset, args.ckpt_name + '_inlier-%d/model_best.pth.tar' % in_cls)

        # Define a model
        ae = models.GradConCAE(in_channel=in_channel)
        ae = torch.nn.DataParallel(ae).to(device)
        ae.eval()

        if os.path.isfile(ae_ckpt):
            print("=> loading checkpoint '{}'".format(ae_ckpt))
            checkpoint_ae = torch.load(ae_ckpt)
            best_loss = checkpoint_ae['best_loss']
            ae.load_state_dict(checkpoint_ae['state_dict'])
            ref_grad = checkpoint_ae['ref_grad']
            print("=> loaded checkpoint '{}' (epoch {}, best_loss {})"
                  .format(ae_ckpt, checkpoint_ae['epoch'], best_loss))
        else:
            print("=> no checkpoint found at '{}'".format(ae_ckpt))
            return

        test_loader = torch.utils.data.DataLoader(
            datasets.AnomalyDataset(dataset_dir, split='test_%d' % in_cls, in_channel=in_channel,
                                    transform=transforms.ToTensor(),
                                    target_transform=transforms.ToTensor(),
                                    cls=in_cls),
            batch_size=batch_size, shuffle=False)

        result = ae_grad_reg.gradcon_score(ae, in_cls, grad_loss_weight, ref_grad,
                                           num_decoder_layers, device, test_loader)

        in_pred = result[np.where(result[:, 0] == 1)]
        out_pred = result[np.where(result[:, 0] == 0)]

        label = np.concatenate((np.ones([in_pred.shape[0], ]), np.zeros([out_pred.shape[0], ])), axis=0)
        score = np.concatenate((in_pred[:, 1], out_pred[:, 1]), axis=0)

        fpr_auc, tpr_auc, _ = roc_curve(label, score, pos_label=1)
        auroc_results[0, in_cls] = auc(fpr_auc, tpr_auc)

    auroc_results[:, -1] = np.mean(auroc_results[:, :-1], axis=1)

    try:
        os.makedirs(args.output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    save_path = os.path.join(args.output_dir, args.dataset + '_' + args.ckpt_name + '_result.txt')
    np.savetxt(save_path, auroc_results, fmt='%.4f')


if __name__ == '__main__':
    main()
