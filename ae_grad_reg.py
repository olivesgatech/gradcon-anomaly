import torch
import torch.nn.functional as func
import numpy as np

import utils


def train(model, device, train_loader, optimizer, epoch, print_freq, grad_loss_weight, ref_grad, nlayer):
    """
    Args:
         model (DataParallel(module)): VAE
         device (Device): CPU or GPU
         train_loader (DataLoader): data loader for training data
         optimizer
         epoch (int)
         print_freq(int): Print frequency
         grad_loss_weight (float): Weight for the gradient loss
         ref_grad (tensor): Average of gradients generated while training
         nlayer (int): Number of decoder layers
    """
    model.train()
    losses = utils.AverageMeter()
    recon_losses = utils.AverageMeter()
    grad_losses = utils.AverageMeter()

    for batch_idx, (data, target_data, label) in enumerate(train_loader):
        data = data.to(device)
        target_data = target_data.to(device)
        optimizer.zero_grad()
        model.zero_grad()

        recon_batch = model(data)
        recon_loss = func.mse_loss(recon_batch, target_data)

        # Calculate the gradient loss for each layer
        grad_loss = 0
        for i in range(nlayer):
            wrt = model.module.up[int(2*i)].weight
            target_grad = torch.autograd.grad(recon_loss, wrt, create_graph=True, retain_graph=True)[0]

            grad_loss += -1 * func.cosine_similarity(target_grad.view(-1, 1),
                                                     ref_grad[i].avg.view(-1, 1), dim=0)

        # In the first iteration, since there is no history of training gradients, gradient loss is not utilized
        if ref_grad[0].count == 0:
            grad_loss = torch.FloatTensor([0.0]).to(device)
        else:
            grad_loss = grad_loss / nlayer
        loss = recon_loss + grad_loss_weight * grad_loss

        losses.update(loss.item(), data.size(0))  # data.size(0): Batch size
        recon_losses.update(recon_loss.item(), data.size(0))
        grad_losses.update(grad_loss.item(), data.size(0))
        loss.backward()

        # Update the reference gradient
        for i in range(nlayer):
            ref_grad[i].update(model.module.up[2*i].weight.grad, 1)

        optimizer.step()

        if batch_idx % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f}) '
                  'Grad Loss {grad_loss.val:.4f} ({grad_loss.avg:.4f})'
                  .format(epoch, batch_idx, len(train_loader), loss=losses, recon_loss=recon_losses,
                          grad_loss=grad_losses))


def test(model, device, test_loader, epoch, print_freq, grad_loss_weight, ref_grad, nlayer):
    """
    Args:
         model (DataParallel(module)): VAE
         device (Device): CPU or GPU
         test_loader (DataLoader): data loader for test data
         epoch (int)
         print_freq(int): Print frequency
         grad_loss_weight (float): Weight for the gradient loss
         ref_grad (tensor): Average of gradients generated while training
         nlayer (int): Number of decoder layers
    """
    model.eval()
    losses = utils.AverageMeter()
    recon_losses = utils.AverageMeter()
    grad_losses = utils.AverageMeter()

    for batch_idx, (data, target_data, label) in enumerate(test_loader):
        data = data.to(device)
        target_data = target_data.to(device)
        model.zero_grad()

        recon_batch = model(data)
        recon_loss = func.mse_loss(recon_batch, target_data)

        # Calculate the gradient loss for each layer
        grad_loss = 0
        for i in range(nlayer):
            wrt = model.module.up[int(2*i)].weight
            target_grad = torch.autograd.grad(recon_loss, wrt, create_graph=True, retain_graph=True)[0]
            grad_loss += -1 * func.cosine_similarity(target_grad.view(-1, 1),
                                                     ref_grad[i].avg.view(-1, 1), dim=0)

        grad_loss = grad_loss / nlayer
        loss = recon_loss + grad_loss_weight * grad_loss

        losses.update(loss.item(), data.size(0))
        recon_losses.update(recon_loss.item(), data.size(0))
        grad_losses.update(grad_loss.item(), data.size(0))

        if batch_idx == 0:
            nimg = 3  # Visualize three sample images on Tensorboard
            input_img = data[:nimg]
            recon_img = recon_batch[:nimg, :].view(nimg, data.shape[1], data.shape[2], data.shape[3])
            target_img = target_data[:nimg]

        if batch_idx % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f}) '
                  'Grad Loss {grad_loss.val:.4f} ({grad_loss.avg:.4f})'
                  .format(epoch, batch_idx, len(test_loader), loss=losses, recon_loss=recon_losses,
                          grad_loss=grad_losses))

    print(' * Loss {loss.avg:.3f}'.format(loss=losses))
    return losses.avg, recon_losses.avg, grad_losses.avg, input_img, recon_img, target_img


def gradcon_score(model, in_cls, grad_loss_weight, ref_grad, nlayer, device, test_loader):
    """
    Args:
        model (DataParallel(module)): AE
        in_cls (int): Inlier class
        grad_loss_weight (float): Weight for the gradient loss
        ref_grad (list): Extracted gradients from training data
        nlayer (int): Number of decoder layers
        device (Device): CPU or GPU
        test_loader (DataLoader): Data loader for test data
    Return:
        result (ndarray): (number of samples) x 2 (label, estimated score)
    """
    model.eval()

    results = np.zeros([len(test_loader.dataset), 2])
    for batch_idx, (data, target_data, class_label) in enumerate(test_loader):

        if batch_idx % 10 == 0:
            print('Evaluation inlier {0}: [{1} / {2}]...'.format(in_cls, batch_idx, len(test_loader)))

        data = data.to(device)
        target_data = target_data.to(device)

        model.zero_grad()

        recon_batch = model(data)
        recon_loss = func.mse_loss(recon_batch, target_data)

        recon_loss.backward()

        grad_loss = 0
        for i in range(nlayer):
            target_grad = model.module.up[int(2*i)].weight.grad
            grad_loss += 1 * func.cosine_similarity(target_grad.view(-1, 1), ref_grad[i].avg.view(-1, 1), dim=0)

        grad_loss = grad_loss / nlayer

        score = -1 * recon_loss + grad_loss_weight * grad_loss
        inout_label = 1 if class_label == in_cls else 0

        results[batch_idx, 0] = inout_label
        results[batch_idx, 1] = score.cpu().detach().numpy()

    return results
