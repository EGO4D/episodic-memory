import sys
sys.dont_write_bytecode = True
import os
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
import Utils.opts as opts
from Utils.dataset import VideoDataSet
from Models.VSGN import VSGN
import datetime
from collections import defaultdict

torch.manual_seed(21)

def Train_VSGN(opt):
    path_appendix = '_'.join(string for string in opt['checkpoint_path'].split('_')[1:])
    writer = SummaryWriter(logdir='runs/' + path_appendix)
    model = VSGN(opt)
    device = "cuda"
    model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt["train_lr"], weight_decay=opt["weight_decay"])

    start_epoch = 0

    kwargs = {'num_workers': 16, 'pin_memory': True, 'drop_last': True}

    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               **kwargs)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=opt["batch_size"], shuffle=False,
                                              **kwargs)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])

    for epoch in range(start_epoch, opt["num_epoch"]):

        train_VSGN_epoch(train_loader, model, optimizer, epoch, writer, opt)
        epoch_loss = test_VSGN_epoch(test_loader, model, epoch, writer, opt)

        print((datetime.datetime.now()))
        state = {'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, opt["checkpoint_path"] + "/checkpoint.pth.tar")
        if epoch_loss < model.module.tem_best_loss:
            print((datetime.datetime.now()))
            print('The best model up to now is from Epoch {}'.format(epoch))
            model.module.tem_best_loss = np.mean(epoch_loss)
            torch.save(state, opt["checkpoint_path"] + "/best.pth.tar")

        scheduler.step()
    writer.close()


def train_VSGN_epoch(data_loader, model, optimizer, epoch, writer, opt, is_train=True):

    if is_train:
        model.train()
    else:
        model.eval()

    epoch_losses = defaultdict(float)
    for n_iter, (input_data, gt_action, gt_start, gt_end, gt_bbox, num_gt, num_frms) in enumerate(data_loader):
        with torch.set_grad_enabled(is_train):
            losses, pred_action, pred_start, pred_end = model(input_data, num_frms, gt_action, gt_start, gt_end,  gt_bbox, num_gt)


        # Overall loss
        loss_cls_dec = torch.mean(losses['loss_cls_dec'])
        loss_reg_dec = torch.mean(losses['loss_reg_dec'])
        loss_action = torch.mean(losses['loss_action'])
        loss_start = torch.mean(losses['loss_start'])
        loss_end = torch.mean(losses['loss_end'])
        loss_bd_adjust = torch.mean(losses['loss_bd_adjust'])

        loss = loss_cls_dec + loss_reg_dec + 0.2*loss_action + 0.2*loss_start +0.2*loss_end + loss_bd_adjust

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_losses['loss'] += loss.cpu().detach().numpy()
        epoch_losses['loss_cls_dec'] += loss_cls_dec.cpu().detach().numpy()
        epoch_losses['loss_reg_dec'] += loss_reg_dec.cpu().detach().numpy()
        epoch_losses['loss_action'] += loss_action.cpu().detach().numpy()
        epoch_losses['loss_start'] += loss_start.cpu().detach().numpy()
        epoch_losses['loss_end'] += loss_end.cpu().detach().numpy()
        epoch_losses['loss_bd_adjust'] += loss_bd_adjust.cpu().detach().numpy()

    for k, v in epoch_losses.items():
        epoch_losses[k] = v / (n_iter + 1)

    to_print = ["%s loss (epoch %d): " % ('Train' if is_train else 'Val', epoch)]
    for k, v in epoch_losses.items():
        writer.add_scalar('%s/%s' % ('train' if is_train else 'val', k), v, epoch)
        writer.flush()
        to_print.append('%s: %.04f' % (k, v))
    print(' '.join(to_print))

    return epoch_losses['loss']



def test_VSGN_epoch(data_loader, model, epoch, writer, opt):
    return train_VSGN_epoch(data_loader, model, None, epoch, writer, opt, is_train=False)




if __name__ == '__main__':

    print(datetime.datetime.now())

    opt = opts.parse_opt()
    opt = vars(opt)

    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])

    print(opt)

    print("---------------------------------------------------------------------------------------------")
    print("Training starts!")
    print("---------------------------------------------------------------------------------------------")
    Train_VSGN(opt)
    print("Training finishes!")

    print(datetime.datetime.now())

