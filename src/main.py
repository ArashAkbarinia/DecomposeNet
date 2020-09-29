import os
import sys
import time
import logging

from torch import optim
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import util as vae_util
import model as vae_model
import data_loaders
from arguments import parse_arguments
from transformations import cv2_preprocessing
from transformations import cv2_transforms

models = {
    'imagenet': {'vqvae': vae_model.VQ_CVAE}
}
datasets_classes = {
    'imagenet': data_loaders.ImageFolder
}
dataset_train_args = {
    'imagenet': {},
}
dataset_test_args = {
    'imagenet': {}
}
dataset_n_channels = {
    'imagenet': 3
}
dataset_target_size = {
    'imagenet': 224,
}
default_hyperparams = {
    'imagenet': {'lr': 2e-4, 'k': 512, 'hidden': 128}
}


def main(args):
    args = parse_arguments(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.mean = (0.5, 0.5, 0.5)
    args.std = (0.5, 0.5, 0.5)

    lr = args.lr or default_hyperparams[args.dataset]['lr']
    k = args.k or default_hyperparams[args.dataset]['k']
    hidden = args.hidden or default_hyperparams[args.dataset]['hidden']
    num_channels = args.num_channels or dataset_n_channels[args.dataset]
    target_size = args.target_size or dataset_target_size[args.dataset]

    dataset_transforms = {
        'imagenet': transforms.Compose(
            [cv2_transforms.Resize(target_size + 32),
             cv2_transforms.CenterCrop(target_size),
             cv2_transforms.ToTensor(),
             cv2_transforms.Normalize(args.mean, args.std)])
    }

    save_path = vae_util.setup_logging_from_args(args)
    writer = SummaryWriter(save_path)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    in_colour_space = args.colour_space[:3]
    out_colour_space = args.colour_space[4:]
    args.colour_space = out_colour_space

    model = models[args.dataset][args.model](
        hidden, k=k, kl=args.kl, colour_space=args.colour_space
    )
    if args.cuda:
        model.cuda()

    if args.load_encoder is not None:
        params_to_optimize = [
            {'params': [p for p in model.decoder.parameters() if
                        p.requires_grad]},
            {'params': [p for p in model.fc.parameters() if
                        p.requires_grad]},
        ]
        optimizer = optim.Adam(params_to_optimize, lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, int(args.epochs / 3), 0.5
    )

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        args.start_epoch = checkpoint['epoch'] + 1
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    elif args.fine_tune is not None:
        weights = torch.load(args.fine_tune, map_location='cpu')
        model.load_state_dict(weights, strict=False)
        model.cuda()

    intransform_funs = []
    if in_colour_space != ' rgb':
        intransform_funs.append(
            cv2_preprocessing.ColourTransformation(in_colour_space)
        )
    intransform = transforms.Compose(intransform_funs)
    outtransform_funs = []
    args.inv_func = None
    if args.colour_space is not None:
        outtransform_funs.append(
            cv2_preprocessing.ColourTransformation(args.colour_space)
        )
    outtransform = transforms.Compose(outtransform_funs)

    if args.data_dir is not None:
        args.train_dir = os.path.join(args.data_dir, 'train')
        args.validation_dir = os.path.join(args.data_dir, 'validation')
    else:
        args.train_dir = args.train_dir
        args.validation_dir = args.validation_dir
    kwargs = {'num_workers': args.workers,
              'pin_memory': True} if args.cuda else {}
    args.vis_func = vae_util.grid_save_reconstructed_images

    train_loader = torch.utils.data.DataLoader(
        datasets_classes[args.dataset](
            root=args.train_dir,
            intransform=intransform,
            outtransform=outtransform,
            transform=dataset_transforms[args.dataset],
            **dataset_train_args[args.dataset]
        ),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets_classes[args.dataset](
            root=args.validation_dir,
            intransform=intransform,
            outtransform=outtransform,
            transform=dataset_transforms[args.dataset],
            **dataset_test_args[args.dataset]
        ),
        batch_size=args.batch_size, shuffle=False, **kwargs
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    for epoch in range(args.start_epoch, args.epochs):
        train_losses = train(
            epoch, model, train_loader, optimizer, args.cuda, args.log_interval,
            save_path, args
        )
        test_losses = test_net(epoch, model, test_loader, args.cuda, save_path,
                               args)
        for k in train_losses.keys():
            name = k.replace('_train', '')
            train_name = k
            test_name = k.replace('train', 'test')
            writer.add_scalars(
                name, {'train': train_losses[train_name],
                       'test': test_losses[test_name]}, epoch
            )
        scheduler.step()
        vae_util.save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'arch': {'k': args.k, 'hidden': args.hidden}
            },
            save_path
        )


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path,
          args):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    for batch_idx, loader_data in enumerate(train_loader):
        data = loader_data[0]
        target = loader_data[1]
        max_len = len(train_loader)
        target = target.cuda()

        data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data)

        loss = model.loss_function(target, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])

        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(
                ['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            logging.info(
                'Train Epoch: {epoch} [{batch:5d}/{total_batch} '
                '({percent:2d}%)]   time: {time:3.2f}   {loss}'
                    .format(epoch=epoch, batch=batch_idx * len(data),
                            total_batch=max_len * len(data),
                            percent=int(100. * batch_idx / max_len),
                            time=time.time() - start_time, loss=loss_string))
            start_time = time.time()
            for key in latest_losses:
                losses[key + '_train'] = 0
        if batch_idx in [18, 180, 1650, max_len - 1]:
            args.vis_func(
                target, outputs, args.mean, args.std, epoch, save_path,
                'reconstruction_train%.5d' % batch_idx, args.inv_func
            )

        if batch_idx * len(data) > args.max_epoch_samples:
            break

    for key in epoch_losses:
        epoch_losses[key] /= (
                len(train_loader.dataset) / train_loader.batch_size)
    loss_string = '\t'.join(
        ['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    return epoch_losses


def test_net(epoch, model, test_loader, cuda, save_path, args):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    i, data = None, None
    with torch.no_grad():
        for i, loader_data in enumerate(test_loader):
            data = loader_data[0]
            target = loader_data[1]
            target = target.cuda()
            data = data.cuda()
            outputs = model(data)
            model.loss_function(target, *outputs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
            if i in [0, 100, 200, 300, 400]:
                args.vis_func(
                    target, outputs, args.mean, args.std, epoch, save_path,
                    'reconstruction_test%.5d' % i, args.inv_func
                )

    for key in losses:
        losses[key] /= (i * len(data))
    loss_string = ' '.join(
        ['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
    logging.info('====> Test set losses: {}'.format(loss_string))
    return losses


if __name__ == "__main__":
    main(sys.argv[1:])
