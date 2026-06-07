import numpy as np
import os
import sys
import random

import torch
from torchvision import transforms

from skimage import color
from skimage import metrics
import cv2
import argparse

import util as vae_util
import model as vae_model
import data_loaders
from transformations import cv2_preprocessing
from transformations import cv2_transforms
from transformations import colour_spaces, normalisations


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')
    parser.add_argument('--model_dir', required=True)
    parser.add_argument(
        '--model', default='vqvae',
        help='autoencoder variant to use: vae | vqvae'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128, metavar='N',
        help='input batch size for training (default: 128)'
    )
    parser.add_argument('--exclude', type=int, default=0, metavar='K',
                        help='number of atoms in dictionary')
    parser.add_argument('--colour_space', type=str, default=None,
                        help='The type of output colour space.')
    parser.add_argument('--target_size', type=int, default=224,
                        dest='target_size', help='target_size of image')

    parser.add_argument(
        '--dataset', default=None,
        help='dataset to use: mnist | cifar10 | imagenet | coco | custom'
    )
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='The specific category (default: None)'
    )
    parser.add_argument(
        '--validation_dir',
        type=str,
        default=None,
        help='The path to the validation directory (default: None)'
    )
    parser.add_argument('--noise', type=str, default=None)

    return parser.parse_args(args)


def main(args):
    args = parse_arguments(args)
    args.out_dir = f"{args.model_dir}/evaluation/"
    args.model_path = f"{args.model_dir}/checkpoints/last_epoch.pth"

    # making the model
    weights_net = torch.load(args.model_path, map_location='cpu', weights_only=False)
    k = weights_net['arch']['k']
    d = weights_net['arch']['hidden']
    stride = weights_net['arch']['stride'] if 'stride' in weights_net['arch'] else 2
    network = vae_model.VQ_CVAE(d, k=k, kl=d, stride=stride, in_chns=3)
    network.load_state_dict(weights_net['state_dict'])

    if args.exclude > 0:
        which_vec = [args.exclude - 1]
        print(which_vec)
        network.state_dict()['emb.weight'][:, which_vec] = 0
    elif args.exclude < 0:
        which_vec = [*range(8)]
        which_vec.remove(abs(args.exclude) - 1)
        print(which_vec)
        network.state_dict()['emb.weight'][:, which_vec] = 0
    network.cuda()
    network.eval()

    # assuming a particular folder structure
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    args.in_colour_space = args.colour_space[:3]
    args.out_colour_space = args.colour_space[4:]

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform_funcs = transforms.Compose([
        cv2_transforms.Resize(args.target_size + 32),
        cv2_transforms.CenterCrop(args.target_size),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std)
    ])

    intransform_funs = []
    if args.in_colour_space != ' rgb':
        intransform_funs.append(cv2_preprocessing.ColourTransformation(args.in_colour_space))
    if args.noise is not None:
        if args.noise == 'sp':
            noise_fun = imutils.s_p_noise
            kwargs = {'amount': 0.01, 'seed': args.random_seed}
        elif args.noise == 'gaussian':
            noise_fun = imutils.gaussian_noise
            kwargs = {'amount': 0.01, 'seed': args.random_seed}
        elif args.noise == 'speckle':
            noise_fun = imutils.speckle_noise
            kwargs = {'amount': 0.01, 'seed': args.random_seed}
        kwargs['eq_chns'] = True
        intransform_funs.append(cv2_preprocessing.UniqueTransformation(noise_fun, **kwargs))
    intransform = transforms.Compose(intransform_funs)

    if args.dataset == 'imagenet':
        test_loader = torch.utils.data.DataLoader(
            data_loaders.ImageFolder(
                root=args.validation_dir,
                intransform=intransform,
                outtransform=None,
                transform=transform_funcs
            ),
            batch_size=args.batch_size, shuffle=False
        )
    elif args.dataset == 'celeba':
        test_loader = torch.utils.data.DataLoader(
            data_loaders.CelebA(
                root=args.validation_dir,
                intransform=intransform,
                outtransform=None,
                transform=transform_funcs,
                split='test'
            ),
            batch_size=args.batch_size, shuffle=False
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            data_loaders.CategoryImages(
                root=args.validation_dir,
                # FIXME
                category=args.category,
                intransform=intransform,
                outtransform=None,
                transform=transform_funcs
            ),
            batch_size=args.batch_size, shuffle=False
        )
    export(test_loader, network, mean, std, args)


def export(data_loader, model, mean, std, args):
    all_des = []
    all_ssim = []
    all_psnr = []
    with torch.no_grad():
        for i, (img_readies, img_target, img_paths) in enumerate(data_loader):
            img_readies = img_readies.cuda()
            out_img = model(img_readies)
            out_img = out_img[0].detach().cpu()

            for img_ind in range(out_img.shape[0]):
                img_path = img_paths[img_ind]
                if np.mod(i, 1000) == 0:
                    print(i, img_path)
                ref_img = img_target[img_ind].unsqueeze(0)

                ref_img_tmp = vae_util.inv_normalise_tensor(ref_img, mean, std)
                ref_img_tmp = ref_img_tmp.numpy().squeeze().transpose(1, 2, 0)

                # target transforms are None so always are in RGB
                ref_img_tmp = normalisations.uint8im(ref_img_tmp)

                rec_img_tmp = vae_util.inv_normalise_tensor(out_img[img_ind].unsqueeze(0), mean, std)
                rec_img_tmp = rec_img_tmp.numpy().squeeze().transpose(1, 2, 0)
                rec_img_tmp = cv2.resize(rec_img_tmp, (ref_img_tmp.shape[1], ref_img_tmp.shape[0]))
                if args.out_colour_space == 'lab':
                    rec_img_tmp = np.uint8(rec_img_tmp * 255)
                    rec_img_tmp = cv2.cvtColor(rec_img_tmp, cv2.COLOR_LAB2RGB)
                elif args.out_colour_space == 'hsv':
                    rec_img_tmp = colour_spaces.hsv012rgb(rec_img_tmp)
                elif args.out_colour_space == 'lms':
                    rec_img_tmp = colour_spaces.lms012rgb(rec_img_tmp)
                elif args.out_colour_space == 'yog':
                    rec_img_tmp = colour_spaces.yog012rgb(rec_img_tmp)
                elif args.out_colour_space == 'dkl':
                    rec_img_tmp = colour_spaces.dkl012rgb(rec_img_tmp)
                else:
                    rec_img_tmp = normalisations.uint8im(rec_img_tmp)

                # SSIM
                ssim = metrics.structural_similarity(ref_img_tmp, rec_img_tmp, channel_axis=2)
                all_ssim.append(ssim)
                # PSNR
                psnr = metrics.peak_signal_noise_ratio(ref_img_tmp, rec_img_tmp)
                all_psnr.append(psnr)
                # Delta E
                img_org = color.rgb2lab(ref_img_tmp)
                img_res = color.rgb2lab(rec_img_tmp)
                de = color.deltaE_ciede2000(img_org, img_res)
                all_des.append([np.mean(de), np.median(de), np.max(de)])

            if np.mod(i, 10000) == 0:
                np.savetxt(args.out_dir + '/ssim.txt', np.array(all_ssim))
                np.savetxt(args.out_dir + '/psnr.txt', np.array(all_psnr))
                np.savetxt(args.out_dir + '/de.txt', np.array(all_des))

    np.savetxt(args.out_dir + '/ssim.txt', np.array(all_ssim))
    np.savetxt(args.out_dir + '/psnr.txt', np.array(all_psnr))
    np.savetxt(args.out_dir + '/de.txt', np.array(all_des))


if __name__ == "__main__":
    main(sys.argv[1:])
