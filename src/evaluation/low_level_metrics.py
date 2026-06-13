import argparse
import os
import sys

import cv2
import numpy as np
import torch
from skimage import color, metrics
from torchvision import transforms

import data_loaders
import model as vae_model
import util as vae_util
from transformations import colour_spaces, cv2_preprocessing, cv2_transforms, normalisations


def parse_arguments(args):
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Report low-level image quality metrics (SSIM, PSNR, DeltaE)"
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Directory containing checkpoints and model configuration."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="Batch size for evaluation (default: 128)."
    )
    parser.add_argument(
        "--exclude",
        type=int,
        default=0,
        metavar="K",
        help=(
            "Exclude embedding vectors. "
            "Positive values exclude a single vector (1-indexed). "
            "Negative values keep only one vector and exclude the rest."
        )
    )
    parser.add_argument(
        "--colour_space",
        type=str,
        required=True,
        help="Colour space conversion, e.g. rgb2lab, lab2rgb."
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=224,
        help="Target image size."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset to use: imagenet | celeba"
    )
    parser.add_argument(
        "--validation_dir",
        type=str,
        default=None,
        help="Path to validation dataset."
    )

    return parser.parse_args(args)


def parse_colour_space(colour_space):
    """
    Parse colour-space specification such as 'rgb2lab'.
    """
    try:
        in_colour_space, out_colour_space = colour_space.lower().split("2")
    except ValueError:
        raise ValueError(
            f"Invalid colour space specification '{colour_space}'. "
            f"Expected format such as 'rgb2lab' or 'lab2rgb'."
        )

    return in_colour_space, out_colour_space


def create_model(model_path, device):
    """
    Load VQ-VAE model and weights.
    """
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    architecture = checkpoint["arch"]

    num_embeddings = architecture["k"]
    hidden_dim = architecture["hidden"]
    stride = architecture.get("stride", 2)

    model = vae_model.VQ_CVAE(
        hidden_dim,
        k=num_embeddings,
        kl=hidden_dim,
        stride=stride,
        in_chns=3
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, num_embeddings


def exclude_embedding_vectors(model, num_embeddings, exclude):
    """
    Zero selected embedding vectors.

    exclude > 0:
        Exclude one vector (1-indexed).

    exclude < 0:
        Keep only one vector and exclude all others.
    """
    if exclude == 0:
        return

    with torch.no_grad():
        if exclude > 0:
            excluded_embeddings = [exclude - 1]
        else:
            keep_idx = abs(exclude) - 1
            if keep_idx >= num_embeddings:
                raise ValueError(
                    f"Requested embedding {keep_idx + 1}, "
                    f"but model only has {num_embeddings} embeddings."
                )
            excluded_embeddings = list(range(num_embeddings))
            excluded_embeddings.remove(keep_idx)

        print(f"Excluding embedding vectors: {excluded_embeddings}")

        model.emb.weight[:, excluded_embeddings] = 0


def create_dataloader(args, input_transform, image_transform):
    """
    Create evaluation dataloader.
    """
    if args.dataset == "imagenet":
        dataset = data_loaders.ImageFolder(
            root=args.validation_dir,
            intransform=input_transform,
            outtransform=None,
            transform=image_transform
        )
    elif args.dataset == "celeba":
        dataset = data_loaders.CelebA(
            root=args.validation_dir,
            intransform=input_transform,
            outtransform=None,
            transform=image_transform,
            split="test"
        )
    else:
        sys.exit(f"Unsupported dataset: {args.dataset}")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )


def convert_reconstruction_to_rgb(reconstruction, output_colour_space):
    """
    Convert reconstruction from output colour space back to RGB.
    """

    if output_colour_space == "lab":
        # Assumes OpenCV Lab encoding.
        reconstruction = np.uint8(reconstruction * 255)
        reconstruction = cv2.cvtColor(
            reconstruction,
            cv2.COLOR_LAB2RGB
        )
    elif output_colour_space == "hsv":
        reconstruction = colour_spaces.hsv012rgb(reconstruction)
    elif output_colour_space == "lms":
        reconstruction = colour_spaces.lms012rgb(reconstruction)
    elif output_colour_space == "yog":
        reconstruction = colour_spaces.yog012rgb(reconstruction)
    elif output_colour_space == "dkl":
        reconstruction = colour_spaces.dkl012rgb(reconstruction)
    else:
        reconstruction = normalisations.uint8im(reconstruction)

    return reconstruction


def save_metrics(output_dir, all_ssim, all_psnr, all_delta_e):
    """
    Save intermediate/final metric results.
    """
    np.savetxt(
        os.path.join(output_dir, "ssim.txt"),
        np.array(all_ssim)
    )

    np.savetxt(
        os.path.join(output_dir, "psnr.txt"),
        np.array(all_psnr)
    )

    np.savetxt(
        os.path.join(output_dir, "de.txt"),
        np.array(all_delta_e)
    )


def run_quality_metrics(
        data_loader,
        model,
        mean,
        std,
        args,
        device
):
    """
    Evaluate image quality metrics.
    """

    all_delta_e = []
    all_ssim = []
    all_psnr = []

    with torch.no_grad():

        for batch_idx, (input_images, target_images, category) in enumerate(data_loader):
            print('running batch:', batch_idx)
            input_images = input_images.to(device)

            reconstructions = model(input_images)[0]
            reconstructions = reconstructions.detach().cpu()

            for image_idx in range(reconstructions.shape[0]):
                reference_rgb = target_images[image_idx].unsqueeze(0)

                reference_rgb = vae_util.inv_normalise_tensor(
                    reference_rgb,
                    mean,
                    std
                )

                reference_rgb = (
                    reference_rgb.numpy()
                    .squeeze()
                    .transpose(1, 2, 0)
                )

                # Targets are always RGB because outtransform=None.
                reference_rgb = normalisations.uint8im(reference_rgb)

                reconstruction_rgb = vae_util.inv_normalise_tensor(
                    reconstructions[image_idx].unsqueeze(0),
                    mean,
                    std
                )

                reconstruction_rgb = (
                    reconstruction_rgb.numpy()
                    .squeeze()
                    .transpose(1, 2, 0)
                )

                reconstruction_rgb = cv2.resize(
                    reconstruction_rgb,
                    (
                        reference_rgb.shape[1],
                        reference_rgb.shape[0]
                    )
                )

                reconstruction_rgb = convert_reconstruction_to_rgb(
                    reconstruction_rgb,
                    args.out_colour_space
                )

                # SSIM
                ssim = metrics.structural_similarity(
                    reference_rgb,
                    reconstruction_rgb,
                    channel_axis=2,
                    data_range=255
                )

                all_ssim.append(ssim)

                # PSNR
                psnr = metrics.peak_signal_noise_ratio(
                    reference_rgb,
                    reconstruction_rgb,
                    data_range=255
                )

                all_psnr.append(psnr)

                # DeltaE2000
                reference_lab = color.rgb2lab(reference_rgb)
                reconstruction_lab = color.rgb2lab(reconstruction_rgb)

                delta_e = color.deltaE_ciede2000(
                    reference_lab,
                    reconstruction_lab
                )

                all_delta_e.append([
                    np.mean(delta_e),
                    np.median(delta_e),
                    np.max(delta_e)
                ])

            if batch_idx % 2 == 0:
                save_metrics(
                    args.out_dir,
                    all_ssim,
                    all_psnr,
                    all_delta_e
                )

    save_metrics(
        args.out_dir,
        all_ssim,
        all_psnr,
        all_delta_e
    )


def main(argv):
    args = parse_arguments(argv)

    args.out_dir = os.path.join(
        args.model_dir,
        "evaluation"
    )

    args.model_path = os.path.join(
        args.model_dir,
        "checkpoints",
        "last_epoch.pth"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    (
        args.in_colour_space,
        args.out_colour_space
    ) = parse_colour_space(args.colour_space)

    model, num_embeddings = create_model(
        args.model_path,
        device
    )

    exclude_embedding_vectors(
        model,
        num_embeddings,
        args.exclude
    )

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    image_transform = transforms.Compose([
        cv2_transforms.Resize(args.target_size + 32),
        cv2_transforms.CenterCrop(args.target_size),
        cv2_transforms.ToTensor(),
        cv2_transforms.Normalize(mean, std)
    ])

    input_transforms = []

    if args.in_colour_space != "rgb":
        input_transforms.append(
            cv2_preprocessing.ColourTransformation(
                args.in_colour_space
            )
        )

    input_transform = transforms.Compose(
        input_transforms
    )

    data_loader = create_dataloader(
        args,
        input_transform,
        image_transform
    )

    run_quality_metrics(
        data_loader,
        model,
        mean,
        std,
        args,
        device
    )


if __name__ == "__main__":
    main(sys.argv[1:])
