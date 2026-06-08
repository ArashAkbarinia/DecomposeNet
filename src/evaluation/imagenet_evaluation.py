import argparse
import os
import csv
import sys

import cv2
import numpy as np
import torch
from torchvision.models import get_model_weights, get_model
from torchvision import transforms

import data_loaders
import model as vae_model
import util as vae_util
from transformations import cv2_preprocessing, cv2_transforms
from transformations import colour_spaces as cpu_colour
from transformations import colour_spaces_gpu as gpu_colour


def parse_arguments(args):
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Report high-level image classification accuracy"
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Directory containing checkpoints and model configuration."
    )
    parser.add_argument(
        "--classification_model",
        default="ResNet50",
        help="Name of the classification model."
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
        reconstruction = cpu_colour.hsv012rgb(reconstruction)
    elif output_colour_space == "lms":
        reconstruction = gpu_colour.lms012rgb(reconstruction)
    elif output_colour_space == "yog":
        reconstruction = gpu_colour.yog012rgb(reconstruction)
    elif output_colour_space == "dkl":
        reconstruction = gpu_colour.dkl012rgb(reconstruction)
    else:
        reconstruction = gpu_colour._uint8im(reconstruction)

    return reconstruction


def run_image_classification(
        data_loader,
        model,
        mean,
        std,
        classification_model,
        classification_preprocess,
        args,
        device
):
    """
    Evaluate image classification accuracy on reconstructed images.

    For each batch:
      1. Reconstruct images through the VQ-VAE.
      2. Convert the reconstruction back to RGB (uint8, [0..255]).
      3. Re-normalise for the ImageNet classifier and run inference.
      4. Record top-1 predictions, ground-truth labels, and per-sample
         correctness, then write everything to a single CSV at the end.
    """

    all_predictions = []  # top-1 predicted class index per sample
    all_gts = []  # ground-truth class index per sample
    all_correct = []  # 1 if correct, 0 otherwise, per sample

    if args.out_colour_space not in ['yog', 'dkl', 'lms', 'rgb']:
        sys.exit(
            f"Color conversion for colour space {args.out_colour_space} is not "
            f"CUDA compatible, therefore it will be too slow."
        )

    with torch.no_grad():
        for batch_idx, (input_images, target_images, class_target) in enumerate(data_loader):
            input_images = input_images.to(device)

            # --- VQ-VAE reconstruction ---
            reconstructions = model(input_images)[0]

            # Map from [-1, 1] back to [0, 1]
            reconstructions_01 = vae_util.inv_normalise_tensor(
                reconstructions,
                mean,
                std
            )

            # Convert from the model's output colour space to RGB uint8 [0, 255]
            reconstructions_rgb = convert_reconstruction_to_rgb(
                reconstructions_01,
                args.out_colour_space
            )

            # --- Prepare tensor for the ImageNet classifier ---
            # reconstructions_rgb is uint8 on GPU; bring to float [0, 1] per image,
            # then apply the classifier's own normalisation
            x = reconstructions_rgb.float().div(255.0)
            x = vae_util.normalise_tensor(
                x, classification_preprocess.mean, classification_preprocess.std
            )

            # --- Forward pass through the frozen classifier ---
            logits = classification_model(x)  # (N, num_classes)

            # Top-1 predicted class for each sample in the batch
            preds = logits.argmax(dim=1)  # (N,)

            # Ground-truth labels; ensure they are on the same device for comparison
            gt = class_target.to(device)

            # Per-sample correctness (True/False → int 1/0)
            correct = preds.eq(gt).int()

            # Accumulate (move to CPU / Python scalars to avoid holding GPU memory)
            all_predictions.extend(preds.cpu().tolist())
            all_gts.extend(gt.cpu().tolist())
            all_correct.extend(correct.cpu().tolist())

            # Optional: log running accuracy every 10 batches
            if (batch_idx + 1) % 10 == 0:
                running_acc = sum(all_correct) / len(all_correct) * 100
                print(
                    f"Batch [{batch_idx + 1}/{len(data_loader)}]  "
                    f"Running top-1 accuracy: {running_acc:.2f}%"
                )

    # --- Compute overall top-1 accuracy ---
    overall_acc = sum(all_correct) / len(all_correct) * 100
    print(f"\nFinal top-1 accuracy: {overall_acc:.2f}%")

    # --- Save results to CSV ---
    # One row per sample: ground_truth, prediction, correct (0 or 1)
    csv_path = os.path.join(args.out_dir, "classification_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth", "prediction", "correct"])
        writer.writerows(zip(all_gts, all_predictions, all_correct))

    print(f"Results saved to: {csv_path}")


def load_model(name):
    weights = get_model_weights(name).DEFAULT
    model = get_model(name, weights=weights)
    preprocess = weights.transforms()
    return model, preprocess


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.in_colour_space, args.out_colour_space = parse_colour_space(args.colour_space)

    model, num_embeddings = create_model(args.model_path, device)

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

    input_transform = transforms.Compose(input_transforms)

    data_loader = create_dataloader(
        args,
        input_transform,
        image_transform
    )

    # creating the classification model
    classification_model, classification_preprocess = load_model(args.classification_model)
    classification_model.eval()
    classification_model.to(device)

    run_image_classification(
        data_loader,
        model,
        mean,
        std,
        classification_model,
        classification_preprocess,
        args,
        device
    )


if __name__ == "__main__":
    main(sys.argv[1:])
