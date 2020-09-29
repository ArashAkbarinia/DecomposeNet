# DecomposeNet
Decomposing an input image into different features.


# Train
The "src" folder contains the source code to train a colour-converting VQ-VAE.
The "arguments.py" specifies the input arguments the "main.py" accepts.

For instance to train a standard VQ-VAE:
python main.py --model vqvae --batch-size 2 --dataset imagenet --data_dir <PATH/TO/IMAGENET> --hidden 128 --k 128 --kl 128 --target_size 224 --max-epoch-samples 50000 --results-dir <PATH/TO/OUT> --colour_space rgb2rgb

The argument "colour_space" defines the input and output colous spaces.
The first three letters are for the input.
The last three letters are for the output.

Examples:
 - rgb2dkl
 - rgb2lab
 - rgb2lms
 - lms2dkl
 - lms2rgb
...
