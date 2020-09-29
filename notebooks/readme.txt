The folder contains six Jupyter-Notebooks:
 - "lesion.ipynb"
   This notebook reads the computed transformation matrices and apply them to RGB CUBE.
   
 - "lesion_transform.ipynb"
   Computes the linear transformation for each lesion.
   Please note for demo purposes, we compute the transformation matrix for each image on the fly. The results reported in the mansucript are obtained with the transformation matrices computed over aggregation of al images.

 - "report_celeba.ipynb"
   Reads and reports the CIE DeltaE-2000 for the validation-set of CelebA dataset.

 - "report_coco.ipynb"
   Reads and reports the CIE DeltaE-2000 and IoU for the validation-set of COCO dataset.
   It also reads the histogram of embedding vectors and computes its correlation to the IoU.

 - "report_imagenet.ipynb"
   Reads and reports the CIE DeltaE-2000 and IoU for the validation-set of ImageNet dataset.
   It also reads the histogram of embedding vectors and computes its correlation to the classification accuracy.

 - "vectors_hue.ipynb"
   It loads all trained model and generate samples from them by inputting them a reconstruction latent of single vector index.
   This notebook shows the colours those networks generate when only one vector is used.

The results directory can be downloaded from this link: https://www.dropbox.com/sh/e1l3p3uot94q0fy/AADg0rmxyiC3UNifTtqIpg2Pa?dl=0
