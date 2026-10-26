# Steps to install Datasets

## DUTS

Follow the below steps to download and unzip the dataset. 
```
wget http://saliencydetection.net/duts/download/DUTS-TE.zip

unzip DUTS-TE.zip
```

Python alternative to extract zip: 
```
from zipfile import ZipFile

with ZipFile("DUTS-TE.zip", 'r') as f:
  f.extractall()
```

Place all the dirs/files extracted in this directory. Verify that these are present:
- DUTS-TE/DUTS-TE-Mask
- DUTS-TE/DUTS-TE-Image

## CUB

Official dataset page: <https://www.vision.caltech.edu/datasets/cub_200_2011/>.
Two files are needed: the images archive (`CUB_200_2011.tgz`, Caltech Data record
`65de6-vp158`) and the segmentation archive (`segmentations.tgz`, Caltech Data
record `w9d68-gec53`). Both records are also reachable from the official page.

Download and unzip:
```
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
wget -O segmentations.tgz "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1"

tar -xzf CUB_200_2011.tgz
tar -xzf segmentations.tgz
```

(`-O segmentations.tgz` avoids the awkward `?download=1` suffix in the filename,
which also has to be quoted in the shell because `?` is a glob character.)

Python alternative to extract tar:
```
import tarfile

file = tarfile.open("CUB_200_2011.tgz")
file.extractall()
file.close()

file = tarfile.open("segmentations.tgz")
file.extractall()
file.close()
```

Place all the dirs/files extracted in this directory. Verify that these are present:
- segmentations
- CUB_200_2011/images
- CUB_200_2011/train_test_split.txt
- CUB_200_2011/images.txt

## ECSSD

No manual loading required: `datasets/dataset.py` loads it online through the
Deeplake hub (`deeplake.load("hub://activeloop/ecssd")`). This needs a working
network connection and the `deeplake` package pinned in `requirements.txt`; it
cannot be run fully offline.

## CVC

Not downloadable from this repository. The loader expects
`datasets/CVC/metadata.csv` plus the image and mask paths listed in its
`png_image_path` / `png_mask_path` columns, relative to `datasets/CVC/`.
