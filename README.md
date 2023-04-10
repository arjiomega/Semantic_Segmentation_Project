# Semantic Segmentation Project

## Virtual Environment Setup
```bash
conda create --no-default-packages -n <env_name>
conda activate <env_name>
conda install python=3.9
```

## Load packages
```bash
pip install -e .
```

## Running test
```bash
pytest --verbose
```

## Updates in the future
1. Work on multiple classes
2. Include bounding boxes (xml file)
3. try on the following datasets (Cityscapes, PASCAL VOC, ADE20K,COCO,SUN RGB-D, Mapillary Vistas, GTA5)

## Dataset(s) being used
[[1] OXFORD-IIIT PET Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## References:
[[1]](https://www.robots.ox.ac.uk/~vgg/data/pets/) O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
   Cats and Dogs
   IEEE Conference on Computer Vision and Pattern Recognition, 2012