# Learning DTW-Preserving Shapelets

## Description

This code is used to learn Shapelet features from time series that form an embedding such that 
L2-norm in the Shapelet Transform space is close to DTW between original time series. 

## Usage

To learn a model and use it to perform $k$-means clustering in the Shapelet Transform space, 
one should run:

```
python clustering.py DatasetName [Conv]
```

## Other implementations

A PyTorch implementation of the model is available at <https://rtavenar.github.io/hdr/parts/02/shapelets_cnn.html#Learning-to-Mimic-a-Target-Distance>.

