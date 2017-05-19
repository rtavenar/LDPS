# Learning to mimic DTW

## Description

This code is used to learn Shapelet features from time series that form an embedding such that 
L2-norm in the Shapelet Transform space is close to DTW between original time series. 

## Usage

To learn a model and use it to perform $k$-means clustering in the Shapelet Transform space, 
one should run (if no distance file is available for the dataset, DTW distances will be computed 
on the fly, making the process somewhat slower):

```
python clustering.py DatasetName [Conv]
```



