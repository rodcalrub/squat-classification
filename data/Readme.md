# Squat Dataset
This is the pose estimation result of "Single Individual Dataset" used in "Temporal Distance Matrices for Squat Classification" (CVPRW2019).

## Overview
One person performed squats in various places and made inferences using the method of Kanazawa et al. ([Hmr] (https://akanazawa.github.io/hmr/)).

## Data structure
### Annotation method
We filmed a squat video and classified it into one of seven types.
It is divided into folders for each squat class, and each file (*.json) is the result of the pose inference of the estimated squat video.

### About the contents of the json file
```
{
    "0":
    {
        "index": 0,
        "3d_joint": [0.4779390275592104, 0.7586504650537319, 0.7917826766099008, ...
    "1": ...
    }
}
```
| パラメーター | 説明 | 大きさ |
|:----------:| :----------:|:------------:|
| Parameter | Description | Size |
| "Number" | Frame number | 0 ~ 300 frame |
| "3d_joint" | Pose estimation result | 171 point [1] |
| | |

[1] The size of 3d_joint is obtained by extracting the upper triangular matrix part of the estimated 19 key points.