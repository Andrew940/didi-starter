
# Background

Perception unit is a key component of self driving car. It relies on various sensors such as LIDAR, CAMERA and RADAR to identify and locate objects around the car. 

The most important objects to be localized are other vehicles.

[This is a work in progress for UDACITY-DIDI challenge.]

[github: https://github.com/experiencor/didi-starter/tree/master/simple_solution]

**Problem**

Given the LIDAR and CAMERA data, determine the location and orientation in 3D of other vehicles around the car.

**Solution**

2D object detection on camera image is easy and can be solved by various CNN-based solutions like YOLO and RCNN. The tricky part here is the 3D requirement. It becomes a little easier with LIDAR data, accompanied with the calibration matrices.

So the solution is straight-forward in three processing steps:

+ Detect 2D BBoxes of other vehicles visible on image frame captured by CAMERA. This can be achieved by YOLOv2 or SqueezeDet. It turns out that SqueezeDet works better for this job and is selected.
+ Determine the dimension and the orientation of detected vehicles. As demonstrated by [https://arxiv.org/abs/1612.00496], dimension and orientation of other vehicles can be regressed from the image patch of corresponding 2D BBoxes.
+ Determine the location in 3D of detected vehicles. This can be achived by localizing the point cloud region whose projection stays within the detected 2D BBoxes.

**Results so far**

https://youtu.be/iesJ-6QeCOQ

**To do list**

+ Complete processing step 3, which is to locate the position of the car using point cloud.
+ Run the whole processing pipeline on UDACITY data, which contains lots of errors at the moment.

# 2D BBox Detection

The basic idea is to divide the image into a 13x13 grid. Each cell is responsible for predicting the location and the size of 5 2D BBoxes. The loss function at the output layer is:

$$\begin{multline}
\lambda_\textbf{coord}
\sum_{i = 0}^{S^2}
    \sum_{j = 0}^{B}
     L_{ij}^{\text{obj}}
            \left[
            \left(
                x_i - \hat{x}_i
            \right)^2 +
            \left(
                y_i - \hat{y}_i
            \right)^2
            \right]
\\
+ \lambda_\textbf{coord} 
\sum_{i = 0}^{S^2}
    \sum_{j = 0}^{B}
         L_{ij}^{\text{obj}}
         \left[
        \left(
            \sqrt{w_i} - \sqrt{\hat{w}_i}
        \right)^2 +
        \left(
            \sqrt{h_i} - \sqrt{\hat{h}_i}
        \right)^2
        \right]
\\
+ \sum_{i = 0}^{S^2}
    \sum_{j = 0}^{B}
        L_{ij}^{\text{obj}}
        \left(
            C_i - \hat{C}_i
        \right)^2
\\
+ \lambda_\textrm{noobj}
\sum_{i = 0}^{S^2}
    \sum_{j = 0}^{B}
    L_{ij}^{\text{noobj}}
        \left(
            C_i - \hat{C}_i
        \right)^2
\\
+ \sum_{i = 0}^{S^2}
L_i^{\text{obj}}
    \sum_{c \in \textrm{classes}}
        \left(
            p_i(c) - \hat{p}_i(c)
        \right)^2
\end{multline}$$

**Reference:** YOLO9000: Better, Faster, Stronger (https://arxiv.org/abs/1612.08242). 

This is the output

<img src="images/2DBox.png" width="400"/>

### Issues and Fixes

1. We don't use the CNN architecture of YOLOv2 but directly use that of VGG16. The reason is that YOLOv2 fails to detect the car on the Udacity test set, although it is able to detect the car on the KITTI test set. We suspect that different preprocessing steps are the root cause.

2. We also try SqueezeDet and it turns out to work more reliable than YOLOv2. So SqueezeDet is used in the end.

# 3D BBox Detection

The network is formulated to learn three different tasks concurrently: the dimension, the confidence and the relative angle.

<img src="images/model_3d.png" width="400"/>

Squared loss and anchors are used to regress dimensions. Anchors are also used to learn orientation. Cross entropy loss is used to regress confidences. The custom loss function to learn orientation is:

$$L_{loc} = \frac{1}{n_{{\theta}^*}}\sum \cos(\theta^* - c_i - \Delta\theta_i)$$

**Reference:** 3D Bounding Box Estimation Using Deep Learning and Geometry (https://arxiv.org/abs/1612.00496). 

This is the output

<img src="images/3DBox.png" width="400"/>

### Issues and Fixes

1. The confidence loss get stuck at 0.6. This turns out to be the result of poor weight initialization. So we train the network initially with squared loss. When the weights are settled, squared loss is replaced by cross entropy loss.

2. We need to change the weights for the losses many times to further improve the train loss and test loss. For example, when the orientation loss gets stuck, we need to increase its contribution to the final loss to improve it faster.

# 3D Localization

The localized point cloud region corresponding to a detected vehicle can be determined via the calibration matrices and 2D BBoxes. The point loud is expected to be very noisy and good localization heuristics will takes some time.
