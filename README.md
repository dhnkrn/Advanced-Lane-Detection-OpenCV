## README
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


### Camera Calibration

Camera calibration is a required step correct for distortion introduced by the camera assembly and as such is specific to a particular camera. The approach for calibration is to take photos of a set of objects whose pixels positions are known and then quantify how much distortion the camera has introduced in the images. The code for this step is contained in the code cell 3 of the IPython notebook "./solution.ipynb"

The reference object is a chessboard and the reference "object points" are the (x, y, z) coordinates of the chessboard corners. Here, the chessboard is fixed on the (x, y) plane with z=0. The object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time chessboard corners are detected in a test calibration image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Here's a calibration test image with the detected corners overlayed.

![alt text](https://github.com/dhnkrn/Advanced-Lane-Detection-OpenCV/blob/master/output_images/camera_calibration.png?raw=true)

The output `imgpoints` is fed to `cv2.calibrateCamera()` function to compute the camera calibration and distortion coefficients. Correcting distortion using the computed coefficients with `cv2.undistort()` looks like this.
 ![alt text](https://github.com/dhnkrn/Advanced-Lane-Detection-OpenCV/blob/master/output_images/camera_cal_undistort.png?raw=true)

### Pipeline for image processing individual frames

#### 1. Distortion Correction

Images before and after correction distortion.

 ![alt text](https://github.com/dhnkrn/Advanced-Lane-Detection-OpenCV/blob/master/output_images/distortion_correction2.png?raw=true)


#### 2. Color transformation and gradients

A combination of color filtering in the HSV color space and gradient thresholding using a Sobel filter is applied to identify lanes. The identified lane pixels are max-thresholded to produce a binary image.

![alt text](https://github.com/dhnkrn/Advanced-Lane-Detection-OpenCV/blob/master/output_images/hsv.png?raw=true)
![alt text](https://github.com/dhnkrn/Advanced-Lane-Detection-OpenCV/blob/master/output_images/s_channel.png?raw=true)
![alt text](https://github.com/dhnkrn/Advanced-Lane-Detection-OpenCV/blob/master/output_images/sobel.png?raw=true)
![alt text](https://github.com/dhnkrn/Advanced-Lane-Detection-OpenCV/blob/master/output_images/thresholded_combination.png?raw=true)
![alt text](https://github.com/dhnkrn/Advanced-Lane-Detection-OpenCV/blob/master/output_images/thresholded_combination_1.png?raw=true)

#### 3. Perspective Transform

The code for my perspective transform includes a function called `transform_perspective()`, which appears in the code cell 12 of the IPython notebook.  The function reads an image of a straight road with two visible lanes. Four pixel co-ordinates of the lane lines forming a trapezium are chosen and then spatially transformed to make a rectangle. The idea is to eliminate the camera perspective on the image. This, like camera calibration, is dependent on the camera setup and the coefficients are calculated just once. The calculated co-efficient are then reused in the pipeline to transform each frame.

```python
   src = np.float32([[325,650],[990,650],[420,580],[880,580]])
   dst = np.float32([[300,670],[1000,670],[300,600],[993,600]])
```

![alt text](https://github.com/dhnkrn/Advanced-Lane-Detection-OpenCV/blob/master/output_images/perspective_transform.png?raw=true)

#### 4. Lane Detection

This is an interesting part of the pipeline that needs relatively more optimization than the other stages. A square filter of 1's is slided across the frame bottom-to-top and left-to-right to calculate the number of 'high' pixels in the thresholded binary input image. The lane line pixels are expected to be thresholded to 'high' in the color and gradient filtering steps. The algorithm here has to basically pixels that not lane lines. I use a combination of techniques to identify lane pixels.
1) Lane pixels are brighter in the thresholded binary image and the pixels appear next to each other. This means the cross-correlation with a 2d filter of 1's will be high.
2) Simplify lane line identification by only choosing one point per left and right lane along the horizontal. The chose point representing the lanes has the highest cross-correlation in the corresponding half of the image.
3) After sliding the filter from the bottom to top, two sets of points one for each lane is then used to fit a 2nd order polynomial.
4) A runnning average of the polynomial co-efficients is used to filter out noise. The final detected lane is combination of past history and the current detection. 
5) The current measurement is outright rejected if it is 1.3 standard deviations away from the mean. In this case, the detection is solely based on the past data.

![alt text](https://github.com/dhnkrn/Advanced-Lane-Detection-OpenCV/blob/master/output_images/histogram.png?raw=true)

![alt text](https://github.com/dhnkrn/Advanced-Lane-Detection-OpenCV/blob/master/output_images/lane_detection.png?raw=true)
	


#### 5.Radius of curvature and off-center position

The radius of curvature computation is the same as the one provided by Udacity and is in calc_roc() function. The position of vehicle w.r.t center is calculated as difference between lane center and image center. This is in the detect_lanes() function.


#### 6. Lane overlay

This code again is the same as the one in Udacity's example. Here's the final output image.

![alt text](https://github.com/dhnkrn/Advanced-Lane-Detection-OpenCV/blob/master/output_images/output.png?raw=true)

---

### Pipeline processing for video input

See project_video_out.mp4

---

### Discussion
The Computer Vision approach employed in this project to detect lanes comprises of several hand-tuned parameters. This is susceptible to failing on inputs that the model was not optimized for. This was very evident, in my case, when the pipleline failed to detect lanes when the lines in the frame were short. This happens in cases when the car's front has just passed lane, which subsequently led to jitters every few frames. Apart from not detecting lanes, sometimes the pipeline detects lanes even if there aren't enough points to fit a line. This leads to lines in that are way off in direction. I had to resort to relying on the past data to handle such cases. Relying on past data could be again problematic when there are sharp turns involved. The running average filter takes more time to adjust to the new angles. These are the problems I noticed while detecting lanes in an input video that has clearly marked lanes, lanes are of only two colors and shot in sunny weather. This pipeline would completely fail if the pixel distribution changes due to overcast conditions, bad lane markings or several other possible conditions. However, one particular advantage in the CV approach is the lower demand on data and computational resources.

