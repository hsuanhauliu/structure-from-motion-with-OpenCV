# Structure From Motion with OpenCV
The program follows the process below:
1. Look for SIFT features from two images of a scene captured from different angle.
2. Compute local matches and find closest neighbors.
3. Filter out bad neighbors by comparing distance.
4. Draw remaining matches on the image and output it.
5. Find inlier matches by calculating the essential matrix and mask with RANSAC.
6. Find rotation and translation matrices of one image.
7. Compute 3D point cloud and plot the graph.

## Results
### Pair 1
![screenshot](input_data/a1.png "") ![screenshot](input_data/a2.png "")

![screenshot](output/inlier_match_1.png "")

![screenshot](output/3-D_1.jpg "")

### Pair 2
![screenshot](input_data/b1.png "") ![screenshot](input_data/b2.png "")

![screenshot](output/inlier_match_2.png "")

![screenshot](output/3-D_2.jpg "")


### Pair 3
![screenshot](input_data/c1.png "") ![screenshot](input_data/c2.png "")

![screenshot](output/inlier_match_3.png "")

![screenshot](output/3-D_3.jpg "")

### Feature Matches
