### Goals

The goal of this project was to design and implement a program that can segment and track eels moving in a video. We were provided with a video of eels swimming in a tank, and the goal was to be able to scan the video and determine the segments in the video where the eels appear.

In particular, the goals were:
  1. Dynamically find the boundary box of the two tanks.
  2. Automatically find the portions of the video with animal action.
  3. Segment the eels from the background.
  4. Track the movement of the eels.
  5. Analyze the movement behavior of eels. Eels make wave-like movements with their bodies. Design an algorithm that describes this movement, in particular, the frequency of undulations.
  6. Output a video of the eels being segmented and tracked in the original video.
  7. save the times when the eels are in the open feeding area (with white background) and the estimated undulation frequency in a CSV file.

### Approach

To process the video, I simply read each frame of the video in a loop. In order to speed up processing, I only process every X number of frames, as defined by `FRAME_PROC_RATE` (5 or 10 were good numbers). On each frame of the video, I would call my `process_frame()` function. In order to get a good segmentation of the eels, I first (manually) found a frame of the video which did not contain any eels. This frame, which I called a 'mask', is defined by the `REFERENCE_MIN` and `REFERENCE_SEC` variables (the minute and second that the frame appears at in the video). I jump to this time in the video, and call my processing function on that frame. Since this is the first time the function is called, it handles the frame differently than it will later.

After converting the image to grayscale, it first finds the bounding box of the tank. To do this I did a fairly straightforward thresholding, and determined the bounding box based on a histogram of the white pixels. The bounding box is only found during this initial processing, and this same bounding box is used on future frames (i.e. I assume that the camera is not moving dramatically throughout the experiment).

The processing of the image is also fairly simple. I first do an adaptive threshold to pick out the object in the bounding box. I then blur that image, then do another thresholding operation on the image within the tank bounding box - that's it! During the first call of this function (the one that doesn't have any eels), we set the `mask` Mat to our processed frame.

Then, I set the video back to the whichever starting place is defined in the program (determined by the `START_MIN` and `START_SEC` variables.) I used these so I could go straight to the interesting segments of the video and they could easily be set to the beginning of the video. Each time a frame is processed, the same processing outlined above is done to the frame (using the same bounding box found in the initial call.) The `mask` is then essentially XOR'ed with the processed frame (with my `subtract()` function).

Using this method, I'm able to pick out differences between the masking frame and the current frame. With this resulting frame, I'm able to pick out the eels! To find them, I used a stack-based connected component algorithm, which I outlined in the last assignment (you can find more information  [here](http://gremerritt.herokuapp.com/cs585/2016/10/05/hw3.html)). Since the eels are relatively big, and the errors in my masking method are small, I can simply ignore small objects picked out by the connected component algorithm, and be fairly sure that the eels will be picked up correctly. I also calculate the boundary of the object, which I use just for display purposes.

Next, I determine the skeleton of each of these objects that I picked out. The algorithm works by going over each pixel in the object, and adding it to the skeleton if its minimum distance to the background is greater than or equal to the those of it's neighbors. I then pick out 3 points on this skeleton which are (or are attempted to be) evenly spaced - this is done in my `getPointsOnSkeleton()` function. If the object is skinny and tall I sort the skeleton points vertically, and if the object short and wide I sort the points horizontally. I then pick my three points to be the first point, middle point, and last point. I also built the function so that I could use more points (I tried out 4 and 5) but the results were really best with just 3 points. These three points are displayed on the frame, so you can get an idea of how well this methods works.

Finally, I calculate the curvature of the eel. To do this, I used a measure of curvature outlined in the Williams and Shah paper on Active Contours ([A Fast Algorithm for Active Contours and Curvature Estimation](http://www.vision.eecs.ucf.edu/papers/shah/92/WIS92A.pdf))

|<sup>u<sub>i</sub></sup> / <sub>|u<sub>i</sub>|</sub> - <sup>u<sub>i+1</sub></sup> / <sub>|u<sub>i+1</sub>|</sub>|<sup>2</sup>

If more than 3 points on the skeleton were used, I would calculate all the curvature values and then take the average.

Lastly I calculated the change in curvature from the last frame. To do this I used a fairly simple matching algorithm to determine which eel in the current frame was matched to an eel in the previous frame. Each time a frame is processed, I keep track of the eels in that frame, including where that eels is (its center of mass) and it's curvature. Then in the current frame, I have another list of the eels in the frame, and their locations and curvatures. The objects that are closest together are said to be the same object. For example, if there were two eels in the last frame, and three in the current frame, then the eel in the current frame with the closest distance to an eel in the last frame are matched. These two eels (one in the current frame, one in the last frame) are then taken out of consideration. The process is done again. Since there is still an eel in the current frame with no match in the last frame, we just ignore it and essentially assume it 'just appeared'.

The difference in curvature divided by the `FRAME_PROC_RATE` (i.e. 'frequency of undulation') is then written to a file, along with the timestamp in the video. Thus, we could run the program over the entire video, and then we could check the file to see the locations in the video where there were eels, and how much their curvatures were changing. Hopefully, this is a good measure of how much distress that eel is in!

A GIF of my results can be seen below, along with a sample of the file produced.

![eels.gif](eels.gif)

    time,change_in_curvature1,change_in_curvature2,...
    0:25:24,0.001636
    0:25:191,0.142991
    0:25:692,0.000409
    0:25:859,0.119079
    0:26:25,0.116096
    0:26:192,0.047265
    0:27:360,0.088157
    0:27:527,0.000853
    1:2:896,0.048859
    1:3:62,0.070057

### Compiling and Running

You'll first want to download the video of the swimming eels
[here](https://www.dropbox.com/l/scl/AADWW-fklT1cWhwrhG3yi5xgXzRu-AmzB08) and make sure it's in this project directory.

We use OpenCV here to process the videos. As it stands, this is written to be ran on OSX with OpenCV installed via Homebrew. The following steps should allow you to compile and run the program.

    $ brew install opencv3 --with-ffmpeg

Then `cd` into the project repo, and run

    $ cmake .
    $ make

Then run the program with

    $ ./main
