<h1> Yoga Pose Identification and Icon Matching </h1>

<h2> Project Goal </h2> 
<p> Detect yoga poses performed by a user and overlay a corresponding icon image. Running the main script starts the videostream with automatic pose detection. </p>

<h3> Part 1: Pose Detection </h3> 

<p> I use the 32 body landmarks provided by MediaPipe to measure joint angles, then determine yoga poses based on key joint angles for each pose. For example, in the star pose, the angle between the shoulder, elbow, and wrist landmarks (elbow flexion) are below 20 degrees and the angle of the elbow, shoulder, and opposite shoulder (shoulder flexion) are also below 20 degrees. </p>

<h3> Part 2: Icon Image Transformation </h3>

<p> To transform the icon image that will be overlayed over the user, I first preprocess the icon image then apply an affine transform. To preprocess the icon, I resize the icon image to be roughly the same heigt as the user, a metric also calculated with MediaPie's landmarks. I then apply a border to the icon image so that its image array has the same dimensions as the video stream frames. These steps help make the affine transform more effective. I select three key pose landmarks for each pose, then find three key points on the icon that should match these points. For example, I chose to match the nose and ankles of the person with the top tip and bottom two tips of the star. </p>
 
<h3> Part 3: Image Overlay </h3>

<p> I overlayed just the icon pixels (the icon background is ignored) by summing .5 of the icon pixel value with .5 of the the video frame value, resulting in a transparent overlay of just the icon. </p>

<h2> Results </h2> 

<h3>Star Pose</h3>

![Star Gif](star.gif)

<h3>Tree Pose</h3>

![Tree Gif](tree.gif)

<h3>Chair pose</h3>

![Chair Gif](chair.gif)
