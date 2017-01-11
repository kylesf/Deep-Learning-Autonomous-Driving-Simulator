## Project 3 Documentation
#### By: Kyle Stewart-Frantz

****

Project Scope:
    
1. Gather Data
2. Data Preprocessing
3. Network Architecture
4. Train Network
5. Fine Tune Network 

[//]: # (Image References)

[image_0]: ./md_resources/1.png "Transformed Image"
[image_1]: ./md_resources/2.png "Transformed Image"
[image_2]: ./md_resources/Network.png "Network Design"

****
##### 1. Gather Data
I initially started this project in the beginning of December and was informed that the "way" to go was use the 50 Hz simulator on linux to generate appreciable amounts of training data. With constant networks evaluated I did not get appreciable results. In the end Udacity provided data to work with and that proved immediately successful.  

##### 2. Data Preprocessing  
Proper preprocessing took even long than expected. A lot of trial and error occured before reaching a basic implementation. Ideas tried include multiple resizing of the image and image color channel transformation. In the end the process for images was boiled down to three distict steps. First the frame was cropped from its original (160 x 320) to (103 x 320) [32:135,0:320]. Secondly, the image was resized to a 32 x 64 images still retaining all three of its color channels (RGB). Lastly, for every random image sampled it was flipped and the steering angle reversed.

It is also noted that in reading in the left and right images a steering angle of 0.16 was added and subtracted, respectfully. 

Final preprocessed images looked like the following:

![alt text][image_0]

![alt text][image_1]


##### 3. Network Architecture
The first network I spent time implementing was nvidia strucutre for end to end learning. In the end I did not have much luck with this network, even though I had mad sure to implement it correctly. After many trials and tribulations I ended up on an architecture as seen below. 

![alt text][image_2]

##### 4. Training Network 
The network was trained using a custom batch generator that fed batch sizes of 128 samples into the model. The batch of 128 consisted of 64 images and thier flipped counterpart. Training was done for 5 epochs on an p2.xlarge aws custom instance.  

##### 5. Fine Tune Network 
Once the network was working okay around the track, I changed the model to fine tune mode in which I loaded the previous weights and models and tuned them. This for problem areas on the track. I collected new data in those trouble areas using the mac osx simulator and then fine tuned the model until it was able to endlessly run on the first track. The model works great on the second track as well despite never seeing it! 
