# AeroLoop:

Not just an edge ML project, but a project for building Edge AI solutions.

One of the common themes of questions and struggles I noticed in the Edge Impulse Hackathon webinar and Discord channel is a challenge that faces all ML/AI projects, and it is perhaps the most crucial in being able to train models that generalise well in the real world... collecting data. Not just data, but varied and representative data. Data collected in the lab (lab conditions) is often too homogenous, and can't produce the variance seen in the real world. Collecting data is actually the easy part. I can collect 24 hours of audio every day if I like, but if only  hour of those 24 is of interest to our problem, that's a lot of audio to annotate every day.

Take that one step further, if only one hour in the day is of interest to us (ie our target class), then we still need to balance our training dataset with samples representing the negative, or non-target class. The key term here is balance. We could train on everything, 23 hours of negative class data and one hour of positive class data - but this would be a highly inefficient use of time and resources.

So, collecting data is easy. Collecting target class data is also somewhat easily done and defined. The biggest issue faced with training robust ML models for the real world, is collecting difficult edge cases that "confuse" a model.

How can we overcome this? How can we "find" only the most confusing samples from those 23 hours? This is where AeroLoop shines as a Smart Dataset Curator.

AeroLoop is essentially a closed CI/CD loop for collecting, annotating, training and deploying an Aircraft sound detection model with Edge Impulse.

## Collecting Aircraft Samples
In order to collect/record aircraft samples, we use an SDR (software defined radio) and decoding software called "dump1090" to monitor the positions of nearby aircraft. When an aircraft comes within 3km of the recording device, a one-minute recording is triggered to capture the aircaft as it flies overhead. 

## Negative Class Samples
Conversly, negative class samples on the are recorded opportunistically when there are no aircraft within 10km's of the recording device. During high air-traffic times, this can mean that very few negative class samples are collected, and vice-versa during low air-traffic times.

## The Problem
Negative class samples are essentially collected at random, so we want to collect as many negative class samples as possible to ensure we have good edge cases, but... 
- We don't want to annotate 23 hours of audio per day, and
- We don't want to overload our training dataset with "easy" samples


## The Solution
The solution to this problem is to run inference on the field recording device at the time of collection - if our model is already "confident" a sample belongs to the negative class (ie, stays below a defined threshold) - then we simply delete the sample right there and then. This has a number of compounding benefits for the efficinet use of time and resources (ie, value):
- Less time spent on human annotation (~30 seconds per negative class sample)
- Smaller, but more balanced and relevant dataset
- Faster training and convergence


## Success Metrics
AeroLoop sounds good in theory, but how will we know if it works in practice?


## Device Specific Considerations
Aircraft detection may seem like a fairly simple machine listening task. And, essentially it's quite straightforward when you're not constrained by device limitations (eg, on a PC). Although the AeroLoop field recorder runs on a RaspberryPi4 with 4GB of RAM - our final target device is an Arduino Nano Sense, a microcontroller with just 256 KB SRAM and 1MB flash memory. Not only does this constrain our model architecture decisions, but also our feature extraction/DSP pipeline. So before we deploy AeroLoop, we must first ensure that DSP and inference will run on the Arduino Nano.


## Domain Specific Considerations
Aircraft come in all sorts of shapes, sizes and types. The noise they emit is a combination of engine noise, and wing/body noise. Something here about car and engine noises, tyres on the road and air pressure - when isolated to a 2-second sample, it can be difficult to discern between a car nearby, or a plane overhead. Every day, conditions change and affect our samples. Some days aircraft will be approaching the runway (landing), other days aircraft might be taking off. Some aircraft pass over at 5,000 ft, others might pass at 1,500 ft. Some days a jackhammer, or lawn mower might persist in the background. Some days are raining, others are windy - all the real world noisy stuff that we can't account for until we encounter it.


## Compounding Iterations
So how is this project a "Project for building Edge Impulse projects"? Aircraft detection is simply our first step. As the detection model improves over time, the dataset balance will begin to shift from predominantly negative class samples, towards mainly aircraft samples. Now that we have a growing dataset of aircraft samples, we can actually start to work on higher-level machine learning tasks, such as; predicting altitude (regression), predicting if the aircraft is taking-off or landing, classifying different types of aircraft (eg, Turbofan, Turboprop, Piston prop or Helicopter), predicting a make/model (eg, Boeing 777, Airbus A320).
Essentially, AeroLoop is a smart collection service. The longer it runs, the smarter it gets. The smarter it gets, the better the data it collects. With more data, comes more ML capabilities.


## how to use it - how does it work?
Firstly, AeroLoop is a system for collecting, annotating, training and deploying aircraft noise ML models to edge devices. Our demonstration project for the hackathon is to deploy an aircraft detection model to an Arduino Nano microcontroller.

So we've first defined the what (aircraft noise detection), and the how (Arduino MCU) - but we should quickly go over the why and where. The where is simply my house (close proximity to a flight path). The why is actually quite a long list of potential applications for deploying aircraft detection on low-powered sensors. AirServices Australia deploy Environmental Monitoring Units around Australian airports to help omnitor Air Traffic noise and deal with noise complaints. Noise-sensitive buildings like schools, universities and hospitals, or specific industrial facilities where noise can negatively impact their commercial operations. There are many applications in military/defence - in particular with border crossing and smuggling detection (smugglers often use small planes at low altitude to avoid detection by traditional radar).

OK, so we've got our what, where and why - how do we use AeroLoop to help build this thing?

our very first step, should be to stress test our MCU to see what it is capable of. We need to ensure both the DSP and inference blocks "fit" within our devices memory and processing capabilities. In Edge Impulse, the DSP parameters are defined by the impulses' "window size" and feature generation block. If the window is too large, our MCU won't have enough memory to hold the raw features - too small and the model might not have enough information to make informed decisions. Our feature generation block can also be thought of as a "dimensionality reduction" step to help us compress the raw data so it is more readily digestable by our model. At the end of the day, this process really depends on the task at hand. A gunshot detection model will work with a window size of 1-second or less, whereas an aircraft detection model would benefit from a larger window size like 5 or 10 seconds.

For this project, we wanted the window size as large as possible - but knew it would come at the expense of richer features being passed to the model. After testing with various sizes, we arrived at a window size of 2-seconds with an MFE block that aggressively reduced the dimensions from 32,000 to 1,984.

### Model architecture and inference time
Model architecture is another critical trade-off between device constraints and model performance. A complex architecture may perfrom well in the lab, but if it takes longer to compute than the inference window - then we're not monitoring in real-time. One excellent way to make neural networks more computationaly efficient, is to use convolutional layers as another feature generation/dimensionality reduction technique. The more layers, the greater the reduction in dimensions. After passing through 4 Conv2D layers, we turn 1,984 features into just 128 in our final hidden layer.

### Stress Testing
Edge Impulse actually makes is possible to estimate how the DSP and inference block will perform on your target device. But before we do that, we should get some initial data into the Edge Impulse studio.

Guided by the theory above, we can then "play around" with DSP and model architecture parameters, taking note of on device processing times and memory requirements.


## Initial data
To get a new project up and running ASAP, we need some baseline data to help us stress-test the target device, and to train a benchmark "dumb" model that we can build upon and hopefully improve over time.

For this, I collected audio samples over an approximately 25 hour period. I labelled the dataset and published it to Zenodo with a commercial open-source license. This initial dataset took approximately 4.5 hours of human annotation time.

### LOOM THIS INTO A GIF
Download the dataset from [https://zenodo.org/records/17666930](https://zenodo.org/records/17666930)
Unzip the 'aircraft-test', 'aircraft-train', 'negative-train' and 'negative-test' folders
Upload each folder with their respective train/test category and labels

## Data and Device Exploration
Select the target device in EI studio (Arduino Nano BLE Sense)
"Create Impulse"


## Deploy Impulse to Target Device to ensure it "Fits"
<iframe width="640" height="348" src="C:\Users\Blake\Repos\aero_loop\submission\images\Videos_Library_Loom-22November2025-ezgif.com-video-speed.mp4" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>


## More words