## AeroLoop: Smart Data Curation & Remote CI/CD at the Edge - Raspberry Pi4 with Software Defined Radio and Arduino Nano 33 BLE Sense

Created By: Blake Downward

Edge Impulse Project:

GitHub Repository:

### TODO: Hero image/GIF



> Collecting data is easy. Curating **valuable** data is the hard part.

## Introduction
Collecting audio is easy; collecting audio that actually improves an edge model is hard. AeroLoop is a field recorder plus remote CI/CD loop for aircraft noise: an SDR on the Pi gives ground-truth "plane nearby", and the current model runs on the Pi to drop easy negatives and keep only confusing edge cases. That lets us train ultra-light models in situ on the Arduino Nano 33 BLE Sense (2 s MFE + compact CNN that fits RAM/flash) with fast turnaround instead of spending 4-5 hours annotating audio each day. Deploy the loop to more sites and you get both rapid, location-specific robustness now and the option to fuse those curated datasets later for tougher tasks like altitude or aircraft type recognition.


<!-- Collecting audio is easy. Annotating audio is laborious. In this project we will deploy AeroLoop - a smart field recorder for collecting and annotating aircraft audio events, then training and deploying an aircraft detection model in a remote CI/CD loop which gets "smarter" as new data arrives. AeroLoop is not just an Edge AI project... it is essentially an project for building aircraft audio projects with Edge Impulse. So to demostrate this, we will aim to deploy a continuous aircraft detection system to an Ardino Nano Microcontroller. Aircraft events are rare, environments are noisy and location-specific (traffic, construction, rail, wind), and “one-size-fits-all” models often fail in situ. AeroLoop tackles this by running a closed-loop, on-device DataOps workflow: a Pi uses SDR to know when a plane is present, and the current model filters out “easy” negatives so we keep only the confusing edge cases. That lets us train ultra-light models locally with fast turnaround instead of hoarding 24 hours of audio a day. Deploy the loop to many sites and you get both: rapid, site-specific robustness now, and the option to fuse those curated datasets later for more ambitious tasks (altitude regression, aircraft type/family differentiation) without changing the workflow. -->


## The Problem

Aircraft events are relatively sparse throughout a day. Collecting 24 houurs of audio per day is simply not feasible for human annotation. It would flood the dataset with irrelevant/easy negative class samples, which would lead to major class imbalances and wasted resources by extended training times. In the real world, different recording times or locations will have different background sounds that can be confused with aircraft noise (road traffic, construction, rail, wind, rain), so if we can focus directly on collecting these "confusing" sounds - we can curate the "right" data in a much more efficient manner. We need a way to keep only valuable samples, adapt quickly to each location, and continuously iterate in situ instead of shipping gigabytes of irrelevant data to the cloud.

## The Solution

### Experiment Log (fill as you iterate)
| Iteration | Date | Model Used | New aircraft minutes | New negative minutes | Total new minutes | Train class mix (aircraft:negative) | Train/Test split | Negative drop threshold | Negative drop rate | Model ID | Accuracy | Notes |
|-----------|------|------------|---------------------|---------------------|---------------------|------------------------------|------------------|------------------------|-------------------|----------|----------|-------|
| 0 (baseline dataset) | 1st, Nov, 2025 | N/A | ~40.5 | ~102.1 | 142.6 | ~28:72 | ~50:50 | N/A | 0% | baseline | 94.7% | Baseline EI model; stress-test DSP/inference fit on Nano. |
| 1 | 23rd, Nov, 2025 | baseline | ~32.8 | ~6.2 | ~39.0 | ~48:52 | ~61:39 | 0.4 | 91% | model_20251123_082259.tflite | 94.7% | First training cycle; triage running on Pi with baseline model. |
| 2 | 24th, Nov, 2025 | model_20251123_082259.tflite | ~13.0 | ~3.2 | ~16.2 | ~52:48 | ~64:36 | 0.4 | 96% | model_20251123_215721.tflite | 95.3% | Second training cycle; model improved and deployed back to Pi/Nano. |
| 3 | 24th, Nov, 2025 | model_20251123_215721.tflite | ~18.2 | ~3.3 | ~21.5 | ~57:43 | ~67:33 | 0.4 | 93% | model_20251124_070130.tflite | 95.6% | Third training cycle; 53 new samples uploaded (43 aircraft, 10 negative). Aircraft samples also difficult because they were predominantly taking-off and at higher altitudes. Model improved and deployed back to Pi/Nano. |
| 4 | 25th, Nov, 2025 | model_20251124_070130.tflite | ~23.3 | ~3.0 | ~26.3 | ~62:38 | ~71:29 | 0.4 | 98% | model_20251124_215810.tflite | 95.9% | Fourth training cycle; 59 new samples uploaded (50 aircraft, 9 negative). Continued improvement with hard-negative mining. Model improved and deployed back to Pi/Nano. |
| 5 | 25th, Nov, 2025 | model_20251124_215810.tflite | ~23 | ~32 | ~55 | ~57:43 | ~TBD | 0.4 | 0% | model_20251125_074241.tflite | 96.4% | Fifth training cycle; 151 new samples uploaded. Model improved and deployed back to Pi/Nano. |
| 6 | 25th, Nov, 2025 | model_20251125_074241.tflite | 3.4 | 8.3 | 11.7 | ~56:44 | ~TBD | 0.4 | 0% | N/A (not deployed) | 96.1% | Sixth training cycle; 25 new samples uploaded. Accuracy did not improve (96.13% < 96.38%), model not deployed. |
| 7 | 26th, Nov, 2025 | model_20251125_074241.tflite | 5.8 | 2.0 | 7.8 | ~56:44 | ~TBD | 0.4 | 0% | model_20251126_012301.tflite | 96.9% | Seventh training cycle; 27 new samples uploaded (21 aircraft, 6 negative). Model improved and deployed back to Pi/Nano. |
| Next iterations | TBD | model_20251126_012301.tflite | TODO | TODO | TODO | TODO | TODO | 0.4 (or updated) | TODO | TODO | TODO | Fill after each download/annotate/train/deploy cycle. |



**Monitor 24/7, BUT only record and keep samples that will improve our current model**

Essentially, this means deploying you best current model to your field recording device and immediately running inference on each sample. If a negative class sample is correctly identified as negative on device - then we can delete it right there and then. Conversly, if a negative class sample is incorrectly identified as an aircraft - this means our model is somewhat confused by this edge case, and including it in our dataset will help future iterations to discern between aircraft and other background noises.

Simple. Right?

Simple in thoery, but a little more convoluted in practice.

The difficulty in performing the above is actually in how we determine our "ground truth" class labels - ie, how do we know to label aircraft samples as aircraft, and vice-versa?

We do this by deploying a software-defined radio (SDR) to a RaspberryPi and monitoring aircraft ADS-B messages (positional information transmitted by aircraft). By monitoring the locations of nearby aircraft, we can trigger an "aircraft" recording when a plane comes within 3km of the recording device (we use the Haversine Python library to estimate distances between sets of coordinates). Negative samples on the other hand are opportunistically recorded when no aircraft are detected within 10km's - this is to ensure no aircraft noise can bleed into our negative class samples. Aircraft samples are event-driven, whereas negative samples are more "randomly sampled". This opportunistic randomness can quickly snowball to a large collection of "easy" negatives, so if we can run inference and triage them on device - we are left with just a small collection of "hard" negatives that will be suitable to add to our training data.

> we cast a wide net for negative samples in order to catch the most interesting ones


## Platform
This project aimed to leverage the Edge Impulse platform as much as possible throughout the system. Unfortunately, due to the 32-bit OS on my Pi I couldn't  incorporate all the features in Edge Impulse I would've liked to (64-bit OS would've been much easier to implement the entire system).



## Hardware
This project incorporates two sets of hardware; 1) the collector, and 2) the target inference device

1) **The Collector**
- Raspberry Pi 4B (4GB)
- Nooelec RTL SDR (Software-Defined Radio)
- Arduino Nano 33 BLE Sense (flashed as a USB microphone)
- USB GPS (optional)
- Power supply (solar charged 12V battery with 5A power module)

2) **The Target**
- Arduino Nano 33 BLE Sense (flashed with Edge Impulse deployment)



## Software

1) The Collector
2) Remote Services
3) The Annotator
4) MLOps
5) DSP* (used for reverse engineering the dsp blocks with just numpy and scipy)
6) Evaluation scripts* (used to pull samples collection and on-device inference data)




## Edge Learnings

### AOI (Area of Interest)
A 4GB RPi can struggle to calculate haversine distances for every aircraft when traffic is high. This can lead to lagging in monitoring, and just consumes more compute and power. To combat this, we simply calculate a square "area of interest" once at the beginning of a recording session, and filter out aircraft operating beyond out area of interest with a lightweight float comparison. By only calculating the distances for targets within the AOI, we significantly reduced the load on the Pi.

### SSH For The Win
I know his might sound dumb, but I had never SSH'd to my Pi prior to this project. I had an initial setback of using a fixed IP as the host address, but soon figured out to use `raspeberrypi.local` instead. Aside from that, being able to remotely post to/fetch from a device is super convenient, and makes it a truly remote deployment.

### Design the Pipeline for Your Target Device (TOO OBVIOUS)




## The Project
To demonstrate AeroLoop in action, let's first define a project scope and desired outcomes.

Goal: Train and Deploy a real-time Aircraft Sound Detection model to an Arduino Nano microcontroller.

Constraints:
- Nano 256 KB RAM / 1 MB flash - affects DSP and model architecture decisions.

Considerations:
- DSP window size: Typically, aircraft sound events are long and slow - so a detection model would benefit from longer inference windows (5, 10 or even 15 seconds).
    - Window ceiling: 256KB ~ 8 seconds of raw 16 kHz 16-bit audio (does not allow for DSP or inference)
    - Window floor: Aircraft sounds tend to "roll" as they propagate and reflect in the environment. The farther away, the longer the "roll cycle". The shorter the window, the less temporal information available which can lead to jittery predictions and make it more difficult to discern between vehicle sounds etc.
    - Must be small enough to fit in memory.
    - Large enough to ensure rich features and allow processing time for DSP and inference.
    - Decision: a 2-second window size is short enough to fit in memory and still allow space for DSP and inference.

- DSP/Feature Generation: Essentially, how do we transform our raw audio data into a set of features suitable to pass to our model? The trick with feature generation is to remove as much unnecesarry information from the raw data as possible, while still maintaing a rich enough set of features that a model can actually learn from. The the Edge Impulse MFE block is ideal for this task.
    - N Mels: Determines the "frequency resolution" of our spectrogram. Went with 32 mel bands for this project.
    - N FFT: 512
    - Frame Size: 0.032
    - Hop Length: 0.032
    - High Frequency: 4,000 Hz
    - Low Frequency: 50 Hz

- Model Architecture
    - Utilised the basic Edge Impulse 2D Conv architecture
    - Added another two 2D Convolution layers to aggressively reduce dimensionality before passing to the hidden layer.
    - Used 0.6 dropout layer after flattening to introduce regularisation/reduce overfitting.
    - Softmax output with two classes - "aircraft" and "negative"

















> Working draft scaffold for the hackathon submission. Replace prompts with your content and evidence. Keep the narrative tight: AeroLoop (system/methodology) + Aircraft Detector (proof/demo).

### 1) Elevator Pitch (3–5 sentences)
- Hook: collecting data is easy; collecting *valuable* data is hard. Describe this pain.
- Solution: AeroLoop as a closed-loop DataOps/active-learning pipeline (SDR ground truth + on-device filtering).
- Result: continuous improvement over 24h; accuracy bump (94.73% → 95.3%); minimal human time.
- Devices: Raspberry Pi 4 collector + Arduino Nano 33 BLE Sense target.
- One-liner on impact/use cases.

### 2) Problem & Why It Matters
- Real-world imbalance: aircraft events are rare; most audio is background that is costly to label but adds little value.
- Site-specific noise: traffic, construction, rail, wind/rain differ by location; "clean" lab models fail in the field.
- Annotation bottleneck: 24/7 capture yields hours of negatives per day; humans cannot keep up.
- Edge constraint: target device (Arduino Nano) forces tiny DSP+model; must be robust without cloud assist.
- Why it matters: airports/noise compliance, border/surveillance for low-flying aircraft, smart-city sensing, rapid site-specific deployments.

### 3) System Architecture (Narrative + Diagram)
- Summarize the two-tier system: Pi collector/manager + Nano target.
- Hardware: Pi 4 + RTL-SDR (dump1090), optional GPS, Nano as USB mic, Pi runs TFLite for triage.
- Target constraint: Nano 256 KB RAM / 1 MB flash drives DSP window + model choices.
- Add loop diagram (SDR/Mic → Pi filter → EI train/build → Pi/Nano deploy → back to collect).

### 4) Innovation: The AeroLoop Workflow
- Ground truth via physics: ADS-B distance < 3 km → record positive aircraft clips.
- Opportunistic negatives: no aircraft within 10 km → record ambient.
- Hard-negative mining: run current model on Pi; if aircraft prob < 0.4 delete (“easy”); keep confused negatives for annotation.
- Continuous loop: download → annotate → train → evaluate → build → deploy; auto-skip deploy if no improvement.

### 5) DSP & Model Optimization (Why It Fits on Nano)
- Window: 2 s audio; MFE compresses ~32k samples → 1,984 features.
- Architecture: 4× Conv2D to shrink to ~128 features before classification head; meets Nano RAM/flash.
- On-device timing: DSP + inference < window for continuous mode (add numbers).
- Trade-offs: window size vs. feature richness vs. memory; why 2 s was the sweet spot.

### 6) Experiment Timeline (Story of the Last 48 Hours)
- Baseline: ~2h20m data, stratified 50:50 train/test to keep a “hard” test; accuracy ~94.6%.
- Day 1 loop: collect with triage; annotate; retrain; accuracy → 95.3%; dataset balance shifts to ~53% aircraft, train/test ~60:40.
- Day 2 morning loop: repeat download/annotate/train/deploy; note incremental gains and qualitative robustness.
- Negative-threshold policy: 0.4 cutoff for deletions; rationale and effect on annotation time.

### 7) Results & Proof
- Metrics: before/after accuracy, F1, confusion matrix notes; class distribution shift.
- Footprint: model size, RAM use, DSP/inference latency on Nano; TFLite performance on Pi.
- Evidence: screenshots (EI confusion matrix, data explorer), Loom/video of Nano continuous inference + Pi triage.
- Links: repo, baseline dataset (Zenodo), EI project (if shareable).

### 8) Impact & Future Work
- Impact: faster data curation, less annotation time, more robust edge behavior.
- Future: extend to altitude/regression, takeoff/landing state, aircraft type; generalize AeroLoop to other domains (e.g., logging, sirens).
- Operational plan: keep hard-negative threshold adaptive; periodic retrains; field deployment steps.

### 9) How to Reproduce (Condensed)
- Prereqs: .env with EI + Pi creds; hardware list.
- Scripts: `run_download.*`, `run_annotator.*`, `run_train_deploy.*`; mention `services/mlops/orchestrator.py` steps.
- Data steps: baseline dataset link, upload instructions, stratified split note.
- Deployment: flash Nano (continuous inference build), Pi TFLite build, SDR setup (`dump1090`), optional GPS sync.

### 10) Hackathon Fit & Differentiation
- Edge MLOps focus: on-device filtering to curate data; not just “trained a model.”
- Sensor fusion: SDR for self-labeling positives.
- Hard-negative mining: keeps annotation lean, quality high.
- Demonstrated improvement within hackathon window (94.6% → 95.3% with balanced set shift).

### 11) Assets to Attach/Embed
- Loop diagram (PNG).
- Confusion matrix + data explorer screenshots.
- Short demo video link.
- Baseline dataset link; updated dataset snapshot stats.

### TODO Checklist (delete when done)
- [ ] Fill each section with concrete numbers/dates from `submission/working_notes.md` and EI metrics.
- [ ] Add loop diagram and screenshots under `submission/images/`.
- [ ] Insert video link.
- [ ] Verify on-device timing/memory numbers.
- [ ] Keep wording concise and cohesive (AeroLoop system + Aircraft Detector demo).





