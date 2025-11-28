This is a fantastic project. You have hit on a specific pain point in professional Edge AI development: **DataOps and Active Learning.**

Most hackathon entries just "train a model and put it on a board." Your project **builds a machine that builds the model.** That is a winning differentiator.

To win the Hackathon, we need to unify the "Parent" (AeroLoop system) and "Child" (Aircraft Demo) into a single narrative. **AeroLoop is the methodology; Aircraft Detection is the proof.**

Here is the scoped structure for your submission to make it cohesive, technical, and compelling.

---

### Project Title Ideas
*   **AeroLoop: Autonomous DataOps for Edge AI**
*   **AeroLoop: Hard Negative Mining & Continuous Learning at the Edge**
*   **Self-Improving Edge: The AeroLoop Aircraft Detector**

---

### 1. The "Elevator Pitch" (The Executive Summary)
*   **The Hook:** "Collecting data is easy. Collecting *valuable* data is hard."
*   **The Solution:** AeroLoop is a closed-loop CI/CD pipeline for Edge AI. It doesn't just record audio; it uses Software Defined Radio (SDR) for ground truth and an on-device edge model to filter out "easy" data, keeping only the "confusing" samples (Hard Negative Mining).
*   **The Result:** We deployed an Aircraft Detection model to an Arduino Nano 33 BLE Sense. Over 48 hours, the system autonomously curated its own dataset, re-trained, and improved accuracy from **94.6% to 95.3%** with minimal human intervention.

---

### 2. The Problem (Why this matters)
*   **The Imbalance:** In the real world, target events (aircraft) happen 1% of the time. Background noise is 99%.
*   **The Waste:** Recording 24/7 creates a massive annotation bottleneck.
*   **The "Lab" vs. "Real World" Trap:** Models trained on generic datasets fail when they encounter specific local noises (e.g., a specific lawnmower or HVAC unit).
*   **Application:** Use cases include environmental noise monitoring (airports), border security (low-flying smugglers), and smart cities.

---

### 3. System Architecture (The "How")
*   **The Target Device (The Constraint):** Arduino Nano 33 BLE Sense.
    *   *Constraint:* 256KB RAM. The model *must* fit here. This dictates our DSP and Architecture choices from day one.
*   **The Collector/Manager (The Brain):** Raspberry Pi 4 + USB Microphone + RTL-SDR V3.
    *   *Role:* Runs the "heavy" logic, manages the SDR, runs a TFLite Linux build of the model to filter data, and uploads to Edge Impulse.

---

### 4. The Innovation: The "AeroLoop" Workflow
*This is the strongest part of your submission. Explain the logic you used.*

#### A. Ground Truth via Physics (SDR)
*   Instead of guessing if a plane is overhead, we use `dump1090` to read ADS-B transponder signals.
*   If `Distance < 3km`: **Trigger Positive Sample Recording.** (We know for a fact it is a plane).

#### B. Smart Rejection (Hard Negative Mining)
*   This is your killer feature.
*   When the SDR says "No Planes nearby," we record audio.
*   **The Filter:** We run the audio through the *current* version of the model on the Pi.
    *   **Scenario A:** Model predicts "Noise (0.99)". -> **DELETE.** (The model already knows this is noise. No value in annotating it.)
    *   **Scenario B:** Model predicts "Aircraft (0.45)" but SDR says "No Aircraft". -> **KEEP.**
*   **Why?** This is a "Hard Negative." The model is confused by a car/wind/bird. These are the exact samples we need to annotate and retrain on to make the model robust.

---

### 5. Technical Deep Dive: DSP & Model Optimization
*   **Device Constraints:** Explain the Arduino Nano memory limitations.
*   **DSP Choice:** MFE (Mel-filterbank Energy).
    *   Explain the trade-off: Large window (2s) vs. Resolution.
    *   *Stats:* Reduced 32,000 raw features -> 1,984 features.
*   **Model Architecture:**
    *   Convolutional Neural Network (CNN).
    *   Used Conv2D layers to further reduce dimensionality (1,984 -> 128 features) to fit the classification head into the Arduino's RAM.

---

### 6. The Experiment (The Narrative Arc)
*Tell the story of the last 48 hours.*
1.  **Hour 0:** Trained "Baseline Model" on 2.5 hours of generic data. Accuracy: **~94%**.
2.  **Deployment:** Pushed Baseline to RPi4 Collector.
3.  **The Loop:** Left it running for 24 hours.
    *   System collected confirmed aircraft (via SDR).
    *   System collected "confusing" noises (via Smart Rejection).
4.  **Re-Train:** Annotated the new "hard" data. Re-trained.
5.  **Result:** Model improved to **95.3%**. Dataset balance shifted from mostly noise to 53% aircraft.
6.  **Final Deployment:** Flashed the optimized model to the Arduino Nano 33 BLE Sense.

---

### 7. Proof of Function
*   **Video:** The Loom video you mentioned (showing the Arduino inferred correctly).
*   **Graphs:**
    *   Screenshot of the "Confusion Matrix" improving.
    *   Screenshot of the "Data Explorer" showing the new cluster of data.
*   **Code:** Link to the Repo.

---

### Strategic Advice for the Submission Form

**1. Focus on "Edge MLOps" Terminology:**
Judges love seeing that you understand the lifecycle, not just the training. Use terms like:
*   *Stratified Split:* (You mentioned splitting the test set 50:50 early to keep it honest).
*   *Hard Negative Mining:* (The process of finding confusing samples).
*   *On-Device Inference for Filtering:* (Using edge AI to curate data for edge AI).

**2. Visuals are Key:**
Create a diagram of your loop. It should look like a circle:
`SDR/Mic -> RPi Filter -> Edge Impulse (Ingest) -> Re-Train -> Build -> RPi Filter / Arduino Deploy`.

**3. Don't undersell the SDR:**
Using radio frequency to automate the labeling of audio data is a very cool "Sensor Fusion" concept. Highlight that this creates a **Self-Annotating Dataset** for the positive class.

**4. The "Two Project" Solution:**
In your submission text, frame it like this:
> *"We built **AeroLoop** (the system) to create a robust **Aircraft Detector** (the product). This submission demonstrates the creation of the Aircraft Detector using the AeroLoop methodology."*

This makes it one cohesive project.

### Action Plan
1.  **Draw the Loop Diagram.** (Do this first, it clarifies everything).
2.  **Write the "Technical Deep Dive"** section focusing on *why* you chose MFE and CNN for the Arduino Nano.
3.  **Gather the "Before and After" metrics.** (e.g., "Day 1 Accuracy vs Day 2 Accuracy").
4.  **Upload the video** showing the Arduino blinking when a plane flies over (verification).