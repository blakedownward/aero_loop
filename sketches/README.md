## Sketches flashed to the Arduino Nano

## Requirements (common)
- Arduino Nano 33 BLE Sense (Rev 2 used in this project)
- Arduino IDE with **Arduino Mbed OS nano Boards** (v4.4.1) installed.
- PDM.h library (included with Arduino IDE)


### Nano Mic
To flash the nano board as a USB microphone, first ensure you have the above requirements installed/available... then:
- Open the "NanoMic.ino" file in the Arduino IDE
- Plug the board into the PC via USB port and ensure the IDE recognises and is selected as the target.
- Finally, click "Upload" to build the package and flash it to the Nano
- The Nano is now effectively a USB microphone.


### Nano Inference
The inference sketch requires an Edge Impulse deployment/build as an Arduino Library.
- Select "Arduino Library" as the build type, and "Arduino Nano" as the target device when building in Edge Impulse.
- With the inference sketch open in Arduino IDE, click "sketch" from the menu, then "Include library", then select "Add .ZIP library".
- Find and select the .ZIP file downloaded from Edge Impulse to make it available to your sketches.
- With the "aero_loop_nano_ble33_sense_microphone_2sec.ino" file open in the IDE - find the "Includes" section, and update `#include <AeroLoop_inferencing.h>` to use the name of library uploaded in the previous step. Eg `#include <[your_library_name].h>`
- Click "Upload" to build the package and flash it to the Nano.
- On completion, verify everything works correctly by opening the "Serial Monitor" in the IDE - you should now see a stream of predictions from the Nano.