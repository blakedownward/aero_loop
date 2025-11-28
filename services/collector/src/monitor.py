#!/usr/bin/env python3
"""
Main monitoring loop for ADS-B aircraft detection and audio recording.

Connects to Dump1090 on port 30003, monitors aircraft in area of interest,
and triggers audio recordings when aircraft are detected.
"""

import os
import sys
import socket
import time
import datetime as dt
import sounddevice
import numpy as np

# Add config directory to path for imports
COLLECTOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(COLLECTOR_DIR, 'config')
sys.path.insert(0, CONFIG_DIR)

# Import session_constants from config/ directory
# Note: You must copy /config/session_constants.py.example to /config/session_constants.py first
import session_constants as c
from logger import init_session_log
from areaofinterest import in_aoi, fetch_dist

# Set the flag for running inference on collected samples
RUN_INFERENCE = True

if RUN_INFERENCE:
    import inference
    from logger import update_inference_log

# Print session configuration
print("Mic Mode:", c.MIC_MODE)

    
import nano_record as rec

init_session_log()


def fetch_stream(buffer_size: int = None, refresh_rate: float = None, debug: bool = False):
    """Fetches stream of ADS-B data and decodes it"""
    if buffer_size is None:
        buffer_size = c.SOCKET_BUFF_SIZE
    if refresh_rate is None:
        refresh_rate = c.REFRESH_RATE
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 30003))
    time.sleep(refresh_rate)
    data = client_socket.recv(buffer_size)
    data = data.decode('UTF-8')
    client_socket.detach()

    if debug:
        print(f'Data length: {len(data)}\nBuffer size: {buffer_size}')

    return data.split(sep='\n')


def discrete_session():
    """Main session loop for monitoring and recording"""
    # Init state of helper variables
    count = 1
    aircraft_count = 0
    silence_count = 0
    consecutive_silence = 0
    socket_errors = 0
    session_craft = []
    last_craft = ''
    # update_craft_log('*')
    
    # Session loop condition
    while silence_count <= c.MAX_SILENCE:
        print('Iteration: ', count)
        
        # Stream ADS-B data
        try:
            data = fetch_stream(buffer_size=c.SOCKET_BUFF_SIZE, refresh_rate=c.REFRESH_RATE, debug=False)
            socket_errors = 0
        except socket.gaierror:
            socket_errors += 1
            if socket_errors < 2:
                print('Socket error... Sleeping for 5 minutes')
                time.sleep(300)
                continue
            else:
                print('Consecutive socket errors... Rebooting.\n')
                print(' ')
                os.system('sudo reboot')
                time.sleep(15)
        
        # Read the data
        coord_list = []
        for i in data:
            try:
                t = i.split(',')
                # Check for message type "3" (positional info)
                if t[1] == '3':
                    coord_list.append(t)
            except IndexError:
                pass

        # Avoid counting silence in the first iteration
        if count < 5:
            print('initial sleeping')
            time.sleep(1)
            count += 1
            continue

        # Log first of each craft, keep only those in the area of interest
        clean_flights = []
        skip_craft = []

        for msg in coord_list:
            # Grab the hex code for each message and append to session
            session_craft.append(msg[4])

            # Skip if in 'skip_craft'
            if msg[4] in skip_craft:
                continue

            try:
                # Get coordinates of the target > check if the craft is within the AOI
                target_pos = (float(msg[14]), float(msg[15]))

                if in_aoi(lat_bounds=c.AOI[0], lon_bounds=c.AOI[1], target_lat_lon=target_pos):
                    dist = fetch_dist(point_a=target_pos, point_b=c.DEVICE_COORDS)
                    flight = (msg[4], msg[6], msg[7], msg[9], msg[11], target_pos, dist)
                    clean_flights.append(flight)
                else:
                    skip_craft.append(msg[4])
            except (ValueError, IndexError):
                pass

        for flight in clean_flights:
            print(flight)
            
        # If nothing inside the AOI > increase the consecutive silence count
        if len(clean_flights) == 0:
            consecutive_silence += 1
            # If enough consecutive silence > record silence
            if (consecutive_silence >= c.SILENCE_BUFFER) & (silence_count <= aircraft_count*3):
                consecutive_silence = 0
                # Record silence
                cur_time = str(dt.datetime.now())[11:16].replace(':', '-')
                filename = f'000000_{c.TODAY}_{cur_time}.wav'
                file_path = os.path.join(c.SESSION_PATH, filename)
                
                try:
                    rec.record_silence(filepath=file_path)
                    time.sleep(1)
                except sounddevice.PortAudioError:
                    print('PortAudioError, skipping...')
                    pass
                except Exception as e:
                    print(f'Exception raised for silence recording: {e}')
                    
                if RUN_INFERENCE:
                    # Check the max prediction for this file
                    preds = inference.predict_file(wav_path=file_path, model_path=c.MODEL_PATH, debug=True)
                    
                    if np.max(preds) >= 0.4:
                        status = "saved"
                        update_inference_log(filename=filename, file_status=status, predictions=preds, model_version=c.MODEL_VERSION, session_path=c.SESSION_PATH)
                        silence_count += 1
                        print(f'{filename} saved.')
                    else:
                        print(f'Max pred for {file_path} below threshold -> deleting...')
                        os.system(f'sudo rm {file_path}')
                        status = "deleted"
                        update_inference_log(filename=filename, file_status=status, predictions=preds, model_version=c.MODEL_VERSION, session_path=c.SESSION_PATH)

        # Check those inside the AOI, zero the consecutive silence count
        else:
            # For those inside the area of recording > trigger a recording
            consecutive_silence = 0
            for flight in clean_flights:
                if flight[-1] < c.TRIGGER_DIST:
                    if flight[0] == last_craft:
                        pass
                    else:
                        aircraft_id = str(flight[0])
                        last_craft = aircraft_id
                        date = flight[1]
                        date = date.replace('/', '-')
                        msg_time = flight[2]
                        msg_time = msg_time.replace(':', '-')
                        msg_time = msg_time.replace('.', '-')
                        msg_time = msg_time[:8]
                        altitude = str(flight[4])
                        filename = str(aircraft_id + '_' + date + '_' + msg_time + '_' + altitude + '.wav')
                        print('filename:', filename)
                        file_path = os.path.join(c.SESSION_PATH, filename)
                        
                        # For troubleshooting
                        print('Message:', flight)
                        
                        try:
                            rec.record_ac(filepath=file_path)
                            aircraft_count += 1
                            
                            if RUN_INFERENCE:
                                preds = inference.predict_file(wav_path=file_path, model_path=c.MODEL_PATH, debug=True)
                                status = "saved"
                                update_inference_log(filename=filename, file_status=status, predictions=preds, model_version=c.MODEL_VERSION, session_path=c.SESSION_PATH)
                        except sounddevice.PortAudioError:
                            print('PortAudioError, skipping...')
                            pass
                        except Exception as e:
                            print(f'Exception raised for aircraft recording: {e}')

        # Housekeeping
        # update_craft_log(session_craft)

        count += 1
        if count > 500:
            open(os.path.join(c.SESSION_PATH, '.processed'), 'a').close()
            print('Session end... Rebooting.\n')
            print(' ')
            # Get username from environment or use default
            username = os.getenv('PI_USERNAME', 'pi')
            os.system(f'sudo rm /home/{username}/monitor_debug.log')
            os.system('sudo reboot')
            time.sleep(15)


if __name__ == '__main__':
    discrete_session()

