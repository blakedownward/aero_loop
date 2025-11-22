#!/usr/bin/python

import os
import sys

# print(f'CWD: {os.getcwd()}')
# print(f'path0: {sys.path[0]}')

import os
import socket
import time
import datetime as dt
import sounddevice
import numpy as np


time.sleep(2)

import session_constants as c
from logger import update_craft_log, init_session_log

# from record import record_ac, record_silence
from areaofinterest import in_aoi, fetch_dist

# set the flag for running inference on collected samples
RUN_INFERENCE = True

if RUN_INFERENCE:
    import inference
    from logger import update_inference_log
    
    
# print session configuration
print("Mic Mode:", c.MIC_MODE)
print("Microphone ID:", c.MIC_ID)
print("Location ID:", c.LOC_ID)

if c.MIC_MODE == "nano":
    import nano_record as rec
else:
    import record as rec

# print(sys.modules['inference'])

init_session_log()

def fetch_stream(buffer_size: int = c.SOCKET_BUFF_SIZE, refresh_rate: float = 4.0, debug: bool = False):
    """Fetches stream of ADS-B data and decodes it"""

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 30003))
    # print('attched to socket...')
    time.sleep(refresh_rate)
    data = client_socket.recv(buffer_size)
    data = data.decode('UTF-8')
    # print('detaching...')
    client_socket.detach()

    if debug:
        print(f'Data length: {len(data)}\nBuffer size: {buffer_size}')

    return data.split(sep='\n')


def discrete_session():
    # init state of helper variables
    count = 1
    aircraft_count = 0
    silence_count = 0
    consecutive_silence = 0
    socket_errors = 0
    session_craft = []
    last_craft = ''
    update_craft_log('*')
    # session loop condition
    while silence_count <= c.MAX_SILENCE:
        print('Iteration: ', count)
        # start_time = time.time()
        # stream ads-b data
        try:
            data = fetch_stream(debug=False)
            socket_errors = 0
        except socket.gaierror:
            socket_errors += 1
            if socket_errors < 2:
                print('Socket error... Sleeping for 5 minutes')
                time.sleep(300)
                continue
            else:
                print('Consecutive socket errors... Rebooting.\n')
                # second print statement to ensure first one is flushed to stdout before shutdown
                print(' ')
                os.system('sudo reboot')
                time.sleep(15)
            
        # fetch_time = time.time() - start_time
        # start_iter = time.time()
        # read the data,
        coord_list = []
        for i in data:
            try:
                t = i.split(',')
                # check for message type "3" (positional info)
                if t[1] == '3':
                    coord_list.append(t)

                else:
                    pass

            except IndexError:
                pass
            finally:
                pass

        # avoid counting silence in the first iteration
        if count < 5:
            print('initial sleeping')
            time.sleep(1)
            count += 1

            continue

        # log first of each craft, keep only those in the area of interest (aoi_craft list)
        clean_flights = []
        skip_craft = []

        for msg in coord_list:
            # grab the hex code for each message and append to session
            session_craft.append(msg[4])

            # skip if in 'skip_craft'
            if msg[4] in skip_craft:
                continue

            try:
                # get coordinates of the target > check if the craft is within the AOI
                target_pos = (float(msg[14]), float(msg[15]))
                # print('Target pos', target_pos)

                if in_aoi(lat_bounds=c.AOI[0], lon_bounds=c.AOI[1], target_lat_lon=target_pos):
                    dist = fetch_dist(point_a=target_pos, point_b=c.DEVICE_COORDS)
                    flight = (msg[4], msg[6], msg[7], msg[9], msg[11], target_pos, dist)
                    clean_flights.append(flight)

                else:
                    skip_craft.append(msg[4])
            finally:
                pass

        for flight in clean_flights:
            print(flight)
            
        # if nothing inside the aoi > increase the consecutive silence count
        if len(clean_flights) == 0:
            consecutive_silence += 1
            # if enough consecutive silence > record silence
            if (consecutive_silence >= c.SILENCE_BUFFER) & (silence_count <= aircraft_count*3):
                consecutive_silence = 0
                # record silence
                cur_time = str(dt.datetime.now())[11:16].replace(':', '-')
                filename = f'000000_{c.TODAY}_{cur_time}.wav'
                file_path = os.path.join(c.SESSION_PATH, filename)
                
                try:
                    rec.record_silence(filepath=file_path)
                    
                    time.sleep(1)
                        
                    
                except sounddevice.PortAudioError:
                    print('PortAudioError, skipping...')
                    pass
                    
                except:
                    print('Exception raised for silence recording')
                    
                if RUN_INFERENCE:
                        
                        # check the max prediction for this file
                        preds = inference.predict_file(wav_path=file_path, model_path=c.MODEL_PATH)
                        
                        
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
                    
            else:
                pass
            

        # check those inside the aoi, zero the consecutive silence count
        else:
            # For those inside the area of recording > trigger a recording > focus only on the target and
            # append its dynamic variables (target or mark)
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
                        
                        # for troubleshooting
                        print('Message:', flight)
                        
                        try:
                            rec.record_ac(filepath=file_path)
                            aircraft_count += 1
                            
                            if RUN_INFERENCE:
                                preds = inference.predict_file(wav_path=file_path, model_path=c.MODEL_PATH)
                                status = "saved"
                                update_inference_log(filename=filename, file_status=status, predictions=preds, model_version=c.MODEL_VERSION, session_path=c.SESSION_PATH)
                            
                            
                        except sounddevice.PortAudioError:
                            print('PortAudioError, skipping...')
                            pass
                            
                        except:
                            print('Exception raised for aircraft recording')
                            

        
                
        # housekeeping    
        update_craft_log(session_craft)

        count += 1
        if count > 500:
            open(os.path.join(c.SESSION_PATH, '.processed'), 'a').close()
            print('Session end... Rebooting.\n')
            # second print statement to ensure first one is flushed to stdout before shutdown
            print(' ')
            os.system('sudo rm /home/protopi/monitor_debug.log')
            os.system('sudo reboot')
            time.sleep(15)
            


if __name__ == '__main__':
    discrete_session()
