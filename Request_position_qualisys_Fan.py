#!/usr/bin/python3
#!/usr/bin/python3
# coding=utf8
#  ____  ____      _    __  __  ____ ___
# |  _ \|  _ \    / \  |  \/  |/ ___/ _ \
# | | | | |_) |  / _ \ | |\/| | |  | | | |
# | |_| |  _ <  / ___ \| |  | | |__| |_| |
# |____/|_| \_\/_/   \_\_|  |_|\____\___/
#                           research group
#                             dramco.be/
#
#  KU Leuven - Technology Campus Gent,
#  Gebroeders De Smetstraat 1,
#  B-9000 Gent, Belgium
#
#      Created: 2025-1-19
#       Author: Daan Delabie (Modified by Fan Wu)
#      Version: 0.1
#
#  #  Description: Create a dataset for indoor positioning: Ground Truth (Qualisys)
# -----------------------------------------------------------------------
import xml.etree.ElementTree as ET
import csv
from datetime import datetime, timedelta
import asyncio
import qtm_rt
import numpy as np
import os

MN_name = 'Robot_Fan'
CSV_HEADER = ["timestamp", "x", "y", "z",
              "rot0", "rot1", "rot2",
              "rot3", "rot4", "rot5",
              "rot6", "rot7", "rot8"]

def check_NaN(position, rotation):
    return np.isnan(float(position[0]))

def create_body_index(xml_string):
    """Extract a name to index dictionary from 6dof settings xml."""
    xml = ET.fromstring(xml_string)
    body_to_index = {}
    for index, body in enumerate(xml.findall("*/Body/Name")):
        body_to_index[body.text.strip()] = index
    return body_to_index

async def main(wanted_body, measuring_time, csv_file_path):
    # Connect to qtm
    connection = await qtm_rt.connect("192.108.0.13")
    if connection is None:
        print("Qualisys: Failed to connect")
        return

    # Take control of qtm; automatically released after scope end.
    async with qtm_rt.TakeControl(connection, "Techtile229"):
        await connection.new()

    # Get 6dof settings from qtm
    xml_string = await connection.get_parameters(parameters=["6d"])
    body_index = create_body_index(xml_string)
    
    # Open CSV file for appending rows.
    csv_file = open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADER)  # write header

    record_count = 0

    def on_packet(packet):
        nonlocal record_count
        info, bodies = packet.get_6d()
        now = datetime.now()
        # Check if wanted body exists
        if wanted_body is not None and wanted_body in body_index:
            wanted_index = body_index[wanted_body]
            position, rotation = bodies[wanted_index]
            if not check_NaN(position, rotation):
                time_str = now.strftime("%H:%M:%S.%f")
                # Convert positions from mm to m
                x = position[0] / 1000
                y = position[1] / 1000
                z = position[2] / 1000
                # Flatten rotation matrix (assuming rotation[0] is a 3x3 matrix)
                rot_array = np.array(rotation[0]).flatten().tolist()
                # Create row: timestamp, x, y, z, rot0, ..., rot8
                row = [time_str, x, y, z] + rot_array
                csv_writer.writerow(row)
                record_count += 1
            else:
                print("Qualisys: No object detected")
        else:
            print("Qualisys: NO BODY FOUND")

    # Start streaming frames
    await connection.stream_frames(components=["6d"], on_packet=on_packet)
    
    # Wait asynchronously for the given measuring time
    await asyncio.sleep(measuring_time)
    
    # Stop streaming and close CSV file
    await connection.stream_frames_stop()
    csv_file.close()

def get_Qualisys_Position(wanted_body, measuring_time, csv_file_path):
    asyncio.get_event_loop().run_until_complete(main(wanted_body, measuring_time, csv_file_path))

if __name__ == '__main__':
    save_dir = "./Dataset_GT"
    os.makedirs(save_dir, exist_ok=True)
    measuring_time = 600  # seconds
    data_filename = datetime.now().strftime("%m-%d_%H-%M") + "_GT.csv"
    full_path = os.path.join(save_dir, data_filename)
    get_Qualisys_Position(MN_name, measuring_time, full_path)
