import requests
import json
import logging
import os
import socket
import time
import csv
import time
from datetime import datetime

import gpsd
import httpx

def get_adafruit_gps():
    try:
        if gpsd.gpsd_stream is None:
            gpsd.connect(host="127.0.0.1", port=2947)
        packet = gpsd.get_current()
        vals= {"timestamp": time.time(),
                "position": packet.position(),
                "altitude": packet.altitude(),
                "gps_time": packet.get_time().timestamp(),
                "map_url": packet.map_url(),
                "heading": None,
                "gps": "fix"}
    except (BrokenPipeError, gpsd.NoFixError, AttributeError) as err:
        logging.error("could not update with GPS: %s", err)
        vals = {
            "timestamp": time.time(),
            "position": None,
            "altitude": None,
            "gps_time": None,
            "map_url": None,
            "heading": None,
            "gps": "no",
        }
    return vals

def get_pixhawk_gps():
    try:
        external_gps_msg = json.loads(httpx.get(f"http://127.0.0.1:8888/gps-data").text)
        
        vals = {
            "timestamp": time.time(),
            "position": (
                external_gps_msg["latitude"],
                external_gps_msg["longitude"],
            ),
            "altitude": external_gps_msg["altitude"],
            "gps_time": external_gps_msg["time_usec"],
            "map_url": None,
            "heading": None,
            "gps": "fix",
        }

    except Exception as err:
        logging.error("could not update with external GPS: %s", err)
        vals = {
            "timestamp": time.time(),
            "position": None,
            "altitude": None,
            "gps_time": None,
            "map_url": None,
            "heading": None,
            "gps": "no",
        }
    return vals

def write_to_csv(filename, data):
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(data)

# Define the file names for the CSV files
adafruit_csv_filename = 'adafruit_gps_data.csv'
pixhawk_csv_filename = 'pixhawk_gps_data.csv'

while True:
    # Get data from Adafruit GPS
    adafruit_data = get_adafruit_gps()
    if adafruit_data:
        write_to_csv(adafruit_csv_filename, adafruit_data)
        print("Adafruit GPS data written to CSV")

    # Get data from Pixhawk GPS
    pixhawk_data = get_pixhawk_gps()
    if pixhawk_data:
        write_to_csv(pixhawk_csv_filename, pixhawk_data)
        print("Pixhawk GPS data written to CSV")

    # Wait for one minute before the next iteration
    time.sleep(60)