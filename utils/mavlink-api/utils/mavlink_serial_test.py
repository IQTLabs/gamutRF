#! /usr/bin/env python3

import sys
from pymavlink import mavutil

if len(sys.argv) < 2:
    print("Usage: python mavlink_serial_test.py <serial_port>")
    sys.exit(1)

serial_port = sys.argv[1]

try:
    mav = mavutil.mavlink_connection(serial_port)
    print("Connected to MAVLink at", serial_port)
    
    # You can perform further operations here
    
    # For example, you can continuously read messages
    while True:
        message = mav.recv_match(blocking=True)
        if message is not None:
            print("Received:", message)
            
        # You can add more logic here based on the received messages
    
except Exception as e:
    print("Error:", str(e))
