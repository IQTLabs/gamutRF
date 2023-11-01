from flask import Flask, jsonify
from pymavlink import mavutil
import threading
import time
import serial

app = Flask(__name__)
class MAVLINKGPSHandler:
    def __init__(self):

        self.STALE_TIMEOUT = 30 #seconds

        self.gps_stale = None
        self.gps_fix_type = None
        self.latitude = None
        self.longitude = None
        self.altitude = None
        self.relative_alt = None
        self.time_usec = None
        self.vx = None
        self.vy = None
        self.vz = None
        self.time_boot_ms = None

        self.latest_GLOBAL_POSITION_INT_msg = None
        self.latest_GLOBAL_POSITION_INT_timestamp = None
        self.latest_GPS_RAW_INT_msg = None
        self.latest_GPS_RAW_INT_timestamp = None
        
        self.mavlink_thread = threading.Thread(target=self.handle_mavlink_messages)
        self.mavlink_thread.daemon = True
        self.mavlink_thread.start()

    def GLOBAL_POSITION_INT_parser(self, data: dict = None) -> dict:
        """Selected required data from data received from Pixhawk, and
        convert to required units.
        Args:
            data (dict): Data received from the Pixhawk device
        Returns:
            clean_data (dict): Required data to be published
        """
        #If data supplied, else use last msg
        if data == None:
            data = self.latest_GLOBAL_POSITION_INT_msg.to_dict()

        #Check for stale GPS
        self.gps_stale_check()

        #Parse msg data
        self.time_boot_ms = data["time_boot_ms"] #mm
        self.latitude = data["lat"]/10000000  # decimal degrees
        self.longitude = data["lon"]/10000000  # decimal degrees
        self.altitude = data["alt"] #mm
        self.relative_alt = data["relative_alt"] #mm
        self.heading = data["hdg"]/100.0  # decimal degrees
        self.vx = data["vx"]/100.0  # meters/second
        self.vy = data["vy"]/100.0  # meters/second
        self.vz = data["vz"]/100.0  # meters/second
        return 

    def GPS_RAW_INT_parser(self, data: dict = None) -> dict:
        """Selected required data from data received from Pixhawk, and
        convert to required units.
        Args:
            data (dict): Data received from the Pixhawk device
        Returns:
            clean_data (dict): Required data to be published
    
        GPS_FIX_TYPE
        [Enum] Type of GPS fix

        Value	Field Name	             Description
        0	    GPS_FIX_TYPE_NO_GPS	    No GPS connected
        1	    GPS_FIX_TYPE_NO_FIX	    No position information, GPS is connected
        2	    GPS_FIX_TYPE_2D_FIX	    2D position
        3	    GPS_FIX_TYPE_3D_FIX	    3D position
        4	    GPS_FIX_TYPE_DGPS	    DGPS/SBAS aided 3D position
        5	    GPS_FIX_TYPE_RTK_FLOAT	RTK float, 3D position
        6	    GPS_FIX_TYPE_RTK_FIXED	RTK Fixed, 3D position
        7	    GPS_FIX_TYPE_STATIC	    Static fixed, typically used for base stations
        8	    GPS_FIX_TYPE_PPP	    PPP, 3D position.
        """
        #If data supplied, else use last msg
        if data == None:
            data = self.latest_GPS_RAW_INT_msg.to_dict()

        #Check for stale GPS
        self.gps_stale_check()

        #Update fix type
        self.time_usec = data["time_usec"] #UNIX Epoch time uSec
        self.gps_fix_type=data["fix_type"]

        return
    
    def gps_stale_check(self):
        #Check for stale GPS data
        if (time.time() - self.latest_GPS_RAW_INT_timestamp > self.STALE_TIMEOUT) or \
            (time.time() - self.latest_GLOBAL_POSITION_INT_timestamp > self.STALE_TIMEOUT):
            self.gps_stale = True
        else:
            self.gps_stale = False

    
    def create_gps_json_payload(self):
        #Check for stale GPS
        self.gps_stale_check()

        #Create payload dict for json
        payload={}

        payload["gps_stale"] = self.gps_stale
        payload["gps_fix_type"] = self.gps_fix_type
        payload["time_boot_ms"] = self.time_boot_ms #mm
        payload["time_usec"] = self.time_usec
        payload["latitude"] = self.latitude  # decimal degrees
        payload["longitude"] = self.longitude  # decimal degrees
        payload["altitude"] = self.altitude #mm
        payload["relative_alt"] = self.relative_alt #mm
        payload["heading"] = self.heading  # decimal degrees
        payload["vx"] = self.vx  # meters/second
        payload["vy"] = self.vy  # meters/second
        payload["vz"] = self.vz  # meters/second

        return payload

    # Function to handle incoming MAVLink messages
    def handle_mavlink_messages(self):
        # Connect to the MAVLink source (e.g., UDP or serial port)
        mavlink_connection = mavutil.mavlink_connection("/dev/tty.serial1", 57600)

        while True:
            #msg = mavlink_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
            msg = mavlink_connection.recv_match(blocking=True)
            if msg.get_type() == 'GLOBAL_POSITION_INT':
                self.latest_GLOBAL_POSITION_INT_msg = msg
                self.latest_GLOBAL_POSITION_INT_timestamp=time.time()
            elif msg.get_type() == 'GPS_RAW_INT':
                self.latest_GPS_RAW_INT_msg = msg
                self.latest_GPS_RAW_INT_timestamp=time.time()
            msg

# Store the latest GPS data handler
mavlink_gps_handler = MAVLINKGPSHandler()

# Routes for latest GPS Data
@app.route('/gps-fix-status', methods=['GET'])
def get_latest_gps_fix_status():
    if mavlink_gps_handler.latest_GPS_RAW_INT_msg:
        mavlink_gps_handler.GPS_RAW_INT_parser()
        return jsonify({"fix_type":mavlink_gps_handler.gps_fix_type,"gps_stale":mavlink_gps_handler.gps_stale}), 200
    else:
        return jsonify({'error': 'No GPS data available'}), 404
    
@app.route('/gps-data', methods=['GET'])
def get_latest_gps_data():
    if mavlink_gps_handler.latest_GLOBAL_POSITION_INT_msg:
        mavlink_gps_handler.GLOBAL_POSITION_INT_parser()
        msg=mavlink_gps_handler.create_gps_json_payload()
        return jsonify(msg), 200
    else:
        return jsonify({'error': 'No GPS data available'}), 404
    
@app.route('/heading', methods=['GET'])
def get_latest_heading():
    if mavlink_gps_handler.latest_GLOBAL_POSITION_INT_msg:
        mavlink_gps_handler.GLOBAL_POSITION_INT_parser()
        return jsonify({"heading":mavlink_gps_handler.heading,"gps_stale":mavlink_gps_handler.gps_stale}), 200
    else:
        return jsonify({'error': 'No heading data available'}), 404

def main():
    app.run(host='0.0.0.0', port=8888)

if __name__ == '__main__':
    main()
