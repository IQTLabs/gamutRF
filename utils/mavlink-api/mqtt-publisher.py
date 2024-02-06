import os
import time
import logging
import json
import httpx
from paho.mqtt import client as mqtt_client

MQTT_BROKER = os.environ.get("MQTT_IP", "localhost")
MQTT_PORT = os.environ.get("MQTT_PORT", "1883")
MQTT_TOPIC = os.environ.get("MQTT_TOPIC", "/targets")
QUERY_INTERVAL = int(os.environ.get("QUERY_INTERVAL", 1))
URL_LIST = json.loads(
    os.environ.get("URL_LIST", '[["default_target", "http://127.0.0.1:8888/gps-data"]]')
)

mqtt_connect_flag = False


def on_connect(client, userdata, flags, rc):
    global mqtt_connect_flag
    if rc == 0:
        print("Connected to MQTT Broker")
        mqtt_connect_flag = True
    else:
        print(f"Failed to connect, return code {rc}")


def on_disconnect(client, userdata, flags, rc):
    global mqtt_connect_flag
    mqtt_connect_flag = False


def publish_data(client, target_name, data):
    topic = f"{MQTT_TOPIC}"
    try:
        rc = client.publish(topic, data)
        print(f"Published data to {topic}: {data}")

    except ConnectionError as err:
        print("could not publish data: %s", err)

    else:
        if rc != 0:
            print(f"could not publish data RC={rc}")


def fetch_and_publish(client, target_name, target_url):
    try:
        response = json.loads(httpx.get(f"{target_url}").text)

        if int(response["gps_fix_type"]) > 0:
            data = {
                "target_name": target_name,
                "gps_stale": response["gps_stale"],
                "gps_fix_type": response["gps_fix_type"],
                "time_boot_ms": response["time_boot_ms"],  # mm
                "time_usec": response["time_usec"],
                "latitude": response["latitude"],  # decimal degrees
                "longitude": response["longitude"],  # decimal degrees
                "altitude": response["altitude"],  # mm
                "relative_alt": response["relative_alt"],  # mm
                "heading": response["heading"],  # decimal degrees
                "vx": response["vx"],  # meters/second
                "vy": response["vy"],  # meters/second
                "vz": response["vz"],  # meters/second
            }
            publish_data(client, target_name, json.dumps(data))

    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as err:
        print(f"Could not update {target_name} with error:{err}")
        data = {
            "target_name": target_name,
            "gps_stale": None,
            "gps_fix_type": None,
            "time_boot_ms": None,  # mm
            "time_usec": None,
            "latitude": None,  # decimal degrees
            "longitude": None,  # decimal degrees
            "altitude": None,  # mm
            "relative_alt": None,  # mm
            "heading": None,  # decimal degrees
            "vx": None,  # meters/second
            "vy": None,  # meters/second
            "vz": None,  # meters/second
        }


def main():
    global mqtt_connect_flag
    while True:
        try:
            print(f"Attempting to connect to MQTT {MQTT_BROKER}:{MQTT_PORT}")
            client = mqtt_client.Client()
            client.on_connect = on_connect
            client.on_disconnect = on_disconnect
            client.connect(MQTT_BROKER, int(MQTT_PORT))
        # except (ConnectionRefusedError, ConnectionError) as err:
        except Exception as err:
            print(
                f"Could not connect to MQTT broker ({MQTT_BROKER}:{MQTT_PORT}): {err}"
            )
        time.sleep(5)

        while True:  # Fix this to mqtt_connect_flag or client.is_connected()
            print(f"Initializing with {URL_LIST}")
            for target in URL_LIST:
                if len(target) == 2:
                    print(f"Attempting to retrieve data from {target}")
                    target_name, target_url = target
                    fetch_and_publish(client, target_name, target_url)
                else:
                    print(
                        "Invalid entry in URL_LIST. Each entry should be a 2-entry list."
                    )

            time.sleep(QUERY_INTERVAL)


if __name__ == "__main__":
    main()
