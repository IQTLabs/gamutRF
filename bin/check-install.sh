#! /usr/bin/env bash

# Script to check GamutRF install

#Check Docker
if [ -x "$(command -v docker)" ]; then
    echo "Docker install...PASS"
else
    echo "Docker install...FAIL"
    echo "ERROR: Please check your installation of docker"
fi

#Check UHD Driver
if [ -x "$(command -v uhd_find_devices)" ]; then
    echo "UHD install...PASS"
else
    echo "ERROR: UHD install...FAIL"
    echo "ERROR: Please check your installation of uhd_host"
fi

#Check for SDR
if uhd_find_devices >&/dev/null; then
    echo "Ettus SDR found...PASS"
else
    echo "ERROR: Ettus SDR found...FAIL"
    echo "ERROR: Please check your B200 is plugged in"
fi

#Check Folders
if [ -d /flash/gamutrf ]; then
    echo "Flash directory install...PASS"
else
    echo "WARNING: Flash directory install...FAIL"
    echo "WARNING: Please check your directory structure"
fi

#Check GamutRF containters
if docker images | grep 'iqtlabs/gamutRF'; then
    echo "GamutRF containers install...PASS"
else
    echo "WARNING: GamutRF containers install...FAIL"
    echo "WARNING: Please pull the GamutRF containers. Run ./bin/update.sh"
fi