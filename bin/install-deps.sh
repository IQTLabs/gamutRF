#! /usr/bin/env bash

# Script to quickly install GamutRF and dependencies

if [ -x "$(command -v docker)" ]; then
    echo "Docker already installed..."
else
    echo "Installing Docker..."
    sudo apt-get update
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

echo "Installing GamutRF dependencies..."
sudo apt install -y git libjpeg-dev python3 python3-pip python3-tk uhd-host gpsd gpsd-clients chrony pps-tools onboard at-spi2-core tmux

echo "Setting up GamutRF..."
sudo uhd_images_downloader -t "b2|usb"
git clone https://github.com/IQTLabs/gamutRF
git clone https://github.com/IQTLabs/BirdsEye
sudo su -
mkdir -p /flash/gamutrf
