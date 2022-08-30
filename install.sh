#!bin/bash
set -e
set -x

docker-compose -v > /dev/null || echo "First install docker-compose then try again."
curl -V > /dev/null || echo "First install curl then try again."

sudo curl https://raw.githubusercontent.com/IQTLabs/gamutRF/main/bin/gamutrf --output /usr/local/bin/gamutrf
sudo chmod +x /usr/local/bin/gamutrf
echo -e "\n\n\n======\n\nFinished installing.\nRun 'gamutrf' to controll various GamutRF tools and services.\n\n======\n"
