#!bin/bash
set -e
set -x

curl -V > /dev/null || echo "First install curl then try again."

sudo curl https://raw.githubusercontent.com/IQTLabs/gamutRF/main/bin/gamutrf --output /usr/local/bin/gamutrf
sudo chmod +x /usr/local/bin/gamutrf
echo -e "\n\n\n======\n\nFinished installing.\nRun 'gamutrf' to control various GamutRF tools and services.\n\n======\n"
