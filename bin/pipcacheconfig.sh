#!/bin/sh

if [ "$POETRY_CACHE" == "" ] ; then
	echo not using pip cache
	exit 0
fi

trusted_host=$(echo $POETRY_CACHE|grep -Po '://\K[^/]+')
cat > /etc/pip.conf <<- EOC
[global]
index-url = $POETRY_CACHE
trusted-host = $trusted_host
EOC

cat /etc/pip.conf
