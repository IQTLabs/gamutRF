#!/bin/bash

PORT=$1
LASTUPDATE=""
for i in {1..5} ; do
    UPDATE=$(wget -q -O- http://0.0.0.0:$PORT/|grep -E '^run_timestamp')
    if [[ "$UPDATE" == "" ]] ; then
        continue
    fi
    if [[ "$LASTUPDATE" == "" ]] ; then
        LASTUPDATE=$UPDATE
        continue
    fi
    if [[ "$UPDATE" != "$LASTUPDATE" ]] ; then
        exit 0
    fi
    sleep 1
done
echo tries: $i
exit 1
