#!/bin/bash

make files
if [ $? -ne 0 ]; then
  echo "$0> Command 'make files' failed. Aborting script." >&2
  exit 1
fi

rm -f logs/{error,output}.txt
mkdir -p logs
nohup ./scripts/tests.sh < /dev/null > logs/output.txt 2> logs/error.txt &
