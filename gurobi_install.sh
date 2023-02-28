#!/bin/bash

# You can get a free key from https://www.gurobi.com/downloads/free-academic-license/
# It looks like: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX

PASSWORD=$1
KEY=$2
GUROBI_USER=$3

OPENCONNECT_PID=""

function startOpenConnect(){
    # start here open connect with your params and grab its pid
    echo "${PASSWORD}" | openconnect "sslvpn.ethz.ch/student-net" -u "markmueller@student-net.ethz.ch" --passwd-on-stdin & OPENCONNECT_PID=$!
}

function checkOpenconnect(){
    ps -p "${OPENCONNECT_PID}"
    # print the status so we can check in the main loop
    echo $?
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo | apt-get install openconnect
startOpenConnect
sleep 5
OPENCONNECT_STATUS=$(checkOpenconnect)
echo $OPENCONNECT_STATUS
echo | sudo -u ${GUROBI_USER} $SCRIPT_DIR/gurobi912/linux64/bin/grbgetkey ${KEY}
sudo killall -SIGINT openconnect