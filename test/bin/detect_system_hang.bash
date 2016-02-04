#!/bin/bash

# CAUTION: this script assumes that only one instances of this script and the target binary 
# are running on the current node; otherwise, multiple script instances will interfere with
# each orther

if [ $# -eq 3 ]; then
	APP=$1
	OUTPUT=$2
	TIME_THRED=$3
elif [ $# -eq 2 ]; then
	APP=$1
	OUTPUT=$2
	TIME_THRED=30
elif [ $# -eq 1 ]; then
	APP=$1
	OUTPUT="/dev/null"
	TIME_THRED=30 #half hour
else
	echo "Error: target program is missing"
	echo "====> Usage of this script"
	echo "      $ detect_system_hang.bash [target program] [time threshold (minutes)]"
	echo "      $ detect_system_hang.bash [target program]"
	exit
fi

#echo "Target program: $APP"
#echo "Time threshold: $TIME_THRED"

if [ ! -f $APP ]; then
	echo "Target program does not exist ($APP); exit!" >> ${OUTPUT}
	exit 1
fi

while [ 1 ] 
do
time1=`ps aux | grep $APP | grep -v "grep" | grep -v "detect_system_hang" | awk '{print $10}' | head -1 | awk '{split($0, array, ":")} END{print array[1]}'` 

if [ -z "$time1" ]; then
	exit
fi

if [ $TIME_THRED -le $time1 ]  
then
pid=`ps aux | grep $APP | grep -v "grep" | grep -v "detect_system_hang" | awk '{print $2}' | head -1`
echo "" >> ${OUTPUT}
echo "system hang. kill $pid" >> ${OUTPUT}
echo "" >> ${OUTPUT}
kill -9 $pid
fi

sleep 120s #wake up every 2 minute
#echo "wake up at every 120s"
done
