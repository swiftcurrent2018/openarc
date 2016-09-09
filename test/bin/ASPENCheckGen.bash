#! /bin/bash

if [ $# -eq 2 ]; then
	aspenmodel=$1
	aspenkernel="main"
	aspenparam=$2
elif [ $# -eq 3 ]; then
	aspenmodel=$1
	aspenkernel=$2
	aspenparam=$3
else
	echo "Incorrect number of inputs; exit"
	echo "==> Usage:"
	echo "    ASPENCheckGen.bash [aspen model] [aspen param]"
	echo "    Or"
	echo "    ASPENCheckGen.bash [aspen model] [aspen param] [kernel name]"
	exit
fi

checkfile="aspenrt.c"

if [ "$aspen" = "" ] || [ ! -d "$aspen" ]; then
    echo "Environment variable, aspen, should be set up correctly to run this script; exit."
    exit
fi

${aspen}/tools/prediction/tradeoff "${aspenmodel}" "${aspen}/models/machine/1cpu1gpu.aspen" ${aspenkernel} nvidia_m2090 intel_xeon_x5660 "${aspenparam}" > "$checkfile"
#${aspen}/tools/prediction/tradeoff "${aspenmodel}" "${aspen}/models/machine/2cpu1gpu.aspen" ${aspenkernel} nvidia_m2090 intel_xeon_x5660 "${aspenparam}" > "$checkfile"
mv "$checkfile" "${checkfile}_tmp"
cat "${checkfile}_tmp" | sed "s|check|HI_aspenpredict|g" > "${checkfile}_tmp2"
cat "${checkfile}_tmp2" | sed "s|cmath|math.h|g" > "${checkfile}_tmp"
cat "${checkfile}_tmp" | sed "s|double max(double a, double b, double c)|//&|g" > "${checkfile}"
rm "${checkfile}_tmp" "${checkfile}_tmp2"
