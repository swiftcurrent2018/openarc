#! /bin/bash

if [ $# -eq 2 ]; then
	aspenmodel=$1
	aspenkernel="main"
	aspenparam=$2
	memchecktool="kernelmemusageinclusive"
elif [ $# -eq 3 ]; then
	aspenmodel=$1
	aspenkernel=$2
	aspenparam=$3
	memchecktool="kernelmemusageinclusive"
elif [ $# -eq 4 ]; then
	aspenmodel=$1
	aspenkernel=$2
	aspenparam=$3
	aspenmap=$4
	memchecktool="singleregionmemusage"
else
	echo "Incorrect number of inputs; exit"
	echo "==> Usage:"
	echo "    ASPENKernelMemUsage.bash [aspen model] [aspen param]"
	echo "    Or"
	echo "    ASPENKernelMemUsage.bash [aspen model] [kernel name] [aspen param]"
	echo "    Or"
	echo "    ASPENKernelMemUsage.bash [aspen model] [kernel name] [aspen param] [map name]"
	exit
fi

if [ "$aspen" = "" ] || [ ! -d "$aspen" ]; then
    echo "Environment variable, aspen, should be set up correctly to run this script; exit."
    exit
fi

#memchecktool="kernelmemusageexclusive"
#memchecktool="kernelmemusageinclusive"
#memchecktool="singleregionmemusage"

checkfile="memcheck.c"

if [ "${memchecktool}" = "singleregionmemusage" ]; then
	${aspen}/tools/analysis/${memchecktool} "${aspenmodel}" ${aspenkernel} "${aspenparam} ${aspenmap}" > "$checkfile"

else
	${aspen}/tools/analysis/${memchecktool} "${aspenmodel}" ${aspenkernel} "${aspenparam}" > "$checkfile"
fi
mv "$checkfile" "${checkfile}_tmp"
cat "${checkfile}_tmp" | sed "s|dssize|HI_aspenmempredict|g" > "${checkfile}_tmp2"
cat "${checkfile}_tmp2" | sed "s|cmath|math.h|g" > "${checkfile}_tmp"
cat "${checkfile}_tmp" | sed "s|double max(double a, double b, double c)|//&|g" > "${checkfile}"
rm "${checkfile}_tmp" "${checkfile}_tmp2"
