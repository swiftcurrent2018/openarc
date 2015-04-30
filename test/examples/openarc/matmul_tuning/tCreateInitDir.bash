#! /bin/bash

#######################################
# Modify the following conf variables #
#######################################
inputData=( 512 ) 
benchmark="matmul_tuning"
inputCFiles=( "matmul.c" )
inputHeaderFiles=( )
inputFiles=( "${inputCFiles[@]}" "${inputHeaderFiles[@]}" )


if [ "$openarc" = "" ] || [ ! -d "$openarc" ]; then
	echo "Environment variable, openarc, should be set up correctly to run this script; exit."
	exit
fi

baseDir="${openarc}/test/examples/openarc"
workDir="${baseDir}/${benchmark}"
cetus_input="cetus_input"
orgSrcDir="${workDir}/src"
outputDirBase="$workDir"


################################
# Create cetus_input directory #
################################
cd ${workDir}
if [ ! -d ${cetus_input} ]; then 
  mkdir ${cetus_input}
fi  

cd "${workDir}/${cetus_input}"

for inputDir in ${inputData[@]}
do
  if [ ! -d ${inputDir} ]; then 
    mkdir ${inputDir}
  fi
	cd ${inputDir}
	srcDir="Src"
  if [ ! -d ${srcDir} ]; then 
    mkdir ${srcDir}
  fi
  for filename in ${inputFiles[@]}
  do
    cp "${orgSrcDir}/${filename}" "./${srcDir}"
  done

  cd "${workDir}/${cetus_input}"
done  

#################################
# Create cetus output directory #
#################################
cd ${outputDirBase}
if [ ! -d "cetus_output" ]; then 
  mkdir "cetus_output"
fi  
