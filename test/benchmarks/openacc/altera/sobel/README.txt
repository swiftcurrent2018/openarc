To build:
  ./O2GBuild.script;
  make;
  
To execute:
  cd ./bin;
  ./sobel_ACC ../butterflies.ppm;

Kernel Versions:
  nd_range.c - 
      ND-Range kernel
  window.c - 
      Single Work Item Kernel, with the window directive applied
