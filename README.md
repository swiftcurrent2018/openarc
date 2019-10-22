# README #
-------------------------------------------------------------------------------
RELEASE
-------------------------------------------------------------------------------
OpenARC V0.17 (October 22, 2019)

Open Accelerator Research Compiler (OpenARC) is a framework built on top of 
the Cetus compiler infrastructure (http://cetus.ecn.purdue.edu), which is 
written in Java for C.
OpenARC provides extensible environment, where various performance 
optimizations, traceability mechanisms, fault tolerance techniques, etc., 
can be built for better debuggability/performance/resilience on the complex 
accelerator computing. 
OpenARC supports the full feature set of OpenACC V1.0 (+ subset of V2.0) and performs 
source-to-source transformations, targeting heterogeneous devices, such as 
NVIDIA GPUs, AMD GPUs, Intel MICs, and Altera FPGAs.
Please refer to the OpenARC website (http://ft.ornl.gov/research/openarc) to 
find more details on OpenARC.


-------------------------------------------------------------------------------
REQUIREMENTS
-------------------------------------------------------------------------------
* JAVA SE 7 or later
* GCC 4.2 or later
* ANTLRv2 
	- Default antlr.jar file is included in this distribution (./lib)
	

 
-------------------------------------------------------------------------------
INSTALLATION
-------------------------------------------------------------------------------
* Obtain OpenARC distribution
    - The latest version of OpenARC can be obtained at:
    https://code.ornl.gov/f6l/OpenARC.git


* Build

    First, copy make.header.sample to make.header and adjust the
    configuration there for your platform. (To target a Linux system with NVIDIA
	GPUs, choose LCUDA as TARGET_SYSTEM and adjust variables in the LCUDA section 
	as necessary.)    
	Next, there are several options
    for building OpenARC:

    - For Apache Ant users
        
        The provided build.xml defines the build targets for OpenARC. The available
    targets are "compile", "jar", "bin", "clean" and "javadoc". Users need to edit
    the location of the Antlr tool. (build.xml has not yet been updated to build
    OpenARC's LLVM support.)

    - For Linux/Unix command line users
        
        If LLVM support is desired, first build LLVM and jllvm as described in jllvm/README-openarc.  (LLVM support is necessary only for NVL-C and FITL, and NOT included in the public release version.)

        Run the script build.sh (e.g., $ build.sh bin #compile and create a wrapper script)

    - For SDK (Eclipse, Netbeans, etc) users
    
        First, run "make -f configure.mk base", and build the parser with the
        Antlr tool.

        Then, if LLVM support is desired, run "make -f configure.mk llvm" and build LLVM and jllvm as described in jllvm/README-openarc.  (LLVM support is necessary only for NVL-C and FITL, and NOT included in the release version.)

        Then, follow the instructions of each SDK to set up a project.

* Build OpenARC runtime

  - To compile the output program that OpenARC translated from the input OpenACC
  program, OpenARC runtime should be compiled too. (refer to 
  readme_openarcrt.txt in openarcrt directory.)


-------------------------------------------------------------------------------
ENVIRONMENT SETUP
-------------------------------------------------------------------------------
* Environment variable, OPENARC_ARCH, is used to set a target architecture, 
for which OpenARC translates the input OpenACC program. 
(Default target is NVIDIA CUDA if the variable does not exist.) 

  - Set OPENARC_ARCH = 0 for CUDA (default)

                       1 for general OpenCL (e.g., AMD GPUs)

                       2 for Xeon Phi with OpenCL 

                       3 for Altera FPGA with OpenCL

                       4 for MCL with OpenCL

  - For example in BASH, 

        export OPENARC_ARCH=0

* To port OpenACC to non-CUDA devices, OpenACC environment variables,
ACC_DEVICE_TYPE, should be set to the target device type.
  - For example in BASH, if target device is an AMD GPU,

        export ACC_DEVICE_TYPE=RADEON 

* OpenMP environment variable, OMP_NUM_THREADS, shoud be set to the maximum
number of OpenMP threads that the input program uses, which is necessary only if OpenMP is used in
the input OpenACC program.

* Environment variable, OPENARC_JITOPTION, may be optinally used to pass
options to the backend runtime compiler (NVCC compiler options for JIT CUDA 
kernel compilation or clBuildProgram options for JIT OpenCL kernel compilation).
  - For example, if output OpenCL kernel file (openarc_kernel.cl) contains
  header files, path to the header files may need to be specified to the backend
  OpenCL compiler.

        export OPENARC_JITOPTION="-I . -I $openarc/openarcrt"

* Environment variable, OPENARCRT_UNIFIEDMEM, sets whether to use unified
memory if the underlying device supports.

	if 0, unified memory is disabled.

	if 1, use unified memory if the device supports it and appropriate APIs are called.

* Environment variable, OPENARCRT_VERBOSITY, is used to set the verbosity
level of profiling by the OpenARC runtime.

	if 0, OpenARC runtime profiler prints the summary information only.

	if 1, OpenARC runtime profiler prints the entry/exit of OpenACC API calls.

	if 2, OpenARC runtime profiler prints the entry/exit of OpenACC API + HeteroIR API calls.

	if 3, OpenARC runtime profiler prints the entry/exit of OpenACC API + HeteroIR API + underlying driver API calls.

* To run some examples in "test" directory, environment variable, openarc,
should be set to the root directory of this OpenARC package (the directory
where this readme file resides).
  

-------------------------------------------------------------------------------
RUNNING OpenARC
-------------------------------------------------------------------------------
* Users can run OpenARC in the following way:

	$ java -classpath=[user_class_path] openacc.exec.ACC2GPUDriver [options] [C files]

* The "user_class_path" should include the class paths of Antlr and Cetus.
"build.sh" and "build.xml" provides a target (bin) that generates a wrapper script
for OpenARC users; if [openarc-path]/bin/openarc exists, the above command can be shortened as following:

	$ [openarc-path]/bin/openarc [options] [C files]

* Use addIncludePath option to pass paths for non-standard header files:

	$ [openarc-path]/bin/openarc -addIncludePath=[openarc-runtime-path] [C files]

* Use either macro option or "#pragma openarc #define" directive to apply macro definitions to OpenACC/OpenARC annotations; see the LIMITATIONS section.

* When compiled by OpenARC, two macro names (_OPENACC and _OPENARC_) are implicitly defined, which can be used for conditional compiliation.

* Available OpenARC commandline options can be found either in [openarc-path]/test/openarcConf.sample or by running the following command:

	$ [openarc-path]/bin/openarc -dump-options

* A recommended way to pass commandline options to OpenARC is to use the sample configuration file ([openarc-path]/test/openarcConf.sample)

	- Copy the openarcConf.sample file to your working directory, modify it as necessary. 

	- Run OpenARC using the gpuConfFile option.

	$ [openarc-path]/bin/openarc -gpuConfFile=openarcConf.sample [C files]


-------------------------------------------------------------------------------
TESTING
-------------------------------------------------------------------------------
* "./test" directory contains examples showing how to use OpenARC.
For example, to compile and run matmul.c in ./test/examples/openarc/matmul directory:

	$ cd [openarc-path]/test/examples/openarc/matmul

	$ O2Gbuild.script //translate OpenACC to output CUDA or OpenCL program.

	$ make            //compile the generated output program.

	$ cd bin; matmul_ACC //run the output binary.
	
	//The output kernel file (openarc_kernel.cu or openarc_kernel.cl) can be either 
pre-compiled by the built-in binBuilder tool (binBuilder_cuda or binBuilder_opencl) or JIT-compiled at runtime. 

	//To JIT-compile the kernel file, be sure to delete any old kernel binary (openarc_kernel_*.ptx).

-------------------------------------------------------------------------------
FEATURES/UPDATES
-------------------------------------------------------------------------------
* New features
	- Add a basic OpenACC-to-MCL translation pass as an experimental feature.

	- Add several FPGA-specific optimizations (e.g., collapse optimization, reduction optimization, and sliding-window optimization): see ./README_FPGA_example.txt file to learn an example configuration procedure to use the OpenACC-to-FPGA translation framework.

	- Add new flags: expand-user-source and inlineFunctionTransformation

	- Add Altera FPGAs as a new target device

	- Add a fake virtual device address space for OpenCL targets, which allows pointer-arithmetics on the virtual device address for both CUDA and OpenCL devices.

* Updates
    - Update CUDA runtime to allow intermixing of both OpenACC and CUDA. (See example in [openarc-path]/test/examples/openarc/matmul_openacc_cuda)

* Bug fixes and improvements
	- Update the OpenARC C parser to support additional types in C99/C11 and CUDA/OpenCL.

	- Fixed bugs in the gang-private variable transformation and workshing-loop transformation.

	- Fixed bugs in the OpenCL backend to correctly handle multiple platforms.

	- Fixed bugs in privatization and reduction transformation passes.

	- Fixes various bugs related to multi-threading and synchronizations.

	- OpenACC update directives allow subarrays with non-zero start index, which 
offers partial-array transfers between the host and device.

	- Fix bugs in setting an OpenCL driver for MICs.

* Updates in flags


-------------------------------------------------------------------------------
CONTENTS
-------------------------------------------------------------------------------
This OpenARC release has the following contents.

* README.md     - This file
* README_FPGA_example.txt     - Example configuration setup procedure to use OpenACC-to-FPGA framework
* lib                    - Archived classes (jar)
* build.sh               - Command line build script
* build.xml              - Build configuration for Apache Ant
* batchCleanup.bash      - Global cleanup script
* src                    - OpenARC source code
* doc                    - OpenARC documents
* openarcrt              - OpenARC runtime (HeteroIR) source code
* llvm                   - LLVM sources
* jllvm                  - Java bindings for LLVM
* test                   - Examples showing how to use OpenARC


-------------------------------------------------------------------------------
REFERENCE
-------------------------------------------------------------------------------
To cite OpenARC, please use the following papers:

Seyong Lee and Jeffrey S. Vetter, OpenARC: Open Accelerator Research Compiler for Directive-Based, Efficient Heterogeneous Computing, HPDC: International ACM Symposium on High-Performance Parallel and Distributed Computing, Short Paper, June 2014

Seyong Lee and Jeffrey S. Vetter, OpenARC: Extensible OpenACC Compiler Framework for Directive-Based Accelerator Programming Study, WACCPD: Workshop on Accelerator Programming Using Directives in Conjunction with SC'14, November 2014

Bibtex files for the above papers and other OpenARC-related research can be found
in the OpenARC website (http://ft.ornl.gov/research/openarc).


-------------------------------------------------------------------------------
LIMITATIONS
-------------------------------------------------------------------------------
- The underlying C parser in the current implementation supports C99 
features only partially. If parsing errors occur for the valid input 
C program, the program may contain unsuppported C99 features.
    - One of C99 features not fully supported in the current implementation 
	is mixed declaration and code; to fix this, put all variable declaration 
	statements in a function before all code sections in the function body.
        - Example: change "int a; a = 1; int b;" to "int a; int b; a = 1;".

- The current OpenARC implementation follows the memory layout requirement of OpenACC V1.0, which enforces that data in OpenACC data clauses should be continguous 
in the memory. This means that double/triple pointers (e.g., float \*\*p) are not
allowed. One way to allocate 2D array in a contiguous way is the following:

		float (*a)[N] = (float (*)[N])malloc(sizeof(float) * M * N);
			
		//Where N should be compile-time constant.
		
		...
		
		a[i][j] = c; //a can be accessed as if 2D array.

- C preprocessor in the current implementation does not expand macros 
in pragma annotations. To enable this, either 1) use the OpenARC commandline option, macro (e.g., -macro=SIZE=1024) or 2) use "#pragma openarc #define macro value" 
directive in the input source code.

- The current implementation does not support a Struct member in OpenACC data clauses; one way to avoid this problem is to manually decompose struct data.

    - Example: change "struct var { int *data1; int *data2;} aa;" to "int *aa_data1; int *aa_data2;"
    
- The current implementation allows a subarray (partial array) if its start index is 0; subarrays with non-zero start index are allowed only in an OpenACC update directive.

- The current implementation does not support Variable Length Arrays (VLAs).
Even if a host variable is a pointer, it will be changed to VLA if it is used as a private variable in a kernel. To prevent this, the number of elements of data pointed by the pointer should be changed to compile-time constant. 

	- Example: in the following example, _N_ should be compile-time constant.

	#pragma acc parallel loop private(z[0:_N_]) ...

- Current implementation ignores vector clause. (e.g., for CUDA target, 
gang clause is used to set the number of thread blocks, and worker clause is 
used to specify the number of  threads in a thread block.) If a compute region 
has only gang clauses without any worker clause (ignoring vector clauses), 
the compiler may automatically add worker clause where appropriate.

- OpenACC standard assumes that there is an implicit barrier at the end of
each compute region if the async clause is not present, but the OpenARC runtime
may skip the implicit barrier if safe. To disable this optimization, which may be
necessary for correct timing of the compute region execution time, set the
forceSyncKernelCall commandline option to 1.

- Current implementation allows data regions to have compute regions 
interprocedurally, but a compute region can have a function call only if 
the called fuction does not contain any OpenACC loop directive.  

- Current implementation mistakenly moves any struct, union, or enum
definition from a function parameter list to the enclosing scope instead
of treating it as belonging to the function's scope.  Such definitions are
often considered bad practice anyway and normally produce warnings from gcc
or clang.

- Current implementation fails to enter into its symbol table any struct or
union that is referenced (e.g., "struct S \*s;") but never explicitly
declared (e.g., "struct S;") or defined (e.g., "struct S { int i; };").
Such usage is permitted in C and can occur when a struct or union is
intended to remain opaque within a translation unit.  So far, this bug does
not appear to impact OpenARC's user-visible behavior, but it may impact code
extending OpenARC.  A workaround is to forward-declare such a struct or
union before referencing it.

- In current implementation, C parser recognizes all directives as standalone 
annotations, which will be incorrect if they are non-standalone types. This error is corrected by the following OpenACC/OpenMP parsers, which re-attaches non-standalone 
directives to structured blocks. The two-step parsing may not work if either for-loop or if-statement has only one child statement without using bracket. Therefore, the following examples will be parsed incorrectly:

	//Example1

	//C parser will incorrectly parse the i-loop having the OpenACC directive as its body, since the C parser recognizes all directives as standalone types.

		for(i=0; i<N; i++)
	
		#pragma acc parallel loop
	
		for(j=0; j<M; j++) { ...//kernel loop body }

	//C parser will parse the i-loop correctly.

		for(i=0; i<N; i++) {
	
		#pragma acc parallel loop
	
		for(j=0; j<M; j++) { ...//kernel loop body }
	
		}

	//Example2

	//C parser will incorrectly parse the if-statement.

		if(cond) 

			#pragma acc parallel loop

			for(j=0; j<M; j++) { ...//kernel loop body }

	//C parser will correctly parse the if-statement.

		if(cond) {

			#pragma acc parallel loop

			for(j=0; j<M; j++) { ...//kernel loop body }

		}

The OpenARC Team

URL: http://ft.ornl.gov/research/openarc

EMAIL: lees2@ornl.gov
