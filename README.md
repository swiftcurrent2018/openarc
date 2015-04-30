# README #
-------------------------------------------------------------------------------
RELEASE
-------------------------------------------------------------------------------
OpenARC V0.3 (April 22, 2015)

Open Accelerator Research Compiler (OpenARC) is a framework built on top of 
the Cetus compiler infrastructure (http://cetus.ecn.purdue.edu), which is 
written in Java for C.
OpenARC provides extensible environment, where various performance 
optimizations, traceability mechanisms, fault tolerance techniques, etc., 
can be built for better debuggability/performance/resilience on the complex 
accelerator computing. 
OpenARC supports the full feature set of OpenACC V1.0 and performs 
source-to-source transformations, targeting heterogeneous devices, such as 
NVIDIA GPUs, AMD GPUs, and Intel MICs.
Please refer to the OpenARC website (http://ft.ornl.gov/research/openarc) to 
find more details on OpenARC.


-------------------------------------------------------------------------------
REQUIREMENTS
-------------------------------------------------------------------------------
* JAVA SE 7
* GCC
* ANTLRv2 
	- Default antlr.jar file is included in this distribution (./lib)
* LLVM 3.2
	- If OpenARC's LLVM support is desired
	- See jllvm/README-openarc for details

 
-------------------------------------------------------------------------------
INSTALLATION
-------------------------------------------------------------------------------
* Obtain OpenARC distribution
    - The latest version of OpenARC can be obtained at:
    https://code.ornl.gov/f6l/OpenARC


* Build

    First, copy make.header.sample to make.header and adjust the
    configuration there for your platform.  Next, there are several options
    for building OpenARC:

    - For Apache Ant users
        
        The provided build.xml defines the build targets for OpenARC. The available
    targets are "compile", "jar", "bin", "clean" and "javadoc". Users need to edit
    the location of the Antlr tool. (build.xml has not yet been updated to build
    OpenARC's LLVM support.)

    - For Linux/Unix command line users
        
        Run the script build.sh after defining system-dependent variables in the script. (e.g., $ build.sh bin #compile and create a wrapper script)

        If LLVM support is desired, first build jllvm as described in jllvm/README-openarc, and then build OpenARC using build.sh.

    - For SDK (Eclipse, Netbeans, etc) users
    
        First, run "make -f configure.mk base", and build the parser with the
        Antlr tool.

        Then, if LLVM support is desired, run "make -f configure.mk llvm" and build jllvm.

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

                       1 for OpenCL (e.g., AMD GPUs)

                       2 for OpenCL for Xeon Phi
  - For example in BASH, 

        export OPENARC_ARCH=0

* To port OpenACC to non-CUDA devices, OpenACC environment variables,
ACC_DEVICE_TYPE, should be set to the target device type.
  - For example in BASH, if target device is an AMD GPU,

        export ACC_DEVICE_TYPE=RADEON 

* OpenMP environment variable, OMP_NUM_THREADS, shoud be set to the maximum
number of OpenMP threads that the input program uses, if OpenMP is used in
the input OpenACC program.

* Environment variable, OPENARC_JITOPTION, may be optinally used to pass
options to the backend runtime compiler (NVCC compiler options for JIT CUDA 
kernel compilation or clBuildProgram options for JIT OpenCL kernel compilation).
  - For example, if output OpenCL kernel file (openarc_kernel.cl) contains
  header files, path to the header files may need to be specified to the backend
  OpenCL compiler.

        export OPENARC_JITOPTION="-I ."

* Environment variable, OPENARCRT_VERBOSITY, is used to set the verbosity
level of profiling by the OpenARC runtime.

	if 0, OpenARC runtime profiler prints the summary information only.

	if 1, OpenARC runtime profiler prints the entry/exit of OpenACC API calls.

	if 2, OpenARC runtime profiler prints the entry/exit of OpenACC API + HeteroIR API calls.

	if 3, OpenARC runtime profiler prints the entry/exit of OpenACC API + HeteroIR API + underlying driver API calls.

* Environment variable, OPENARCRT_UNIFIEDMEM, sets whether to use unified
memory if the underlying device supports.

	if 0, unified memory is disabled.

	if 1, use unified memory if the device supports it and appropriate APIs are called.

* To run some examples in "test" directory, environment variable, openarc,
should be set to the root directory of this OpenARC package (the directory
where this readme file resides).
  

-------------------------------------------------------------------------------
RUNNING OpenARC
-------------------------------------------------------------------------------
Users can run OpenARC in the following way:

  $ java -classpath=[user_class_path] openacc.exec.ACC2GPUDriver [options] [C files]

The "user_class_path" should include the class paths of Antlr and Cetus.
"build.sh" and "build.xml" provides a target (bin) that generates a wrapper script
for OpenARC users; if [openarc-path]/bin/openarc exists, the above command can be shortened as following:

  $ [openarc-path]/bin/openarc [options] [C files]


-------------------------------------------------------------------------------
TESTING
-------------------------------------------------------------------------------
"./test" directory contains examples showing how to use OpenARC.


-------------------------------------------------------------------------------
FEATURES/UPDATES
-------------------------------------------------------------------------------
* New features

* Updates

* Bug fixes and improvements

* Updates in flags


-------------------------------------------------------------------------------
CONTENTS
-------------------------------------------------------------------------------
This OpenARC release has the following contents.

* README.md     - This file
* lib                    - Archived classes (jar)
* build.sh               - Command line build script
* build.xml              - Build configuration for Apache Ant
* batchCleanup.bash      - Global cleanup script
* src                    - OpenARC source code
* doc                    - OpenARC documents
* openarcrt              - OpenARC runtime (HeteroIR) source code
* jllvm                  - Java bindings for LLVM
* test                   - Examples showing how to use OpenARC


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

	float (\*a)[N] = (float (\*)[N])malloc(sizeof(float) \* M \* N);

	//Where N should be compile-time constant.

	...

	a[i][j] = c; //a can be accessed as if 2D array.

- C preprocessor in the current implementation does not expand macros 
in pragma annotations. To enable this, either 1) use the OpenARC commandline option, macro (e.g., -macro=SIZE=1024) or 2) use "#pragma openarc #define macro value" 
directive in the input source code.

- Class member in OpenACC data clauses may not work; one way to avoid this problem is to manually decompose struct data.

    - Example: change "struct var { int *data1; int *data2;} aa;" to "int *aa_data1; int *aa_data2;"
    
- Class member is not allowed in an OpenACC subarray, and the start index 
of a subarray should be 0 (partial array passing is allowed only if its start 
index is 0.)

- Current implementation ignores vector clause. (e.g., for CUDA target, 
gang clause is used to set the number of thread blocks, and worker clause is 
used to specify the number of  threads in a thread block.) If a compute region 
has only gang clauses without any worker clause (ignoring vector clauses), 
the compiler may automaticall add worker clause where appropriate.
 
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


April 22, 2015

The OpenARC Team

URL: http://ft.ornl.gov/research/openarc

EMAIL: lees2@ornl.gov
