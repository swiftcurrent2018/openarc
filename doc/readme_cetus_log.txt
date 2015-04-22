RELEASE NOTES IN THE PAST
-------------------------

--------------------------------------------------------------------------------
RELEASE
--------------------------------------------------------------------------------
Cetus 1.3 (June 13, 2011)

Cetus is a source-to-source compiler infrastructure for C written in Java, and
can be downloaded from http://cetus.ecn.purdue.edu. This version contains
improvements to various analysis/transformation passes, both in their
effectiveness and efficiency.
As a side note, we are preparing for Cetus Users and Compiler Infrastructure
Workshop in October -- visit http://cetus.ecn.purdue.edu/cetusworkshop for more
information.

--------------------------------------------------------------------------------
FEATURES/UPDATES
--------------------------------------------------------------------------------
* Bug fixes
  - Points-to analysis produces the same result over repeated invocations while
    the previous version did not.
  - Added checks in LoopProfiler to avoid profiling loops that contain jumps to
    program points other than the loop exits.
  - Fixed the OpenMP code generator so that it excludes variables declared
    within the relevant loop from the private clause.
  - Fixed the expression simplifier's behavior that produced incorrect results
    on logical expressions.
  - Fixed the dependence analyzer's behavior that produced incorrect results on
    certain scalar variables which contain pointer dereferences.
  - Fixed a bug that prevented selecting parallel loops for OpenMP
    parallelization in certain cases.

* New analysis/transform passes
  - Inline expansion
    This pass has been mature since the last release. Various sub-options
    provide configurable behaviors of the inline transformation.
  - Unreachable branch elimination
    This pass detects unreachable code due to branches that can be evaluated
    at compile time and eliminates such code. The tested branches are in IF,
    FOR, WHILE, DO, and SWITCH statements. See the "-teliminate-branch" flag to
    see how to invoke this transformation.
  - Reduction transform
    This pass produces OpenMP-conforming code from the reduction items
    recognized by the reduction analysis in Cetus. Two types of transformations
    are performed; one is scalar replacement for loop-invariant expressions, and
    the other is array reduction code generation. This transformation pass is
    invoked by adjusting the value of the flag "-reduction".
  - Def-Use chains and its interface
    This analysis computes def-use chains and use-def chains
    interprocedurally for a program. The analysis returns a list of
    statements from a given def or use expression. The analysis is located
    in the package "cetus.application". The interface, DefUseChain and
    UseDefChain, and its implementation, IPChainAnalysis are the key classes
    for this feature.    

* Updates to automatic parallelization
  - Enhanced report of automatic parallelization
    We added options that enables detailed reporting of automatic
    parallelization. The report shows variables that may carry data dependences
    if the enclosing loop is not parallelized.
    This reporting is turned on when users pass the following flags:
      -parallelize-loops=3
        Every loop is analyzed for automatic parallelization but only the
        outermost parallel loops are scheduled for parallelization (OpenMP)
      -parallelize-loops=4
        Every loop is analyzed for automatic parallelization and every parallel
        loop is scheduled for parallelization
  - Enhanced handling of known library calls
    Handling of library calls in analysis passes has been improved to produce
    less conservative analysis results. We inserted comprehensive check routines
    in range analysis, array privatization pass, and loop parallelization pass
    so that they identify side-effect-free standard library calls and make less
    conservative decisions. This behavior is unsafe when the input program
    contains a call to a non-standard-library function that matches the
    signature of a standard-library function.
  - Enhanced handling of reductions
    The previous version was not properly handling reduction items that do not
    conform to the OpenMP specification. A new pass was added and called by the
    loop parallelization pass to overcome this limitation.
    See the "ReductionTransform" pass for more details.
  - Enhanced OpenMP code generation with a simple performance model
    Cetus' OpenMP code generator now inserts simple run-time checks through
    OpenMP IF clauses so that it serialize parallel loops that contain
    insufficient work.
  - As a result, the current version finds 21 more parallel loops than the
    previous version does for our test suite (NPB, SPEC OMP).

* Updates for efficiency and consistency
  We found that Cetus suffers from limited scalability in terms of memory usage 
  and put significant effort into improving Cetus' memory usage. This
  optimization includes minimization of object clones, use of in-place
  operations if possible, use of efficient collections, and other optimizations
  for time efficiency.
  The following numbers summarize the result of this optimization:
    Improvements since the last release (1.2.1)
    Environments: Quad-core AMD Opteron at 2.5GHz
                  JAVA SE 1.6.0_25 with -Xmx=1500m
    Elapsed time for building and printing IRs for 25 NPB/SPEC codes
        1.2.1 : 709 seconds
        1.3   : 242 seconds
    Elapsed time for automatic parallelization for 11 NPB/SPEC codes
        1.2.1 : 143 seconds
        1.3   : 110 seconds

* Updates in flags
  - New flags
    -debug_parser_input
        Print a single preprocessed input file before sending it to the parser,
        and exit
    -debug_preprocessor_input
        Print a single pre-annotated input file before sending it to the
        preprocessor, and exit
    -dump-options
        Create file options.cetus with default options
    -dump-system-options
        Create a system wide file options.cetus with default options
    -parser=PARSERNAME
        Name of the parser to be used for parsing the source files
    -teliminate-branch=N
        Eliminates unreachable branch targets
    -tinline=SUBOPTION
        This flags replaces "-tinline-expansion"
    -profitable-omp=N
        Inserts runtime checks for selecting profitable omp parallel region
  - Modified flags
    -ddt=N
        ddt=3 was removed from the help message
    -parallelize-loops=N
        By default, only outermost loops are parallelized
        Added options for printing parallelization report
    -profile-loops=N
        Clarified the types of profiled regions -- loop, OpenMP parallel region,
        and OpenMP for loop
  - Removed flags
    -tinline-expansion
        -tinline replaces this flag

--------------------------------------------------------------------------------
CONTENTS
--------------------------------------------------------------------------------
This Cetus release has the following contents.

  lib            - Archived classes (jar)
  license.txt    - Cetus license
  build.sh       - Command line build script
  build.xml      - Build configuration for Apache Ant
  src            - Cetus source code
  readme.txt     - This file
  readme_log.txt - Archived release notes
  readme_omp2gpu.txt - readme file for OpenMP-to-CUDA translator

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
* JAVA SE 6
* ANTLRv2 
* GCC
 
--------------------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------------------
* Obtain Cetus distribution
  The latest version of Cetus can be obtained at:
  http://cetus.ecn.purdue.edu/

* Unpack
  Users need to unpack the distribution before installing Cetus.
  $ cd <directory_where_cetus.tar.gz_exists>
  $ gzip -d cetus.tar.gz | tar xvf -

* Build
  There are several options for building Cetus:
  - For Apache Ant users
    The provided build.xml defines the build targets for Cetus. The available
    targets are "compile", "jar", "clean" and "javadoc". Users need to edit
    the location of the Antlr tool.
  - For Linux/Unix command line users.
    Run the script build.sh after defining system-dependent variables in the
    script.
  - For SDK (Eclipse, Netbeans, etc) users
    Follow the instructions of each SDK.

--------------------------------------------------------------------------------
RUNNING CETUS
--------------------------------------------------------------------------------
Users can run Cetus in the following way:

  $ java -classpath=<user_class_path> cetus.exec.Driver <options> <C files>

The "user_class_path" should include the class paths of Antlr and Cetus.
"build.sh" and "build.xml" provides a target that generates a wrapper script
for Cetus users.

--------------------------------------------------------------------------------
TESTING
--------------------------------------------------------------------------------
We have tested Cetus successfully using the following benchmark suites:

* SPEC CPU2006
  More information about this suite is available at http://www.spec.org

* SPEC OMP2001
  More information about this suite is available at http://www.spec.org

* NPB 2.3 written in C
  More information about this suite is available at
  http://www.hpcs.cs.tsukuba.ac.jp/omni-openmp/

June 13, 2011
The Cetus Team

URL: http://cetus.ecn.purdue.edu
EMAIL: cetus@ecn.purdue.edu


RELEASE
-------
Cetus 1.2.1 (September 10, 2010)

Cetus is a source-to-source compiler infrastructure for C written in Java.
http://cetus.ecn.purdue.edu. This release is a minor one that includes updates
for bug fixes since version 1.2. If you are interested in the OpenMP-to-CUDA
package, check readme.omp2gpu for more details.

FEATURES/UPDATES
----------------
* Bug fixes

  version 1.2.1
  - Fixed bugs in points-to analysis which misses points-to relationship when
    passing a whole global array as a parameter.
  - Fixed incorrect and inefficient use of alias analysis during data dependence
    analysis.
  - Fixed incorrect transformation in the following normalizing passes.
     Single-call transformation (-tsingle-call)
     Loop-normalization transformation (-normalize-loops)
  - Improved stability of the inline expansion transformation.
  - Several bug fixes in OpenMP-to-CUDA translator (see readme.omp2gpu).

  version 1.2
  - Minimal support for GCC __asm__ extension.
  - K&R-style functions can be preserved in the output code with a command-line
    option.

* New flags
  -alias=N
    Specify level of alias analysis
      =0 disable alias analysis
      =1 advanced interprocedural analysis (default)
         Uses interprocedural points-to analysis
  -normalize-return-stmt
    Normalize return statements for all procedures
  -range=N
    Specifies the accuracy of symbolic analysis with value ranges
      =0 disable range computation (minimal symbolic analysis)
      =1 enable local range computation (default)
      =2 enable inter-procedural computation (experimental)
  -preserve-KR-function
    Preserves K&R-style function declaration

* Updated flags
  -ddt=N
    Perform Data Dependence Testing
      =1 banerjee-wolfe test (default)
      =2 range test
  -reduction=N
    Perform reduction variable analysis
      =1 enable only scalar reduction analysis (default)
      =2 enable array reduction analysis as well

* Removed flags
  -argument-noalias        -> merged into -alias flag
  -argument-noalias-global -> merged into -alias flag
  -no-alias                -> merged into -alias flag
  -no-side-effect          -> obsolete
  -normalize               -> renamed as -normalize-loops
  -loop-interchange        -> not active
  -loop-tiling             -> not active

* Experimental flags
  -tinline-expansion

* Updates in cetus.hir package

  - Identifier and Symbol interface
    We found that management of identifiers and symbols in the previous version
    may introduce IR inconsistency when a transformation pass makes low-level
    modifications to the program IR, resulting in a descent amount of increase
    in development time. To alleviate this problem, we decided to choose
    consistency over flexibility in the Cetus IR. One important
    consistency-enhancing change is to protect low-level modifications related
    to symbol and identifiers. The followings list the outstanding differences
    from the previous Cetus IR.
    1. Construction of an Identifier object from a string name is prohibited.
      An Identifier object should be created from a Symbol object that has been
      created before the Identifier is being created.
    2. A new class, NameID, is used in place of Identifier when it needs direct
      name creation. Declaring variables, functions, and user types falls to
      this situation.
    3. The link to the corresponding Symbol object is stored in an Identifier
      object, not in an IDExpression object.
    4. Direct modification to the symbol look-up table in a SymbolTable object
      is prohibited. It is still possible to access the list of declarations and
      symbols through the newly added interface methods SymbolTable.getSymbols()
      and SymbolTable.getDeclarations(). 

  - Protection of IR
    We also found many modification operations may allow reuse of a Traversable
    object that is already in the program IR, which should not be allowed.
    Typical examples of such operations are constructors and high-level
    modifiers of the IR classes. We revisited the implementation of such
    operations and provided a safety net by making them throw exceptions when
    they encounter such attempts.

  - IR comparison
    The comparison semantics of the base classes were clarified. We have been
    using Traversable objects in a collection without explicitly defined
    comparison methods. We provided implementation of equals() and hashCode()
    methods in the four base classes, Declaration, Declarator, Expression, and
    Statement. Expression is the only base IR that performs contents comparison,
    with the others performing identity comparison.

  - Comprehensive list of changes in cetus.hir package
    1. Added NameID.java (see "Identifier and Symbol interface")
    2. IDExpression.getSymbol() .setSymbol() were moved to Identifier.java
    3. SymbolTable.getTable() was removed
    4. Added containsSymbol(), containsDeclaration(), getSymbols(), and
      getDeclarations() in SymbolTable.java
    5. Added getDeclaration() and setName() in Symbol.java
    6. Added missing equals() and hashCode() methods in the base classes
    7. Disabled addition of a DeclarationStatement object to a CompoundStatement
      object (respecting the C language)
    8. Separated Tools.java into DataFlowTools, IRTools, PrintTools,
      SymbolTools, and Tools, following their semantics. Access to Tools method
      is still allowed but will throw "deprecated" warnings.
    9. Added StandardLibrary.java for minimal support of standard library calls.

* New analysis passes
  - Interprocedural analysis (IPA)
    Basic tools for performing interprocedural analysis are provided. The tools
    include call-graph/workspace generation and a set of simple solvers. The
    call-graph utility is implemented in IPAGraph, IPANode, and CallSite classes
    in the cetus.analysis package, and those classes represent a call graph, a
    node in the graph, and a call site in the graph respectively. Base framework
    for IPA is implemented in IPAnalysis, from which pass writers can
    instantiate their own IPA passes.
    The current version of Cetus uses this common framework to increase the
    capability of Range analysis and Points-to analysis.

  - Points-to analysis
    Cetus now provides a powerful pointer analysis framework through its 
    implementation of an interprocedural points-to analyzer based on the 
    design described in:
    "A practice interprocedural alias analysis for an optimizing/parallelizing 
    C compiler", M. Emami, McGill University, 1993.
    
    PointsToAnalysis implements an intraprocedural flow-sensitive 
    driver for this analyzer. Pointer relationships are created at 
    every program point i.e. every Statement, through iterative work-list 
    based traversal of the control flow graph provided by CFGraph. 
    Data structure information and functionality for representing and 
    handling these pointer relationships can be found in PointsToDomain, 
    PointsToRel, UniverseDomain and the Symbol interface. The output 
    of the intraprocedural analysis which is a map from Statement to 
    PointsToDomain (set of points-to information), is used by the 
    interprocedural analyzer.

    The interprocedural analysis of points-to relation was built upon the
    common interprocedural framework introduced in the current version. The
    role of the interprocedural analysis is to reflect the effect of entering
    or returning from a called procedure, by appropriately renaming the
    variables that appear in the points-to relations. The renaming algorithm is
    quite similar to one that appears in the reference implementation, but we
    adjusted the interprocedural driver so that it works on top of our common
    IPA framework. Currently, the IPA points-to analysis does not support full
    context sensitivity and gives up analysis of programs containing function
    calls through function pointers.

    Currently, we have extensively tested and verified the results of the points-to 
    analysis framework on the SPEC OMP2001 (except ammp) and NPB benchmark suites. 

* Updates to Alias Analysis
  Cetus originally provided a simple conservative alias analysis pass 
  that was flow and context insensitive. This analysis was supported by 
  command-line flags that allowed manual input in order to improve 
  interprocedural handling of alias information. With the addition of 
  an interprocedural points-to analyzer to Cetus, the default for 
  generating alias information is via the result provided by points-to 
  analysis. Changes are reflected in the update to input flags outlined 
  above. The API for accessing alias information via the analysis pass 
  remains the same, and provides results whose accuracy is based on that 
  provided by the points-to analyzer.

* Automatic Parallelization updates
1. Improved alias information
  As outlined above, advanced interprocedural points-to analysis is used 
  to provide more accurate 'statement-specific' alias information. This 
  is used to its advantage by the dependence analysis framework, thus 
  increasing the chances of disambiguation via alias information. The 
  result is a significant increase in the number of loops identified by 
  Cetus as parallel.
 
2. Field access disambiguation
  Cetus handles scalar dependences through the information provided by 
  the privatization and reduction analysis passes. This release of 
  Cetus handles disambiguation of fields inside aggregate structures 
  more accurately, thus providing better loop parallelization.  

3. Parallelization results
  Automatic interprocedural alias information significantly improves 
  parallelization capabilities of Cetus. This is reflected strongly 
  in the number of loops automatically parallelized using the Cetus 
  infrastructure. 

* Normalize Return Statements
  This pass is a simple transformation added to this release of Cetus 
  in order to normalize return statements inside procedures. This 
  transformation simplifies one or more return statements inside the 
  procedure by introducing a return variable. All expressions inside 
  a return statement are then moved to an assignment to the new 
  variable before the return statement. This ensures a simple 
  return statement that does not modify program state and 
  is specifically used in the interprocedural points-to analyzer in 
  this release. NOTE: This transformation is different from the 
  pre-existing SingleReturn pass in Cetus. 

* Deprecated features
  - cetus.analysis.NormalExpression - replaced by cetus.hir.Symbolic
  - cetus.hir.Simplifier - obsolete
  - cetus.hir.Tools - separated into five classes (see above)

* OpenMP-to-CUDA Translation 
  This release includes the alpha version of OpenMP-to-CUDA translator,
  which is built on top of Cetus compiler infrastructure. The current version
  of the translator is invoked separately with a customized driver, but we
  envision a Cetus driver that invokes the translator as one of translation
  passes in a future release.
  More information can be found in readme.omp2gpu file.

CONTENTS
--------
This Cetus release has the following contents.

  src       - Cetus source code
  lib       - Archived classes (jar)
  api       - JAVA documents 
  build.sh  - Command line build script
  build.xml - Build configuration for Apache Ant
  readme    - this file
  readme.omp2gpu - readme file for OpenMP-to-CUDA translator
  license   - Cetus license

REQUIREMENTS
------------
* JAVA 2 SDK, SE 1.5.x (or later)
* ANTLRv2 
* GCC
 
INSTALLATION
------------
* Obtain Cetus distribution
  The latest version of Cetus can be obtained at:
  http://cetus.ecn.purdue.edu/

* Unpack
  Users need to unpack the distribution before installing Cetus.
  $ cd <directory_where_cetus.tar.gz_exists>
  $ gzip -d cetus.tar.gz | tar xvf -

* Build
  There are several options for building Cetus:
  - For Apache Ant users
    The provided build.xml defines the build targets for Cetus. The available
    targets are "compile", "jar", "clean" and "javadoc". Users need to edit
    the location of the Antlr tool.
  - For Linux/Unix command line users.
    Run the script build.sh after defining system-dependent variables in the
    script.
  - For SDK (Eclipse, Netbeans, etc) users
    Follow the instructions of each SDK.

RUNNING CETUS
-------------
Users can run Cetus in the following ways:

  $ java -classpath=<user_class_path> cetus.exec.Driver <options> <C files>

The "user_class_path" should include the class paths of Antlr and Cetus.
"build.sh" and "build.xml" provides a target that generates a wrapper script
for Cetus users.

TESTING
-------
We have tested Cetus successfully using the following benchmark suites:

* SPECCPU2006
  More information about this suite available at www.spec.org

* SPECOMP2001
  More information about this suite available at www.spec.org

* NPB3.0
  More information about NAS Parallel Benchmarks at www.nas.nasa.gov/Software/NPB/

September 10, 2010
The Cetus Team

URL: http://cetus.ecn.purdue.edu
EMAIL: cetus@ecn.purdue.edu



RELEASE
--------
Cetus 1.1 (July 10, 2009)

Cetus is a source-to-source compiler infrastructure for C written in Java.
http://cetus.ecn.purdue.edu

FEATURES/UPDATES
----------------
* Bug fixes
  - Illegal identifiers from parsing unnamed structs
    The previous version used the file name as the prefix for unnamed structs
    allowing struct names that start with numeric characters. The current
    version adds a fixed head "named_" before the original name and handles
    such cases safely.
  - Fixed bugs in the following transformation passes
    -tsingle-declarator
    -tsingle-return
    -normalize

* New flags
  -argument-noalias
    Specifies that arguments (parameters) don't alias each other but may alias
    global storage
  -argument-noalias-global
    Specifies that arguments (parameters) don't alias each other and don't
    alias global storage
  -loop-interchange
    Interchange loop to improve locality
  -macro
    Sets macros for the specified names with comma-separated list (no space is
    allowed). e.g., -macro=ARCH=i686,OS=linux
  -no-side-effect
    Assume there is no side-effect in the function calls (for range analysis)
  -profile-loops
    Inserts loop profiling calls (1=every, 2=outer, 3=cetus parallel, 4=outer
    cetus parallel, 5=openmp 6=outer openmp)

* Removed flags
  -antlr     : default behavior
  -parse-only: default behavior
  -usage     : default behavior
  -cfg       : no support
  -inline=N  : no support
  -openmp    : no support
  -procs=N   : no support
  -tloops-to-subroutine: no support

* Experimental flags
  -tsingle-call
    It is not stable for now. Contact the cetus developers for support.

* Improved symbolic tools
  The accuracy of the symbolic range analysis was improved substantially and
  it provides tighter bounds of integer variables used as index variables than
  the previous version does. We have also improved the symbolic expression
  comparator so that it handles complex expressions such as division
  expressions.
  We also provide a new set of simplification utilities in
  "cetus.hir.Symbolic" which is cleaner and efficient compared to the
  existing one in "cetus.analysis.NormalExpression". It is backward-compatible
  with "NormalExpression", so users can safely switch to "Symbolic". The
  following methods are commonly used features:
  - Symbolic.simplify(e)     : simplifies the given expression e
  - Symbolic.add(e1, e2)     : performs e1+e2 followed by simplification
  - Symbolic.multiply(e1, e2): performs e1*e2 followed by simplification
  - Symbolic.subtract(e1, e2): performs e1-e2 followed by simplification
  - Symbolic.divide(e1, e2)  : performs e1/e2 followed by simplification

* Annotations
  The new version of Cetus extends our initial implementation of Annotations
  to a completely new IR structure and API. Comments, pragmas and other
  annotations were initially parsed in as Annotations and enclosed inside
  DeclarationStatements (in most cases).
  In the new implementation, parsed in annotations are converted to a new
  internal representation through the AnnotationParser. The AnnotationParser
  stores annotations under corresponding subclasses:
  - PragmaAnnotation
    - CetusAnnotation (#pragma cetus ...)
    - OmpAnnotation (#pragma omp ...)
  - CommentAnnotation (e.g. /* ... */)
  - CodeAnnotation (Raw printing)
  Annotations are stored as member objects of "Annotatable" IR (Statements
  and Declarations), the new API for manipulating these is hence available
  through Annotatable. Standalone annotations are enclosed in special IR
  such as AnnotationDeclaration or AnnotationStatement (note that
  AnnotationStatement is different from previous release).
  The API in Annotatable *COMPLETELY REPLACES* previous functionality
  provided through Tools.java. We understand this can require substantial
  updates for Cetus users, but we believe the new representation is much
  cleaner and warrants a small amount of time required to port over to the
  new release.

* Automatic Parallelization updates:
1. Improved symbolic handling by Banerjee test
  The DDTDriver framework now uses symbolic information through range analysis
  in order to simplify symbolic subscripts and symbolic values associated with
  loop bounds and increments. This information enhances the data dependence
  results provided by the Banerjee-Wolfe inequalities leading to better
  parallelization results.

2. Nonperfect loop nest handling
  The DDT and parallelization frameworks now support testing for parallel
  loops in non-perfect nests by using common enclosing loop information
  and  accommodating variable length direction vectors during
  parallelization.

3. Improved function call handling inside of loops
  The DDT framework can now handle system calls inside of loops that are
  known to have no side effects. This is currently a fixed list of functions
  that includes log(), sqrt(), fabs() and possibly other calls
  encountered in parallelizable loops in our benchmarks. Future releases 
  will allow this list to be extended. If you are aware of a parallelizable
  function call within your application, it can be easily added to this 
  list. Look for details in LoopTools.java
  The framework also uses simple interprocedural side-effect analysis in
  order to determine eligible loops for parallelization.

4. Interface to Data Dependence Graph
  A more efficient and convenient implementation of the Data Dependence graph
  is used in this version of Cetus. The dependence graph is created once and
  attached to the Program IR object for use by other passes. A reference to this
  DDGraph object gives access to the dependence information API provided by
  DDGraph. The DDGraph implementation and API provided in Cetus v1.0 have been
  deprecated.

5. Induction variable substitution pass
  We provide a fully working induction variable substitution pass which was
  experimental before. It was designed to support "Generalized Inudction
  Variables" that may increase multiple times in multiply nested loops. We use
  aggressive symbolic manipulators for this transformation pass to maximize
  the coverage of the transformation and the subsequent loop parallelizer.
  The capability of the IV substitution pass is illustrated in the example code
  for the range test below.

6. Non-linear and symbolic data dependence test (Range Test)
  We ported the powerful non-linear and symbolic DD test, "Range Test", which
  was implemented in the Polaris parallelizing compiler for Fortran programs.
  The test disproves dependences based on the "overlap" checking which compares
  extreme values of subscripts for the whole loop or between two consecutive
  iterations in conjunction with the monotonicity properties of the subscripts.
  It works seamlessly with the data dependence framework in Cetus, and users
  can turn on the range test by specifying the flag "-ddt=3".
  The following example highlights the capability of the range test when
  parallelizing the given loop in conjunction with the induction variable
  substitution pass:

    k = 0;                           k = 0;
    for (i=0; i<10; i++) {           #pragma omp parallel for private(i)
      k = k+i;                -->    for (i=0; i<imax; i++) {
      a[k] = ...;                      a[(i+i*i)/2] = ...;
    }                                }

7. Improved parallelization results
  The above updates have led to significant improvement in automatic 
  parallelization results using Cetus. You can refer to our latest
  Cetus tutorial at http://cetus.ecn.purdue.edu for detailed
  information.

CONTENTS
--------
This Cetus release has the following contents.

  src       - Source codes of Cetus
  lib       - Archived classes (jar)
  api       - JAVA documents 
  build.sh  - Command line build script
  build.xml - Build configuration for Apache Ant
  readme    - this file
  license   - Cetus license

REQUIREMENTS
------------
* JAVA 2 SDK, SE 1.5.x (or later)
* ANTLRv2 
* GCC
 
INSTALLATION
------------
* Obtain Cetus distribution
  The latest version of Cetus can be obtained at:
  http://cetus.ecn.purdue.edu/

* Unpack
  Users need to unpack the distribution before installing Cetus.
  $ cd <directory_where_cetus.tar.gz_exists>
  $ gzip -d cetus.tar.gz | tar xvf -

* Build
  There are several options for building Cetus:
  - For Apache Ant users
    The provided build.xml defines the build targets for Cetus. The available
    targets are "compile", "jar", "clean" and "javadoc". Users need to edit
    the location of the Antlr tool.
  - For Linux/Unix command line users.
    Run the script build.sh after defining system-dependent variables in the
    script.
  - For SDK (Eclipse, Netbeans, etc) users
    Follow the instructions of each SDK.

RUNNING CETUS
-------------
Users can run Cetus in the following ways:

  $ java -classpath=<user_class_path> cetus.exec.Driver <options> <c codes>

The "user_class_path" should include the class paths of Antlr and Cetus.
"build.sh" and "build.xml" provides a target that generates a wrapper script
for Cetus users.

TESTING
-------
We have tested Cetus successfully using the following benchmark suites:

* SPECCPU2006
  More information about this suite available at www.spec.org

* SPECOMP2001
  More information about this suite available at www.spec.org

* NPB3.0
  More information about NAS Parallel Benchmarks at www.nas.nasa.gov/Software/NPB/

KNOWN BUGS
----------
In addition to the limited scope of the features mentioned above, Cetus 1.1
currently does not handle the following cases.

* Does not support the simultaneous usage of ANSI C and K&R C 
  function declaration formats within the same source file.
  e.g.
  void temp_func(int a, int b);
  ....
  ....
  void temp_func()
    int a;
    int b;
  {
  ....
  }

  Affected benchmarks: 456.hmmer (hsregex.c)

* Does not preserve line number information and hence fails during SPECCPU
  validation.

  Affected benchmarks: 482.sphinx3

* Does not handle parsing and IR creation for GNU GCC __asm__ extensions. Will
  be addressed in the next release.

July 10, 2009
The Cetus Team

URL: http://cetus.ecn.purdue.edu
EMAIL: cetus@ecn.purdue.edu


RELEASE
--------
Cetus 1.0 (September 20, 2008)

Cetus is a source-to-source compiler infrastructure for C written in Java.

FEATURES
--------
Cetus 1.0 has the following updated features:

* New Symbol/Symbol Table Interface
  The previous symbol table interface attached a lookup-table for each IR
  region that defines a scope, and each symbol table entry points to the
  declaration statement that defines the corresponding variable (table key).
  The disadvantages of this method are 1) the provided search method is not
  sufficient to support access to members of a variable such as struct members
  2) it allows undeclared identifiers in the IR 3) on-demand search can be
  expensive depending on the behavior of the compiler passes.

  A new symbol interface is introduced to provide an easy way to access the
  attributes of variables. The new method defines each declarator as an object
  that implements "cetus.hir.Symbol" interface and creates a link from each
  identifier to its declarator while reporting warnings for undeclared
  variables. Once this is done, access to symbol attributes can be performed in 
  one step by other compiler passes. The "Symbol" object is also useful when
  used as a key in any data structures, as it is unique to each variable in the
  program.

  The range analysis and the array privatization passes described below were
  implemented using this new symbol interface, and many utility methods in
  "cetus.hir.Tools" use this new interface.

* Symbolic Expression Manipulation
  This Cetus release provides enhanced expression manipulation tools that are
  useful in several ways. Those tools can convert a given expression to a
  simplified and normalized form and perform symbolic algebraic operations.
  The following example shows the feature of the simplification tools:

    1+4-a+2*a   -> 5+a      (folding)
    a*(b+c)     -> a*b+a*c  (distribution)
    a*b+a*c     -> a*(b+c)  (factorization)
    (8*a)/(2*b) -> (4*a)/b  (division)

  By default, the simplifier provided in "cetus.hir.NormalExpression" performs
  all the simplification methods except for factorization but pass writers can
  also customize their manipulation processes by calling each individual
  transformation in sequence. Look for more details in
  "cetus.hir.NormalExpression".

* Symbolic Range Analysis
  The symbolic range analysis performs symbolic execution of the program and
  computes a possible value range for each integer-typed variable at each
  program point. The resulting set of value ranges provide a method to compare
  the values of two expressions at compile time. This framework is based on the
  counterpart in the Polaris parallelizing compiler (predecessor of Cetus) and
  has been verified with SPEC CPU2006.
  The following example shows the set of value ranges computed at each program
  point after the range analysis.

                                      Range Domain for Procedure foo
                                              []
                                      {
                                              []
      int foo(int k)                    int i, j;
      {                                       []
        int i, j;                       double a;
        double a;                             []
                                        for ( i=0; i<10;  ++ i )
        for ( i=0; i<10; ++i )                []
        {                               {
          a = 0.5*i;                          [0<=i<=9]
        }                                 a=(0.5*i);
        j = i+k;                        }
        return j;                             [i=10]
      }                                 j=(i+k);
                                              [i=10, j=(i+k)]
                                        return j;
                                      }
  
  See more details in "cetus.analysis.RangeAnalysis" and
  "cetus.analysis.RangeDomain" for examples of how to use the result of the
  range analysis in a compiler pass.

* Array Privatization
  The array privatization pass detects privatizable scalars and arrays in each
  loop. The current approach considers variables that are written first and read
  later in every iteration, and access at the same memory location as private
  candidates. The privatizer internally incorporates live variable analysis to
  mark any "last" private variables after detecting privatizable variables.
  The current limitations of this pass are:
  - Loops containing function calls are handled conservatively; any variables
    that appear in actual parameters and global variables are not listed as
    private.
  - Variables with a user-defined type are not listed as private.

* Reduction Recognition
  The reduction pass recognizes additive and multiplicative reduction variables
  in the loop.
  Essentially, for additive reductions, the algorithm recognizes statements of
  the form x = x + expr, where expr is typically a real-valued, loop-variant
  expression.
  The reduction variable x can be a scalar or an array expression. One or several
  reduction statements may appear in a loop, however x must not appear in any
  non-reduction statement.

* Data Dependence Analysis
  The data dependence analyzer determines whether dependences exist between
  two subscripted references to the same/aliased arrays in a given loop nest.
  Array accesses within *eligible* loop nests are tested for dependences, and 
  a data dependence graph (DDG) is constructed using the resulting direction
  vectors for each loop nest.
  Loop nests considered *eligible* are:
  - Canonical loops (Conventional FORTRAN DO loop format
    with positive stride)
  - Perfect Loop Nests
  - Loops without function calls
  - Loops without control flow modifiers such as break or goto statements

  Cetus uses the Banerjee-Wolfe Inequalities for dependence testing.
  The framework handles single and multiple index variable subscripts, multi-
  dimensional array subscripts as well as coupled subscripts (conservatively).
  The dependence analyzer also interfaces with array privatization, reduction
  and alias analyses to filter/include dependences.
  More details can be found in "cetus.analysis.DDTDriver".

* Automatic Loop Parallelization
  The Loop Parallelization Pass uses data dependence information obtained from
  the data dependence analysis described above. This information is available 
  via the loop dependence graph (cetus.analysis.DDGraph) for a given loop nest.
  For every loop within a nest, the dependence graph is checked for 
  loop-carried dependences at that nest-level. If none exist, the loop is 
  annotated as parallel.
  Limitations:
  - Serialization of outer loops to enable parallelization of additional inner
    loops (e.g. when a dependence with a [<,>] direction vector is found) is
    not yet supported.

* Internal Cetus Annotation Statements
  Results of the various analyses such as privatization, reduction and loop
  parallelization are now stored internally within the IR as Cetus pragmas.
  Output source code might contain, for example, the following directives:
  
  #pragma cetus reduction(*: s) 
  #pragma cetus private(i) 
  #pragma cetus parallel
  for (i=0; i<N; i ++ )
  {
    s=((s*A[i])+(s*B[i]));
  }

* OpenMP Code Generation
  The OpenMP directive generation pass derives parallelization information by
  parsing internal Cetus annotation statements mentioned in the previous section.
  This information is used to generate OpenMP directives for *For loops* that 
  have been annotated as parallel by the loop parallelization pass.
  Privatization and reduction clauses are appended to openMP directives based 
  on the Cetus private and reduction annotations.

CONTENTS
--------
This Cetus release has the following contents.

  src       - Source codes of Cetus
  lib       - Archived classes (jar)
  api       - JAVA documents 
  build.sh  - Command line build script
  build.xml - Build configuration for Apache Ant
  readme    - this file
  license   - Cetus license

REQUIREMENTS
------------
* JAVA 2 SDK, SE 1.5.x (or later)
* ANTLRv2 
* GCC
 
INSTALLATION
------------
* Obtain Cetus distribution
  The latest version of Cetus can be obtained at:
  http://cetus.ecn.purdue.edu/

* Unpack
  Users need to unpack the distribution before installing Cetus.
  $ cd <directory_where_cetus.tar.gz_exists>
  $ gzip -d cetus.tar.gz | tar xvf -

* Build
  The organization of the directories has changed since the last release and
  there are several options for building Cetus:
  - For Apache Ant users
    The provided build.xml defines the build targets for Cetus. The available
    targets are "compile", "jar", "clean" and "javadoc". Users need to edit
    the location of the Antlr tool.
  - For Linux/Unix command line users.
    Run the script build.sh after defining system-dependent variables in the
    script.
  - For SDK (Eclipse, Netbeans, etc) users
    Follow the instructions of each SDK.

RUNNING CETUS
-------------
Users can run Cetus in the following ways:

  $ java -classpath=<user_class_path> cetus.exec.Driver <options> <c codes>

The "user_class_path" should include the class paths of Antlr and Cetus.
"build.sh" and "build.xml" provides a target that generates a wrapper script
for Cetus users.

TESTING
-------
We have tested Cetus successfully using the following benchmark suites:

* SPECCPU2006
  More information about this suite available at www.spec.org

* SPECOMP2001
  More information about this suite available at www.spec.org

* NPB3.0
  Only tested with one existing C benchmark: IS (Integer Sort)
  More information about NAS Parallel Benchmarks at www.nas.nasa.gov/Software/NPB/

LIMITATIONS
-----------
In addition to the limited scope of the features mentioned above, Cetus 1.0
currently does not handle the following cases.

* Does not support the simultaneous usage of ANSI C and K&R C 
  function declaration formats within the same source file.
  e.g.
  void temp_func(int a, int b);
  ....
  ....
  void temp_func()
    int a;
    int b;
  {
  ....
  }

  Affected benchmarks: 456.hmmer (hsregex.c)

* Does not preserve line number information and hence fails during SPECCPU
  validation.

  Affected benchmarks: 482.sphinx3

* Parallelization passes were tested with the C codes in SPECOMP2001 and NPB3.


September 20, 2008
The Cetus Team

URL: http://cetus.ecn.purdue.edu
EMAIL: cetus@ecn.purdue.edu

-------------------------------------------------------------------------------

Cetus 0.5.1 (November 6, 2007)

Cetus is a source-to-source compiler infrastructure written in Java. We
currently support ANSI C transformation and support for C++ is currently under
development.


FEATURES
---------
Cetus 0.5.1 has the following updated features:

For Users:
* New and improved website interface for our users
* Web-based feedback and bug reporting via Bugzilla
* Cetus mailing lists for better exchange of 
  Cetus-related information

Source-related:
* Cetus is now being regression tested using the 
  SPEC2006 CPU benchmarks and a few related bug-fixes
  have been incorporated in this release.
* Support for C99 datatypes _Bool, _Complex, and
  _Imaginary
* Current interface to symbol table is under revision and
  will be improved for the next release, while still
  supporting the old interface


REQUIREMENTS
-------------
* Java 2 SDK, SE 1.5.0 (or later)    (Required)
* ANTLRv2 2.7.5        (or later)    (Required)
* Bash                               (Required)


INSTALLATION
-------------
* Obtain Cetus distribution
  The latest version of Cetus can be obtained at:
  http://cetus.ecn.purdue.edu/

* Unpack
  Users need to unpack the distribution before installing Cetus.
  $ cd <directory_where_cetus.tar.gz_exists>
  $ gzip -d cetus.tar.gz | tar xvf -

* Build
  The Makefile in the "cetus" directory controls the compilation of Cetus.
  It is important to let Makefile know the location of ANTLR runtime binary.
  $ cd cetus
  $ make antlr=<directory_where_antlr.jar_exists>


LIMITATIONS
------------
The following bugs are currently under review and have not been
fixed for release 0.5.1. The affected SPEC2006 CPU benchmarks
are mentioned for reference.

* Currently, Cetus does not support the passing of variable
  types as parameters to a function call.
  e.g.
  va_arg(temp, int);
  
  The above function is a built-in GCC function that accepts
  variable types as parameters.

  Affected benchmarks: 400.perlbench
                       445.gobmk
                       462.libquantum

* Currently, Cetus does not support the C99 operator
  __alignof__ which accepts a variable type or an expression
  as its operand.

  Affected benchmarks: 403.gcc

* Currently, Cetus does not support the simultaneous usage of 
  ANSI C and K&R C function declaration formats within the 
  same source file.
  e.g.
  void temp_func(int a, int b);
  ....
  ....
  void temp_func()
    int a;
    int b;
  {
  ....
  }

  Affected benchmarks: 456.hmmer (hsregex.c)


November 6, 2007
The Cetus Team

URL: http://cetus.ecn.purdue.edu/
EMAIL: cetus@ecn.purdue.edu
