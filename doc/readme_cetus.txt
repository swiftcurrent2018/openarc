#######################################################################
## This file contains information on the underlying Cetus framework, ##
## some of which may be outdated.                                    ##
#######################################################################
-------------------------------------------------------------------------------
RELEASE
-------------------------------------------------------------------------------
Cetus 1.3.1 (June 5, 2012)

Cetus is a source-to-source compiler infrastructure for C written in Java, and
can be downloaded from http://cetus.ecn.purdue.edu. This version contains
minor updates and fixes in the existing passes.

-------------------------------------------------------------------------------
FEATURES/UPDATES
-------------------------------------------------------------------------------
* New features
  - Added partial support for c99 features including mixed declarations and
    code, new block scope for iteration statements, designated initializers,
    compound literals, hexadecimal floating constants, and restrict pointers.
    Visit the link below to find out more details about these c99 features:
    http://cetus.ecn.purdue.edu/FAQ/unsupported_c99_features.html
  - Added program slicing tools in the cetus.application package, which is based
    on the def-use/use-def chain tools. This package is still experimental.

* Bug fixes and improvements
  - Enhanced the accuracy of the data dependence graph by excluding unreachable
    dependence pair under the "equal" direction vector.
  - Fixed a bug in the data dependence test which was not doing the merge
    operation of direction vectors in certain types of multiple subscripts.
  - Fixed a bug in the parser which was not handling certain types of struct
    declaration with bit fields.
  - Fixed a bug in the induction variable substitution pass which was failing
    to generate runtime test for possible zero-trip loops in some cases.
  - Fixed a bug in IR consistency checker which was not strong enough to check
    a certain type of garbled IR.
  - Fixed a bug in the branch eliminator which was not removing code section
    in certain types of "if" statements.
  - Fixed a bug in the IRTools.isReachable() method.

* Updates in flags
  - Modified flags
    -induction=N
        Added sub options which processes only linear induction variables.
    -privatize=N
        Added sub options which processes only scalar variables.

-------------------------------------------------------------------------------
CONTENTS
-------------------------------------------------------------------------------
This Cetus release has the following contents.

  lib            - Archived classes (jar)
  license.txt    - Cetus license
  build.sh       - Command line build script
  build.xml      - Build configuration for Apache Ant
  src            - Cetus source code
  readme.txt     - This file
  readme_log.txt - Archived release notes

-------------------------------------------------------------------------------
REQUIREMENTS
-------------------------------------------------------------------------------
* JAVA SE 6
* ANTLRv2 
* GCC
 
-------------------------------------------------------------------------------
INSTALLATION
-------------------------------------------------------------------------------
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
  - For Linux/Unix command line users
    Run the script build.sh after defining system-dependent variables in the
    script.
  - For SDK (Eclipse, Netbeans, etc) users
    First, build the parser with the Antlr tool.
    Then, follow the instructions of each SDK to set up a project.

-------------------------------------------------------------------------------
RUNNING CETUS
-------------------------------------------------------------------------------
Users can run Cetus in the following way:

  $ java -classpath=<user_class_path> cetus.exec.Driver <options> <C files>

The "user_class_path" should include the class paths of Antlr and Cetus.
"build.sh" and "build.xml" provides a target that generates a wrapper script
for Cetus users.

-------------------------------------------------------------------------------
TESTING
-------------------------------------------------------------------------------
We have tested Cetus successfully using the following benchmark suites:

* SPEC CPU2006
  More information about this suite is available at http://www.spec.org

* SPEC OMP2001
  More information about this suite is available at http://www.spec.org

* NPB 2.3 written in C
  More information about this suite is available at
  http://www.hpcs.cs.tsukuba.ac.jp/omni-openmp/

June 5, 2012
The Cetus Team

URL: http://cetus.ecn.purdue.edu
EMAIL: cetus@ecn.purdue.edu
