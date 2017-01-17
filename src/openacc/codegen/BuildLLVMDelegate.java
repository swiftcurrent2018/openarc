package openacc.codegen;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import cetus.codegen.CodeGenPass;
import cetus.hir.Program;

/**
 * {@link BuildLLVM} super class and factory that can be used whether or not
 * {@link BuildLLVM} is compiled.
 * 
 * <p>
 * {@link BuildLLVM} is compiled if and only if OpenARC is built with LLVM
 * support enabled.  All code that needs to use {@link BuildLLVM} should use
 * {@link BuildLLVMDelegate} instead so that such code can be compiled whether
 * or not {@link BuildLLVM} is compiled.
 * </p>
 * 
 * <p>
 * To construct an instance of {@link BuildLLVM}, call the static method
 * {@link BuildLLVMDelegate#make}, which throws a
 * {@link BuildLLVMDisabledException} if {@link BuildLLVM} has not been
 * compiled.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public abstract class BuildLLVMDelegate extends CodeGenPass {
  private static final String BUILD_LLVM_CLASS
    = BuildLLVMDelegate.class.getPackage().getName()
      + ".llvmBackend.BuildLLVM";

  /**
   * Thrown by {@link #make} if {@link BuildLLVM} has not been compiled.
   */
  public static class BuildLLVMDisabledException extends Exception {
    private static final long serialVersionUID = 1L;
  }

  private static Constructor<BuildLLVMDelegate> ctor = null;

  /**
   * Factory method for {@link BuildLLVM}.
   * 
   * @param llvmTargetTriple
   *          the target triple string for LLVM
   * @param llvmTargetDataLayout
   *          the target data layout string for LLVM
   * @param program
   *          the program for which to build LLVM IR
   * @param debugOutput
   *          whether to produce debug output
   * @param reportLine
   *          whether to include statement line numbers in error messages.
   *          This is an experimental feature, so the line numbers might be
   *          wrong. Moreover, expected error messages in the test suite
   *          were originally written before this feature was implemented,
   *          and we don't want to take the time to update them.
   * @param warningsAsErrors
   *          whether to report warnings as errors
   * @param enableFaultInjection
   *          whether to enable FITL or ignore OpenARC fault-injection
   *          directives
   * @return an instance of {@link BuildLLVM}, ready for a call to
   *         {@link #start}
   * @throws BuildLLVMDisabledException
   *           if {@link BuildLLVM} was not compiled
   */
  public static BuildLLVMDelegate make(
    String llvmTargetTriple, String llvmTargetDataLayout, Program program,
    boolean debugOutput, boolean reportLine, boolean warningsAsErrors,
    boolean enableFaultInjection) throws BuildLLVMDisabledException
  {
    if (ctor == null) {
      Class <?> clazz;
      try {
        clazz = ClassLoader.getSystemClassLoader()
                .loadClass(BUILD_LLVM_CLASS);
      }
      catch (ClassNotFoundException e) {
        throw new BuildLLVMDisabledException();
      }
      if (!BuildLLVMDelegate.class.isAssignableFrom(clazz)) {
        throw new IllegalStateException(
          BUILD_LLVM_CLASS+" is not a subclass of "
          +BuildLLVMDelegate.class.getName());
      }
      @SuppressWarnings("unchecked")
      Class<BuildLLVMDelegate> clazz1 = (Class<BuildLLVMDelegate>)clazz;
      try {
        ctor = clazz1.getDeclaredConstructor(
          String.class, String.class, Program.class, boolean.class,
          boolean.class, boolean.class, boolean.class);
      }
      catch (NoSuchMethodException | SecurityException e) {
        throw new IllegalStateException(
          BUILD_LLVM_CLASS+" does not have the expected constructor", e);
      }
      System.loadLibrary("jllvm");
    }
    try {
      return ctor.newInstance(llvmTargetTriple, llvmTargetDataLayout,
                              program, debugOutput, reportLine,
                              warningsAsErrors, enableFaultInjection);
    }
    catch (InstantiationException | IllegalAccessException
           | IllegalArgumentException | InvocationTargetException e)
    {
      throw new IllegalStateException(
        BUILD_LLVM_CLASS+" constructor failed", e);
    }
  }

  protected BuildLLVMDelegate(Program program) {
    super(program);
  }

  @Override
  public String getPassName() {
    return new String("[BuildLLVM]");
  }

  /**
   * Write LLVM bitcode to files, one {@code .bc} file per input translation
   * unit, in directory {@code outdir}.
   */
  public abstract void printLLVM(String outdir);

  /**
   * Dump LLVM IR for all modules to stderr.
   */
  public abstract void dumpLLVM();
}
