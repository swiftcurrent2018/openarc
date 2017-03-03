package openacc.transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import openacc.hir.OpenMPLibrary;
import cetus.analysis.InlineExpansion;
import cetus.exec.Driver;
import cetus.hir.IRTools;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.Specifier;
import cetus.hir.StandardLibrary;
import cetus.transforms.TransformPass;

/**
 * Transforms a program by performing simple subroutine in-line expansion in its main function.
 */

public class InlineFunctionTransformation extends TransformPass {
	
	/** Name of the inline expansion pass */
	private static final String NAME = "[InlineFunctionTransformation]";
	private static final String MODE = "mode";
	private static final String DEPTH = "depth";
	private static final String PRAGMA = "pragma";
	private static final String DEBUG = "debug";
	private static final String FORONLY = "foronly";
	private static final String FUNCTIONS = "functions";
	private static final String COMPLEMENT = "complement";
	
	/**
	 * Constructs an inline expansion pass 
	 * @param program - the program to perform inline expansion on
	 */
	public InlineFunctionTransformation(Program program) {
		super(program);
	}
	
	@Override
	public void start() {
		String options = Driver.getOptionValue("inlineFunctionTransformation");
		InlineExpansion inlineExpansion = new InlineExpansion();
		inlineExpansion.setMode(2);
		inlineExpansion.setHonorPragmas(false);
		inlineExpansion.setLevel_1(false);
		inlineExpansion.setComplementFunctions(false);
		StringTokenizer tokenizer = new StringTokenizer(options, ":");
		String option;
		while(tokenizer.hasMoreTokens()) {
			option = tokenizer.nextToken();
			int eqIndex = option.indexOf('='); 
			if( eqIndex != -1) {
				String opt = option.substring(0, eqIndex).trim();
				try {
					int value = new Integer(option.substring(eqIndex+1).trim()).intValue();
					if(opt.equals(DEBUG)) {
						inlineExpansion.setDebugOption(value == 1? true : false);
					}
					else if(opt.equals(FORONLY)) {
						inlineExpansion.setInsideForOnly(value == 1? true : false);
					}
					else if(opt.equals(COMPLEMENT)) {
					}
				}
				catch(NumberFormatException ex){
				}
			}
		}
		ArrayList<String> inlinefunctions = new ArrayList<String>();
		List<Procedure> procList = IRTools.getProcedureList(program);
		for( Procedure tProc : procList ) {
			if( tProc.getTypeSpecifiers().contains(Specifier.INLINE) ) {
				String fName = tProc.getSymbolName();
				if( !fName.startsWith("_") && !StandardLibrary.contains(fName) &&
						!OpenMPLibrary.contains(fName) ) { //exclude intrinsic functions and standard libraries.
					inlinefunctions.add(fName);
				}
			}
		}
		PrintTools.println("User functions to be inlined : " + PrintTools.collectionToString(inlinefunctions, ", "), 0);
		inlineExpansion.setCommandlineFunctions(inlinefunctions);

		inlineExpansion.inline(program);
	}
	
	@Override
	public String getPassName() {
		return NAME;
	}
}
