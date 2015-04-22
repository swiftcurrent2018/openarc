/**
 * 
 */
package openacc.transforms;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import openacc.analysis.ACCAnalysis;
import openacc.analysis.AnalysisTools;
import openacc.hir.ACCAnnotation;
import cetus.analysis.LoopTools;
import cetus.hir.Annotatable;
import cetus.hir.AssignmentExpression;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.Program;
import cetus.hir.Statement;
import cetus.transforms.LoopNormalization;
import cetus.transforms.TransformPass;

/**
 * Normalize OpenACC gang/worker/vector loops.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class ACCWorkshareLoopNormalization extends TransformPass {
	private static String pass_name = "[ACCWorkshareLoopNormalization]";

	/**
	 * @param program
	 */
	public ACCWorkshareLoopNormalization(Program program) {
		super(program);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return pass_name;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#start()
	 */
	@Override
	public void start() {
		// TODO Auto-generated method stub
		List<ACCAnnotation> workshareLoops = 
			AnalysisTools.collectPragmas(program, ACCAnnotation.class, ACCAnnotation.worksharingClauses, false);
		if( workshareLoops != null ) {
			for( ACCAnnotation annot : workshareLoops ) {
				Annotatable at = annot.getAnnotatable();
				if( at instanceof ForLoop ) {
					ForLoop fLoop = (ForLoop)at;
					if (LoopTools.isCanonical(fLoop)) {
						Expression origIndexVar = LoopTools.getIndexVariable(fLoop);
						LoopNormalization.normalizeLoop(fLoop);
						Expression newIndexVar = LoopTools.getIndexVariable(fLoop);
						//Replace the original index symbol with the new one if existing in OpenACC annotations.
						Map<String, String> nameChangeMap = new HashMap<String, String>();
						nameChangeMap.put(origIndexVar.toString(), newIndexVar.toString());
						ACCAnalysis.updateSymbolsInACCAnnotations(fLoop, nameChangeMap);
						//[DEBUG] LoopNormalization.normalizeLoop() adds a statement to update the last value
						//of the original loop-index variable, but this is not needed and incorrect in 
						//OpenACC worksharing loop translation. Therefore, remove the last-value assignment
						//statement.
						CompoundStatement cStmt = (CompoundStatement)fLoop.getParent();
						if( cStmt != null ) {
							Statement nStmt = AnalysisTools.getStatementAfter(cStmt, fLoop);
							if( (nStmt != null) && (nStmt instanceof ExpressionStatement) ) {
								Expression nExp = ((ExpressionStatement)nStmt).getExpression();
								if( nExp instanceof AssignmentExpression ) {
									if( origIndexVar.equals(((AssignmentExpression)nExp).getLHS()) ) {
										cStmt.removeStatement(nStmt);
									}
								}
							}
						}
					}
				}
			}
		}

	}

}
