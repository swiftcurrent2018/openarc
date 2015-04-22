/**
 * 
 */
package openacc.transforms;

import java.util.*;

import cetus.hir.*;
import cetus.analysis.LoopTools;
import cetus.transforms.TransformPass;
import openacc.hir.*;
import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;

/**
 * <b>CollapseTransformation</b> performs source-level transformation to collapse
 * the iterations of all associated loops into one larger iteration space.
 * 
 * @author Seyong Lee <lees2@ornl.gov
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class CollapseTransformation extends TransformPass {

	/**
	 * @param program
	 */
	public CollapseTransformation(Program program) {
		super(program);
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return new String("[collapse transformation]");
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#start()
	 */
	@Override
	public void start() {
		List<ForLoop> outer_loops = new ArrayList<ForLoop>();
		// Find loops containing OpenACC collapse clauses with parameter value > 1.
		List<ACCAnnotation> collapseAnnotList = IRTools.collectPragmas(program, ACCAnnotation.class, "collapse");
		if( collapseAnnotList == null ) return;
		for( ACCAnnotation cAnnot : collapseAnnotList ) {
			Annotatable at = cAnnot.getAnnotatable();
			if( at instanceof ForLoop ) {
				Object cObj = cAnnot.get("collapse");
				if( !(cObj instanceof IntegerLiteral) ) {
					Tools.exit("[ERROR] the argument of OpenACC collapse clause must be a constant positive integer expression"
							+ ", but the following loop construct has an incompatible argument:\n" + at.toString() + "\n");
				}
				int collapseLevel = (int)((IntegerLiteral)cObj).getValue();
				if( collapseLevel > 1 ) {
					outer_loops.add((ForLoop)at);
				}
			}
		}
		if( outer_loops.isEmpty() ) {
			return;
		} else {
			PrintTools.println("[INFO] Found OpenACC-collapse clauses that associate more than one loop.",1);
			int collapsedLoops = 0;
			for( ForLoop accLoop : outer_loops ) {
				collapsedLoops +=collapseLoop(accLoop);
			}
			PrintTools.println("[INFO] Number of collapsed OpenACC loops: " + collapsedLoops, 0);
		}
	}
	
	public static int collapseLoop(ForLoop accLoop) {
		int collapsedLoops = 0;
		Traversable t = (Traversable)accLoop;
		while(true) {
			if (t instanceof Procedure) break;
			t = t.getParent(); 
		}
		Procedure proc = (Procedure)t;
		Statement enclosingCompRegion = null;
		if( !accLoop.containsAnnotation(ACCAnnotation.class, "kernels") && 
				!accLoop.containsAnnotation(ACCAnnotation.class, "parallel")) {
			Annotatable att = (Annotatable)accLoop.getParent();
			while( att != null ) {
				if( att.containsAnnotation(ACCAnnotation.class, "kernels" ) ||
						att.containsAnnotation(ACCAnnotation.class, "parallel")) {
					enclosingCompRegion = (Statement)att;
					break;
				} else {
					att = (Annotatable)att.getParent();
				}
			}
		}
		ArrayList<Symbol> indexSymbols = new ArrayList<Symbol>();
		ArrayList<ForLoop> indexedLoops = new ArrayList<ForLoop>();
		indexedLoops.add(accLoop);
		ACCAnnotation collapseAnnot = accLoop.getAnnotation(ACCAnnotation.class, "collapse");
		int collapseLevel = (int)((IntegerLiteral)collapseAnnot.get("collapse")).getValue();
		boolean pnest = true;
		pnest = AnalysisTools.extendedPerfectlyNestedLoopChecking(accLoop, collapseLevel, indexedLoops, null);
		if( !pnest ) {
			Tools.exit("[ERROR] OpenACC collapse clause is applicable only to perfectly nested loops;\n"
					+ "Procedure name: " + proc.getSymbolName() + "\nTarget loop: \n" +
					accLoop.toString() + "\n");
		}
		if( indexedLoops.size() < collapseLevel ) {
			PrintTools.println("\n[WARNING] Number of found loops (" + indexedLoops.size() + 
					") is smaller then collapse parameter (" + collapseLevel + "); skip the loop in procedure " 
					+ proc.getSymbolName() + ".\n",0);
			PrintTools.println("OpenACC loop\n" + accLoop + "\n", 2);
			return collapsedLoops;
		}
		for( ForLoop currLoop : indexedLoops ) {
			indexSymbols.add(LoopTools.getLoopIndexSymbol(currLoop));
		}
		collapsedLoops++;
		ForLoop innerLoop = indexedLoops.get(collapseLevel-1);
		Statement fBody = innerLoop.getBody();
		CompoundStatement cStmt = null;
		Statement firstNonDeclStmt = null;
		if( fBody instanceof CompoundStatement ) {
			cStmt = (CompoundStatement)fBody;
			firstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(fBody);
		} else {
			cStmt = new CompoundStatement();
			cStmt.addStatement(fBody);
			firstNonDeclStmt = fBody;
		}
		ArrayList<Expression> iterspaceList = new ArrayList<Expression>();
		ArrayList<Expression> lbList = new ArrayList<Expression>();
		Expression collapsedIterSpace = null;
		for( int i=0; i<collapseLevel; i++ ) {
			ForLoop loop = indexedLoops.get(i);
			Expression lb = LoopTools.getLowerBoundExpression(loop);
			lbList.add(i, lb);
			Expression ub = LoopTools.getUpperBoundExpression(loop);
			Expression itrSpace = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
			iterspaceList.add(i, itrSpace);
			if( i==0 ) {
				collapsedIterSpace = itrSpace;
			} else {
				collapsedIterSpace = Symbolic.multiply(collapsedIterSpace, itrSpace);
			}
		}
		//Create a new index variable for the newly collapsed loop.
		CompoundStatement procBody = proc.getBody();

		Identifier newIndex = null;
		if( enclosingCompRegion instanceof CompoundStatement ) {
			newIndex = TransformTools.getNewTempIndex(enclosingCompRegion);
		} else if( enclosingCompRegion instanceof ForLoop ) {
			newIndex = TransformTools.getNewTempIndex(((ForLoop)enclosingCompRegion).getBody());
		} else {
			newIndex = TransformTools.getNewTempIndex(procBody);
			//If the current accLoop is in a compute region, the above new private variable should be
			//added to the private clause.
			Set<SubArray> privateSet = null;
			if( collapseAnnot.containsKey("private") ) {
				privateSet = collapseAnnot.get("private");
			} else {
				privateSet = new HashSet<SubArray>();
				collapseAnnot.put("private", privateSet);
			}
			privateSet.add(AnalysisTools.createSubArray(newIndex.getSymbol(), true, null));
		}
		if( !collapseAnnot.containsKey("gang") && !collapseAnnot.containsKey("worker") &&
				!collapseAnnot.containsKey("vector") && !collapseAnnot.containsKey("seq") ) {
			collapseAnnot.put("seq", "true");
		}
		/////////////////////////////////////////////////////////////////////////////////
		//Swap initialization statement, condition, and step of the OpenACC loop with  //
		//those of the new, collapsed loop.                                            //
		/////////////////////////////////////////////////////////////////////////////////
		Expression expr1 = new AssignmentExpression(newIndex.clone(), AssignmentOperator.NORMAL,
				new IntegerLiteral(0));
		Statement initStmt = new ExpressionStatement(expr1);
		accLoop.getInitialStatement().swapWith(initStmt);
		expr1 = new BinaryExpression((Identifier)newIndex.clone(), BinaryOperator.COMPARE_LT,
				collapsedIterSpace);
		accLoop.getCondition().swapWith(expr1);
		expr1 = new UnaryExpression(
				UnaryOperator.POST_INCREMENT, (Identifier)newIndex.clone());
		accLoop.getStep().swapWith(expr1);
		Identifier accforIndex = new Identifier(indexSymbols.get(0));
		int i = collapseLevel-1;
		while( i>0 ) {
			Identifier tIndex = new Identifier(indexSymbols.get(i));
			if( i == (collapseLevel-1) ) {
				expr1 = new BinaryExpression(newIndex.clone(), BinaryOperator.MODULUS, 
						iterspaceList.get(i).clone());
			} else {
				expr1 = new BinaryExpression(accforIndex.clone(), BinaryOperator.MODULUS, 
						iterspaceList.get(i).clone());
			}
			Expression lbExp = lbList.get(i).clone();
			if( !(lbExp instanceof Literal) || !lbExp.toString().equals("0") ) {
				expr1 = new BinaryExpression(expr1, BinaryOperator.ADD, lbExp);
			}
			Statement stmt = new ExpressionStatement(new AssignmentExpression(tIndex.clone(), 
					AssignmentOperator.NORMAL, expr1));
			cStmt.addStatementBefore(firstNonDeclStmt, stmt);
			if( i == (collapseLevel-1) ) {
				expr1 = new BinaryExpression(newIndex.clone(), BinaryOperator.DIVIDE, 
						iterspaceList.get(i).clone());
			} else {
				expr1 = new BinaryExpression(accforIndex.clone(), BinaryOperator.DIVIDE, 
						iterspaceList.get(i).clone());
			}
			if( i == 1 ) {
				lbExp = lbList.get(0).clone();
				if( !(lbExp instanceof Literal) || !lbExp.toString().equals("0") ) {
					expr1 = new BinaryExpression(expr1, BinaryOperator.ADD, lbExp);
				}
			}
			stmt = new ExpressionStatement(new AssignmentExpression(accforIndex.clone(), 
					AssignmentOperator.NORMAL, expr1));
			cStmt.addStatementBefore(firstNonDeclStmt, stmt);
			i--;
		}
		/////////////////////////////////////////////////////////////////////////
		//Swap the body of the OpenACC loop with the one of the innermost loop //
		//among associated loops.                                              //
		/////////////////////////////////////////////////////////////////////////
		accLoop.getBody().swapWith(cStmt);

		return collapsedLoops;
	}

}
