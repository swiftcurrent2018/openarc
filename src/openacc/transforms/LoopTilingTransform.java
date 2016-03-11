package openacc.transforms;

import cetus.analysis.LoopTools;
import cetus.hir.*;
import cetus.transforms.LoopNormalization;
import cetus.transforms.TransformPass;
import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;
import openacc.hir.ACCAnnotation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Arrays;
import java.util.Set;

/**
 * Created by Putt on 12/3/13.
 * Fixed by Hoshino on 19/9/14
 */
public class LoopTilingTransform extends TransformPass
{
	public LoopTilingTransform(Program program)
	{
		super(program);
	}

	@Override
	public String getPassName()
	{
		return "[LoopTilingTransform]";
	}


	
	//@Override
	public void start_1D()
	{

		//////////////////////////////////////////////////////////////////////
		// this program transform a tile annotated loop like following      //
		//                                                                  //
		// #pragma acc kernels loop tile(64,4) gang worker                  //
		// for(int i = 0;i < 100;i++){                                      //
		//    for(int j = 0;j < 100;j++){                                   //
		//    }                                                             //
		// }                                                                //
		//                                                                  //
		// #pragma acc kernels loop gang                                    //
		// for(int i = 0;i < (100*100-1)/(64*4)+1;i++){                     //
		// #pragma acc loop worker                                          // 
		//    for(int ii = i*(64*4);ii < MIN(10000, i*(64*4)+(64*4));ii++){ //
		//    }                                                             //
		// }                                                                //
		//                                                                  //
		//////////////////////////////////////////////////////////////////////

		List<Procedure> procedureList = IRTools.getProcedureList(program);
		List<ForLoop> outer_loops = new ArrayList<ForLoop>();
		for( Procedure proc : procedureList )
		{
			List<ACCAnnotation> annotationList =
					AnalysisTools.collectPragmas(proc, ACCAnnotation.class);

			for(ACCAnnotation annot : annotationList)
			{
				if(!annot.containsKey("tile"))
					continue;

				Object tileClause = annot.get("tile");
				List<Expression> exprList = (List<Expression>)tileClause;

				List<Loop> loopList = LoopTools.calculateInnerLoopNest((Loop)annot.getAnnotatable());
				if(loopList.size() < exprList.size()) 
				{
					PrintTools.println("[Warning] Invalid tile clause " + annot + ".\n Skipped", 0);
					continue;
				}

				int collapseLevel = exprList.size();
				if(collapseLevel > 1){
					outer_loops.add((ForLoop)annot.getAnnotatable());
				}

			}
		}

		if(outer_loops.isEmpty()) {
			return;
		}else {
			int tiledLoops = 0;
			for( ForLoop accLoop : outer_loops ){
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
							if( att.getParent() instanceof Annotatable ) {
								att = (Annotatable)att.getParent();
							} else {
								break;
							}
						}
					}
				}
				ArrayList<Symbol> indexSymbols = new ArrayList<Symbol>();
				ArrayList<ForLoop> indexedLoops = new ArrayList<ForLoop>();
				indexedLoops.add(accLoop);
				ACCAnnotation tileAnnot = accLoop.getAnnotation(ACCAnnotation.class, "tile");
				Object tileClause = tileAnnot.get("tile");
				List<Expression> exprList = (List<Expression>)tileClause;
				int collapseLevel = exprList.size();
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
					continue;
				}
				for( ForLoop currLoop : indexedLoops ) {
					indexSymbols.add(LoopTools.getLoopIndexSymbol(currLoop));
				}
				tiledLoops++;
				ForLoop innerLoop = indexedLoops.get(collapseLevel-1);
				ACCAnnotation newPragma = new ACCAnnotation();
				innerLoop.annotate(newPragma);
				CompoundStatement cStmt = null;
				Statement firstNonDeclStmt = null;
				if( innerLoop.getBody() instanceof CompoundStatement ) {
					cStmt = (CompoundStatement)innerLoop.getBody();
					firstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(innerLoop.getBody());
				} else {
					cStmt = new CompoundStatement();
					firstNonDeclStmt = innerLoop.getBody();
				}
				ArrayList<Expression> iterspaceList = new ArrayList<Expression>();
				ArrayList<Expression> lbList = new ArrayList<Expression>();
				Expression collapsedIterSpace = null;
				ArrayList<Expression> elementIterspaceList = new ArrayList<Expression>();
				Expression elementCollapsedIterSpace = null;
				for( int i=0; i<collapseLevel; i++ ) {
					ForLoop loop = indexedLoops.get(i);
					Expression lb = LoopTools.getLowerBoundExpression(loop);
					lbList.add(i, lb);
					Expression ub = LoopTools.getUpperBoundExpression(loop);
					Expression itrSpace = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
					iterspaceList.add(i, itrSpace);
					Expression elementItrSpace = (IntegerLiteral)exprList.get(i); 
					elementIterspaceList.add(i, elementItrSpace);
					if( i==0 ) {
						collapsedIterSpace = itrSpace;
						elementCollapsedIterSpace = elementItrSpace;
					} else {
						collapsedIterSpace = Symbolic.multiply(collapsedIterSpace, itrSpace);
						elementCollapsedIterSpace = Symbolic.multiply(elementCollapsedIterSpace, elementItrSpace);
					}
				}

				Identifier newIndex = null;
				Identifier newElementIndex = null;
				if( enclosingCompRegion instanceof CompoundStatement ) {
					newIndex = TransformTools.getNewTempIndex(enclosingCompRegion);
					newElementIndex = TransformTools.getNewTempIndex(enclosingCompRegion);
				} else if( enclosingCompRegion instanceof ForLoop ) {
					newIndex = TransformTools.getNewTempIndex(((ForLoop)enclosingCompRegion).getBody());
					newElementIndex = TransformTools.getNewTempIndex(((ForLoop)enclosingCompRegion).getBody());
				} else {
					//Create a new index variable for the newly collapsed loop.
					CompoundStatement procBody = proc.getBody();
					newIndex = TransformTools.getNewTempIndex(procBody);
					newElementIndex = TransformTools.getNewTempIndex(procBody);
					//If the current accLoop is in a compute region, the above new private variable should be
					//added to the private clause.
					Set<SubArray> privateSet = null;
					if( tileAnnot.containsKey("private") ) {
						privateSet = tileAnnot.get("private");
					} else {
						privateSet = new HashSet<SubArray>();
						tileAnnot.put("private", privateSet);
					}
					privateSet.add(AnalysisTools.createSubArray(newIndex.getSymbol(), true, null));
					privateSet.add(AnalysisTools.createSubArray(newElementIndex.getSymbol(), true, null));
				}
				if( !tileAnnot.containsKey("gang") && !tileAnnot.containsKey("worker") &&
						!tileAnnot.containsKey("vector") && !tileAnnot.containsKey("seq") ) {
					tileAnnot.put("seq", "true");
				}
				if( tileAnnot.containsKey("worker")){
					Object worker = tileAnnot.get("worker");
					newPragma.put("loop", "true");
					newPragma.put("worker", worker);
					tileAnnot.remove("worker");
				}

				/////////////////////////////////////////////////////////////////////////////////
				//Swap initialization statement, condition, and step of the OpenACC loop with  //
				//those of the new, collapsed and normalized loop.                             //
				/////////////////////////////////////////////////////////////////////////////////

				//for original loop 
				Expression expr1 = new AssignmentExpression(newIndex.clone(), AssignmentOperator.NORMAL,
						new IntegerLiteral(0));
				Statement initStmt = new ExpressionStatement(expr1);
				accLoop.getInitialStatement().swapWith(initStmt);
				expr1 = new BinaryExpression((Identifier)newIndex.clone(), BinaryOperator.COMPARE_LT,
						new BinaryExpression( 
								new BinaryExpression (
										new BinaryExpression (collapsedIterSpace, BinaryOperator.SUBTRACT, new IntegerLiteral(1)), 
										BinaryOperator.DIVIDE, elementCollapsedIterSpace.clone())
								, BinaryOperator.ADD, new IntegerLiteral(1)));
				accLoop.getCondition().swapWith(expr1);
				expr1 = new UnaryExpression(
						UnaryOperator.POST_INCREMENT, (Identifier)newIndex.clone());
				//expr1 = new BinaryExpression((Identifier)newIndex.clone(), BinaryOperator.ADD, elementCollapsedIterSpace.clone());
				accLoop.getStep().swapWith(expr1);

				//for element loop 
				expr1 = new AssignmentExpression(newElementIndex.clone(), AssignmentOperator.NORMAL,
						new BinaryExpression(newIndex.clone(), BinaryOperator.MULTIPLY, elementCollapsedIterSpace.clone()));
				Statement initElementStmt = new ExpressionStatement(expr1);
				innerLoop.getInitialStatement().swapWith(initElementStmt);
				expr1 = new BinaryExpression((Identifier)newElementIndex.clone(), BinaryOperator.COMPARE_LT,
						new FunctionCall(new NameID("MIN"), collapsedIterSpace.clone(),
								new BinaryExpression(elementCollapsedIterSpace.clone(), BinaryOperator.ADD, 
										new BinaryExpression((Identifier)newIndex.clone(), BinaryOperator.MULTIPLY, elementCollapsedIterSpace.clone()))));
				innerLoop.getCondition().swapWith(expr1);
				expr1 = new UnaryExpression(
						UnaryOperator.POST_INCREMENT, (Identifier)newElementIndex.clone());
				innerLoop.getStep().swapWith(expr1);


				Identifier accforIndex = new Identifier(indexSymbols.get(0));
				int i = collapseLevel-1;
				while( i>0 ) {
					Identifier tIndex = new Identifier(indexSymbols.get(i));
					expr1 = new BinaryExpression(newElementIndex.clone(), BinaryOperator.MODULUS, iterspaceList.get(i).clone());
					Expression lbExp = lbList.get(i).clone();
					if( !(lbExp instanceof Literal) || !lbExp.toString().equals("0") ) {
						expr1 = new BinaryExpression(expr1, BinaryOperator.ADD, lbExp);
					}
					Statement stmt = new ExpressionStatement(new AssignmentExpression(tIndex.clone(), 
							AssignmentOperator.NORMAL, expr1));
					cStmt.addStatementBefore(firstNonDeclStmt, stmt);
					expr1 = new BinaryExpression(newElementIndex.clone(), BinaryOperator.DIVIDE, iterspaceList.get(i).clone());
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
				//				accLoop.getBody().swapWith(cStmt);
				innerLoop.getBody().swapWith(cStmt);
				CompoundStatement cStmt2 = new CompoundStatement();
				cStmt2.addStatement(innerLoop.clone());
				accLoop.getBody().swapWith(cStmt2);
			}
			PrintTools.println("[INFO] Number of tiled OpenACC loops: " + tiledLoops, 0);

		}
	}
	
	public void start()
	{
		
		///////////////////////////////////////////////////////////////////
		// this program transform a tile annotated loop like following   //
		// and call                                                      //
		// openacc.transform.CollapseTransformation.collapseLoop(loop);  //
		// cetus.transform.LoopNormalization.normalizeLoop(loop);        //
		// but there are any bugs                                        //
		//                                                               //
		// #pragma acc kernels loop tile(64,4) gang worker               //
		// for(int i = 0;i < 100;i++){                                   //
		//    for(int j = 0;j < 100;j++){                                //
		//    }                                                          //
		// }                                                             //
		//                                                               //
		// #pragma acc kernels loop collapse(2) gang                     //
		// for(int i = 0;i < 100;i+=4){                                  //
		//    for(int j = 0;j < 100;j+=64){                              //
		// #pragma acc loop collapse(2) worker                           //
		//       for(int ii = i;ii < MIN(100, i+4);ii++){                //
		//          for(int jj = j;jj < MIN(100, j+64);jj++){            //
		//          }                                                    //
		//       }                                                       //
		//    }                                                          //
		// }                                                             //
		//                                                               //
		///////////////////////////////////////////////////////////////////

		List<Procedure> procedureList = IRTools.getProcedureList(program);
		List<ForLoop> outer_loops = new ArrayList<ForLoop>();
		for( Procedure proc : procedureList )
		{
			List<ACCAnnotation> annotationList =
					AnalysisTools.collectPragmas(proc, ACCAnnotation.class);

			for(ACCAnnotation annot : annotationList)
			{
				if(!annot.containsKey("tile"))
					continue;

				Object tileClause = annot.get("tile");
				List<Expression> exprList = (List<Expression>)tileClause;

				List<Loop> loopList = LoopTools.calculateInnerLoopNest((Loop)annot.getAnnotatable());
				if(loopList.size() < exprList.size()) 
				{
					PrintTools.println("[Warning] Invalid tile clause " + annot + ".\n Skipped", 0);
					continue;
				}

				int collapseLevel = exprList.size();
				if(collapseLevel > 1){
					outer_loops.add((ForLoop)annot.getAnnotatable());
				}

			}
		}

		if(outer_loops.isEmpty()) {
			return;
		}else {
			int tiledLoops = 0;
			for( ForLoop accLoop : outer_loops ){
				Traversable t = (Traversable)accLoop; 
				while(true) {
					if (t instanceof Procedure) break;
					t = t.getParent(); 
				}
				if( t == null ) {
					Tools.exit("[ERROR in LoopTilingTransformation.collapseLoop()] Cannot find an enclosing procedure for the following loop:\n"
					+ accLoop + "\n");
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
							if( att.getParent() instanceof Annotatable ) {
								att = (Annotatable)att.getParent();
							} else {
								break;
							}
						}
					}
				}
				ArrayList<Symbol> indexSymbols = new ArrayList<Symbol>();
				ArrayList<ForLoop> indexedLoops = new ArrayList<ForLoop>();
				indexedLoops.add(accLoop);
				OmpAnnotation ompAnnot = accLoop.getAnnotation(OmpAnnotation.class, "for");
				ACCAnnotation tileAnnot = accLoop.getAnnotation(ACCAnnotation.class, "tile");
				Object tileClause = tileAnnot.get("tile");
				List<Expression> exprList = (List<Expression>)tileClause;
				int collapseLevel = exprList.size();
				boolean pnest = true;
				pnest = AnalysisTools.extendedPerfectlyNestedLoopChecking(accLoop, collapseLevel, indexedLoops, null);

				if( !pnest ) {
					Tools.exit("[ERROR] OpenACC tile clause is applicable only to perfectly nested loops;\n"
							+ "Procedure name: " + proc.getSymbolName() + "\nTarget loop: \n" +
							accLoop.toString() + "\n");
				}
				if( indexedLoops.size() < collapseLevel ) {
					PrintTools.println("\n[WARNING] Number of found loops (" + indexedLoops.size() + 
							") is smaller then tile dimension size (" + collapseLevel + "); skip the loop in procedure " 
							+ proc.getSymbolName() + ".\n",0);
					PrintTools.println("OpenACC loop\n" + accLoop + "\n", 2);
					continue;
				}
				for( ForLoop currLoop : indexedLoops ) {
					indexSymbols.add(LoopTools.getLoopIndexSymbol(currLoop));
				}
				tiledLoops++;

				ForLoop tiledLoop = null;
				for(int i = 0; i < exprList.size(); i++){
					ForLoop currentLoop = indexedLoops.get(i);
					long tileSize = ((IntegerLiteral)exprList.get(exprList.size()-1-i)).getValue();
					Expression lowerbound = LoopTools.getLowerBoundExpressionNS(currentLoop);
					Expression upperbound = LoopTools.getUpperBoundExpressionNS(currentLoop);

					Expression newLowerbound = new IntegerLiteral(0);
					//Expression newUpperbound = Symbolic.add(Symbolic.subtract(upperbound, lowerbound), new IntegerLiteral(1));
					Expression newUpperbound = Symbolic.subtract(upperbound, lowerbound);

					Expression indexVar = LoopTools.getIndexVariable(currentLoop);
					BinaryExpression condition = (BinaryExpression)currentLoop.getCondition();

					currentLoop.setInitialStatement(new ExpressionStatement(new AssignmentExpression(indexVar.clone(), AssignmentOperator.NORMAL, newLowerbound)));
					currentLoop.setCondition(new BinaryExpression(indexVar.clone(), BinaryOperator.COMPARE_LE, Symbolic.divide(newUpperbound, new IntegerLiteral(tileSize))));
					currentLoop.setStep(new AssignmentExpression(indexVar.clone(), AssignmentOperator.ADD, new IntegerLiteral(1)));
					//IRTools.replaceAll(currentLoop.getBody(), indexVar, Symbolic.multiply(new IntegerLiteral(tileSize), indexVar.clone()));

					Identifier newIndex = null;
					if( enclosingCompRegion instanceof CompoundStatement ) {
						newIndex = TransformTools.getNewTempIndex(enclosingCompRegion);
					} else if( enclosingCompRegion instanceof ForLoop ) {
						newIndex = TransformTools.getNewTempIndex(((ForLoop)enclosingCompRegion).getBody());
					} else {
						//Create a new index variable for the newly collapsed loop.
						CompoundStatement procBody = proc.getBody();
						newIndex = TransformTools.getNewTempIndex(procBody);
						//If the current accLoop is in a compute region, the above new private variable should be
						//added to the private clause.
						Set<SubArray> privateSet = null;
						if( tileAnnot.containsKey("private") ) {
							privateSet = tileAnnot.get("private");
						} else {
							privateSet = new HashSet<SubArray>();
							tileAnnot.put("private", privateSet);
						}
						privateSet.add(AnalysisTools.createSubArray(newIndex.getSymbol(), true, null));
						if( ompAnnot != null ) {
							Set<String> ompPrivSet = ompAnnot.get("private");
							if( ompPrivSet == null ) {
								ompPrivSet = new HashSet<String>();
								ompAnnot.put("private", ompPrivSet);
							}
							ompPrivSet.add(newIndex.toString());
						}
					}

					Expression newIndexExp = Symbolic.add(Symbolic.multiply(new IntegerLiteral(tileSize), indexVar.clone()), lowerbound.clone());
					Expression newInit = new AssignmentExpression(newIndex.clone(), AssignmentOperator.NORMAL, newIndexExp.clone());

					Expression newCondition = new BinaryExpression(newIndex.clone(), BinaryOperator.COMPARE_LE,
							new FunctionCall(new NameID("MIN"), upperbound,
									Symbolic.add(newIndexExp.clone(), new IntegerLiteral(tileSize-1))));
					Expression newStep = new UnaryExpression(UnaryOperator.POST_INCREMENT, newIndex.clone());

					ForLoop newTiledLoop = null;
					Statement innermostLoopBody = indexedLoops.get(exprList.size()-1).getBody();
					if(i == 0)
					{
						newTiledLoop = new ForLoop(new ExpressionStatement(newInit),newCondition,newStep,innermostLoopBody.clone());
						tiledLoop = newTiledLoop;
						CompoundStatement cstmt_tmp = new CompoundStatement();
						cstmt_tmp.addStatement(tiledLoop);
						innermostLoopBody.swapWith(cstmt_tmp);
					}
					else
					{
						newTiledLoop = new ForLoop(new ExpressionStatement(newInit),newCondition,newStep,tiledLoop.getBody().clone());
						CompoundStatement cstmt_tmp = new CompoundStatement();
						cstmt_tmp.addStatement(newTiledLoop);
						tiledLoop.getBody().swapWith(cstmt_tmp);
						if( (i == exprList.size()-1) ){
							ACCAnnotation newPragma = new ACCAnnotation();
							tiledLoop.annotate(newPragma);
							Object worker = tileAnnot.get("worker");
							newPragma.put("loop", "true");
							if( worker != null ) {
								newPragma.put("worker", worker);
								tileAnnot.remove("worker");
							}
							Expression ColLevel = new IntegerLiteral(exprList.size());
							newPragma.put("collapse", ColLevel);
							tileAnnot.put("collapse", ColLevel.clone());
							if( ompAnnot != null ) {
								Expression newIndext = LoopTools.getIndexVariable(tiledLoop);
								Set<String> ompPrivSet = ompAnnot.get("private");
								if( ompPrivSet == null ) {
									ompPrivSet = new HashSet<String>();
									ompAnnot.put("private", ompPrivSet);
								}
								ompPrivSet.add(newIndext.toString());
							}
						}
					}
					IRTools.replaceAll(newTiledLoop.getBody(), indexVar, newIndex.clone());
					if( (i == exprList.size()-1) ){
						CollapseTransformation.collapseLoop(tiledLoop, false);
						ACCAnnotation newPragma = tiledLoop.getAnnotation(ACCAnnotation.class, "collapse");
						newPragma.remove("collapse");
					}
				}
				ArrayList<ForLoop> newIndexedLoops = new ArrayList<ForLoop>();
				newIndexedLoops.add(accLoop);
				CollapseTransformation.collapseLoop(newIndexedLoops.get(0), false);
				tileAnnot.remove("collapse");
				LoopNormalization.normalizeLoop(newIndexedLoops.get(0));
				if( ompAnnot != null ) {
					Expression newIndext = LoopTools.getIndexVariable(accLoop);
					Set<String> ompPrivSet = ompAnnot.get("private");
					if( ompPrivSet == null ) {
						ompPrivSet = new HashSet<String>();
						ompAnnot.put("private", ompPrivSet);
					}
					ompPrivSet.add(newIndext.toString());
				}
			}
			PrintTools.println("[INFO] Number of tiled OpenACC loops: " + tiledLoops, 0);

		}
	}

	
	//@Override
	public void start_original()
	{
		// this program is created by Putt on 12/3/13. 
		
		List<Procedure> procedureList = IRTools.getProcedureList(program);
		for( Procedure proc : procedureList )
		{
			List<ACCAnnotation> annotationList =
					AnalysisTools.collectPragmas(proc, ACCAnnotation.class);

			for(ACCAnnotation annot : annotationList)
			{
				if(!annot.containsKey("tile"))
					continue;

				Object tileClause = annot.get("tile");
				List<Expression> exprList = (List<Expression>)tileClause;

				List<Loop> loopList = LoopTools.calculateInnerLoopNest((Loop)annot.getAnnotatable());
				if(loopList.size() != exprList.size())
				{
					PrintTools.println("[Warning] Invalid tile clause " + annot + ".\n Skipped", 0);
					continue;
				}

				// Reverse the list so the innermost loop show first
				Collections.reverse(loopList);

				ForLoop tiledLoop = null;

				for(int i = 0; i < loopList.size(); i++)
				{
					if(!(loopList.get(i) instanceof ForLoop))
					{
						PrintTools.println("[Warning] Can tile only for loop " + loopList.get(i) + ".\n Aborted", 0);
						return;
					}

					long tileSize = ((IntegerLiteral)exprList.get(i)).getValue();
					ForLoop origLoop = (ForLoop)loopList.get(i);

					//
					// Create tiled loop 
					//
					Expression lowerbound = LoopTools.getLowerBoundExpression(origLoop);
					Expression upperbound = LoopTools.getUpperBoundExpression(origLoop);

					Expression newLowerbound = new IntegerLiteral(0);
					Expression newUpperbound = Symbolic.divide(Symbolic.add(Symbolic.subtract(upperbound, lowerbound), new IntegerLiteral(1)), new IntegerLiteral(tileSize));

					Expression indexVar = LoopTools.getIndexVariable(origLoop);
					BinaryExpression condition = (BinaryExpression)origLoop.getCondition();

					origLoop.setInitialStatement(new ExpressionStatement(new AssignmentExpression(indexVar.clone(), AssignmentOperator.NORMAL, newLowerbound)));
					origLoop.setCondition(new BinaryExpression(indexVar.clone(), BinaryOperator.COMPARE_LT, 
							new BinaryExpression( 
									new BinaryExpression( 
											new BinaryExpression(newUpperbound, BinaryOperator.SUBTRACT, new IntegerLiteral(1)),
											BinaryOperator.DIVIDE, new IntegerLiteral(tileSize)), 
											BinaryOperator.ADD, new IntegerLiteral(1))));
					//origLoop.setStep(new AssignmentExpression(indexVar.clone(), AssignmentOperator.ADD, new IntegerLiteral(tileSize)));


					String tileLoopIndexVarName = indexVar.toString() + "__tile";

					VariableDeclarator indexVarDeclarator = new VariableDeclarator(new NameID(tileLoopIndexVarName));
					VariableDeclaration indexVarDecl = new VariableDeclaration(Arrays.asList(Specifier.INT) ,indexVarDeclarator);
					((CompoundStatement)annot.getAnnotatable().getParent()).addDeclaration(indexVarDecl);

					Symbol tileLoopIndexSymbol = SymbolTools.getSymbolOfName(tileLoopIndexVarName, annot.getAnnotatable().getParent());

					Identifier tileLoopIndexVar = new Identifier(tileLoopIndexSymbol);//TransformTools.getNewTempIndex(enclosingCompRegion);

					Expression newInit = new AssignmentExpression(tileLoopIndexVar.clone(), AssignmentOperator.NORMAL,
							Symbolic.add(Symbolic.multiply(indexVar.clone(), new IntegerLiteral(tileSize)), lowerbound));

					Expression bound = null;
					if(condition.getOperator() == BinaryOperator.COMPARE_LE)
					{
						bound = upperbound;
					}
					else if(condition.getOperator() == BinaryOperator.COMPARE_LT)
					{
						bound = Symbolic.add(upperbound, new IntegerLiteral(1));
					}
					PrintTools.println(condition.getOperator().toString() + " " + bound, 0);

					Expression newCondition = new BinaryExpression(tileLoopIndexVar.clone(), condition.getOperator(),
							new FunctionCall(new NameID("MIN"), bound,
									Symbolic.add(Symbolic.multiply(Symbolic.add(indexVar.clone(), new IntegerLiteral(1)), new IntegerLiteral(tileSize)),lowerbound)));
					Expression newStep = new UnaryExpression(UnaryOperator.POST_INCREMENT, tileLoopIndexVar.clone());

					ForLoop newTiledLoop = null;
					if(i == 0)
					{
						newTiledLoop = new ForLoop(new ExpressionStatement(newInit),newCondition,newStep,origLoop.getBody().clone());
					}
					else
					{
						newTiledLoop = new ForLoop(new ExpressionStatement(newInit),newCondition,newStep,tiledLoop);
					}

					IRTools.replaceAll(newTiledLoop.getBody(), indexVar, tileLoopIndexVar.clone());
					tiledLoop = newTiledLoop;
					if(annot.containsKey("private"))
					{
						HashSet privateVarList = (HashSet)annot.get("private");
						privateVarList.add(new SubArray(tileLoopIndexVar));
					}
					else
					{
						HashSet privateVarList = new HashSet();
						privateVarList.add(new SubArray(tileLoopIndexVar));
						annot.put("private", privateVarList);
					}
				}
				((ForLoop)loopList.get(0)).setBody(tiledLoop);
				//The loop is worker-only loop
				if(annot.containsKey("worker") && !annot.containsKey("gang"))
				{
					tiledLoop.annotate(annot.clone());
					annot.getAnnotatable().removeAnnotations(annot.getClass());
				}

				PrintTools.println(annot.getAnnotatable().getParent().toString(), 0);
			}
		}
	}
	

}
