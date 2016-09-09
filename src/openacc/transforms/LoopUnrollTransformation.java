package openacc.transforms;

/**
 * Created with IntelliJ IDEA.
 * User: asabne
 * Date: 12/3/13
 * Time: 3:03 PM
 * To change this template use File | Settings | File Templates.
 */


import java.util.*;

import cetus.analysis.LoopInfo;
import cetus.hir.*;
import cetus.analysis.LoopTools;
import cetus.transforms.TransformPass;
import openacc.hir.*;
import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;

public class LoopUnrollTransformation extends TransformPass {
  private int unrollFactor;
  
  
  public LoopUnrollTransformation(Program program, int factor) {
    super(program);
    unrollFactor = factor;
    verbosity = 1;
  }

  /* (non-Javadoc)
   * @see cetus.transforms.TransformPass#getPassName()
   */
  @Override
  public String getPassName() {
    return new String("[Loop Unroll transformation]");
  }

  /* (non-Javadoc)
   * @see cetus.transforms.TransformPass#start()
   */
  @Override
  public void start() {
    List<ACCAnnotation>  cRegionAnnots = AnalysisTools.collectPragmas(program, ACCAnnotation.class,
            ACCAnnotation.computeRegions, false);
    for (Annotation annot : cRegionAnnots) {
      Statement region = (Statement) annot.getAnnotatable();
      ArrayList<ForLoop> forLoops = new ArrayList<ForLoop>();
      //(ArrayList)IRTools.getStatementsOfType(region, ForLoop.class)

      DepthFirstIterator dfs_iter = new DepthFirstIterator(region);
      for (;;)
      {
        ForLoop loop = null;
        try {
          loop = (ForLoop)dfs_iter.next(ForLoop.class);
          forLoops.add(0, loop);
        } catch (NoSuchElementException e) {
          break;
        }
      }

      for( ForLoop f : forLoops) {
        ARCAnnotation arcAnnots = f.getAnnotation(ARCAnnotation.class, "transform");
        boolean containUnrollClause = false;
        if( (arcAnnots != null) && arcAnnots.containsKey("unroll")) {
        	containUnrollClause = true;
        }
        if( !containUnrollClause && (unrollFactor <= 1) ) {
        	continue;
        }
    	//Fixed by T.Hoshino Sep.12 2014  =================
    	ForLoop newFor = f.clone();
        //=================================================
        ACCAnnotation loopAnnots = f.getAnnotation(ACCAnnotation.class, "loop");
        ACCAnnotation seqAnnots = f.getAnnotation(ACCAnnotation.class, "seq");
        //Perform unrolling if the loop does not have OpenACC clauses, or has a sequential clause
        if(loopAnnots == null || seqAnnots != null) {
        	LoopInfo forLoopInfo = new LoopInfo(f);
        	Expression index = forLoopInfo.getLoopIndex();
        	Expression lb = forLoopInfo.getLoopLB();
        	Expression ub = forLoopInfo.getLoopUB();
        	Statement loopBody = f.getBody().clone();
        	//Perform unrolling only if the increment is 1
        	if(forLoopInfo.getLoopIncrement().toString().equals("1")) {
        		//Check if there is an openarc transform unroll pragma
        		long factor = unrollFactor;
        		if(containUnrollClause) {
        			//factor = Integer.parseInt(arcAnnots.get("unroll").toString());
        			Expression factorExp = arcAnnots.get("unroll");
        			if( factorExp instanceof IntegerLiteral ) {
        				factor = ((IntegerLiteral)factorExp).getValue();
        			} else {
        				//Skip the unrolling.
        				PrintTools.println("[WARNING in LoopUnrollingTransformation] Current implementation allows only integer unrolling factor; "
        						+ "the following unroll clause will be ignored.\n"
        						+ "OpenARC annotation: " + arcAnnots + AnalysisTools.getEnclosingAnnotationContext(arcAnnots), 0);
        				factor = 0;
        			}
        		}

        		if(factor > 1) {
        			Expression itrSize = Symbolic.simplify(Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1)));
        			long nItr = 0;
					if( itrSize instanceof IntegerLiteral ) {
						nItr = ((IntegerLiteral)itrSize).getValue();
						if( factor > nItr ) {
							//Skip the unrolling.
							PrintTools.println("[WARNING in LoopUnrollingTransformation] unrolling factor (" + factor + 
									") is bigger than the number of iterations (" + nItr + "); "
        						+ "the following unroll clause will be ignored.\n"
        						+ "OpenARC annotation: " + arcAnnots + AnalysisTools.getEnclosingAnnotationContext(arcAnnots), 0);
							continue;
							
						}
					}
					if( nItr == factor ) {
						PrintTools.println("complete unrolling!", 0);
						CompoundStatement parent = (CompoundStatement) f.getParent();
						for(long i=factor-1; i>= 0; i--) {
							Statement newBody = loopBody.clone();
							IRTools.replaceAll(newBody,index, new IntegerLiteral(i));
							parent.addStatementAfter(f, newBody);
						}
						parent.removeStatement(f);
					} else {
						for(int i=1; i< factor; i++) {
							Statement newBody = loopBody.clone();
							BinaryExpression newIndex = new BinaryExpression(index.clone(), BinaryOperator.ADD,
									new IntegerLiteral(i));
							IRTools.replaceAll(newBody,index, newIndex);
							((CompoundStatement)f.getBody()).addStatement(newBody);
						}
						AssignmentExpression newStep = new AssignmentExpression( index.clone(),
								AssignmentOperator.NORMAL, new BinaryExpression(index.clone(), BinaryOperator.ADD,
										new IntegerLiteral(factor)) );
						f.setStep(newStep);
						boolean addTrailCode = true;
						boolean completeUnrolling = false;
						if( itrSize instanceof IntegerLiteral ) {
							if( nItr == factor ) {
								completeUnrolling = true;
								addTrailCode = false;
							} else if( nItr%factor == 0 ) {
								addTrailCode = false;
							}
						}
						if( completeUnrolling ) {
							PrintTools.println("complete unrolling!", 0);
							loopBody = f.getBody();
							f.setBody(null);
							f.swapWith(loopBody);

						} else if( addTrailCode ) {
							newFor.setInitialStatement(null);
							Statement parent = (Statement) f.getParent();
							((CompoundStatement)parent).addStatementAfter(f, newFor);
						}
					}
        			//=================================================
        		}
        	}
        }
        
      }


    }


  }
}

