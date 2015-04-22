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
    	//Fixed by T.Hoshino Sep.12 2014  =================
    	ForLoop newFor = f.clone();
        //=================================================
        ACCAnnotation loopAnnots = f.getAnnotation(ACCAnnotation.class, "loop");
        ACCAnnotation seqAnnots = f.getAnnotation(ACCAnnotation.class, "seq");
        //Perform unrolling if the loop does not have OpenACC clauses, or has a sequential clause
        if(loopAnnots == null || seqAnnots != null) {
          LoopInfo forLoopInfo = new LoopInfo(f);
          Expression index = forLoopInfo.getLoopIndex();
          Statement loopBody = f.getBody().clone();
          //Perform unrolling only if the increment is 1
          if(forLoopInfo.getLoopIncrement().toString().equals("1")) {
            //Check if there is a cuda unroll pragma
            ACCAnnotation cudaAnnots = f.getAnnotation(ACCAnnotation.class, "cuda");
            int factor = unrollFactor;
            if(cudaAnnots != null)  {
              if(cudaAnnots.containsKey("unroll")) {
                factor = Integer.parseInt(cudaAnnots.get("unroll").toString());
              }
            }

            if(factor > 1) {
            	//[FIXME] below transformation works only if unrolling factor evenly divides into 
            	//the number of iterations.
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
              //Fixed by T.Hoshino Sep.12 2014  =================
              newFor.setInitialStatement(null);
              Statement parent = (Statement) f.getParent();
              ((CompoundStatement)parent).addStatementAfter(f, newFor);
              //=================================================
            }
          }
        }
        
      }


    }


  }
}

