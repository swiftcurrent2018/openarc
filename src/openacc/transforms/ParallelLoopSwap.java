package openacc.transforms;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import cetus.transforms.LoopInterchange;
import cetus.analysis.DDGraph;
import cetus.analysis.DependenceVector;
import cetus.analysis.LoopTools;
import cetus.exec.Driver;
import cetus.hir.*;
import openacc.analysis.AnalysisTools;
import openacc.hir.ACCAnnotation;
import openacc.hir.ARCAnnotation;

/**
 * ParallelLoopSwap is a transformation, which is a reverse operation of LoopInterchange.
 * [FIXME] this pass should be customized to OpenACC context.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 *
 */
public class ParallelLoopSwap extends LoopInterchange {
	
	private int debug_level = 1;
	private int loopSwapMode = 1; //1 => swap loops so that the innermost loop has the lowest rank.
	                              //2 => swap loops so that the innermost loop has the highest rank.
	
	public ParallelLoopSwap( Program program, int mode ) {
		super(program);
		loopSwapMode = mode;
	}
	
	public String getPassName()
	{
		return new String("[ParallelLoopSwap]");
	}

	public void start()
	{
		LinkedList<Loop> loops = new LinkedList<Loop>();
		LinkedList<Loop> loops2 = new LinkedList<Loop>();
		List<ForLoop> outer_loops = new ArrayList<ForLoop>();
		List<DependenceVector> depVec = new ArrayList<DependenceVector>();
		List<Expression> expList = new LinkedList<Expression>();
		DepthFirstIterator iter = null;
		int i;
		int target_loops = 0;
		int num_single = 0, num_non_perfect = 0, num_contain_func = 0, num_loop_interchange=0;
		int num_skipped = 0, num_not_in_parallel = 0, num_not_interchangeable = 0;
		
		String value = Driver.getOptionValue("extractTuningParameters");
		boolean extractTuningParameters = false;
		if( value != null ) {
			extractTuningParameters = true;
		}
		
		List<ACCAnnotation> forAnnotList = IRTools.collectPragmas(program, ACCAnnotation.class, "loop");
		for( ACCAnnotation fAnnot : forAnnotList ) {
			Annotatable at = fAnnot.getAnnotatable();
			if( at instanceof ForLoop ) {
				outer_loops.add((ForLoop)at);
			}
		}


		boolean initStmtPrinted = false;
		for(i = outer_loops.size()-1; i >= 0; i--)
		{
			ForLoop accLoop = outer_loops.get(i);
			iter = new DepthFirstIterator(accLoop);
			ACCAnnotation pAnnot = null;
			ARCAnnotation noAnnot = null;
			Statement pRegion = null;
			boolean LoopSwapped = false;
			if( accLoop.containsAnnotation(ACCAnnotation.class, "kernels") || accLoop.containsAnnotation(ACCAnnotation.class, "parallel")) {
				pRegion = accLoop;
			} else {
				pAnnot = AnalysisTools.ipFindFirstPragmaInParent(accLoop, ACCAnnotation.class, ACCAnnotation.computeRegions, 
						false, null, null);
				if( pAnnot != null ) {
					pRegion = (Statement)pAnnot.getAnnotatable();
				}
			}
			if( pRegion == null ) {
				if( !initStmtPrinted ) {
					PrintTools.println("[WARNING in ParallelLoopSwap()] Can not find compute regions " +
							"enclosing the following OpenACC loops; parallel loop-swap will not be applied the following loop:", 1);
					initStmtPrinted = true;
				}
				ACCAnnotation fAnnot = accLoop.getAnnotation(ACCAnnotation.class, "loop");
				PrintTools.println("==> OpenACC loop annotation : " + fAnnot, 1);
				Procedure proc = IRTools.getParentProcedure(accLoop);
				PrintTools.println("    enclosing procedure : " + proc.getSymbolName(), 1);
				num_not_in_parallel++;
				continue;
			}
			if( !extractTuningParameters ) {
				noAnnot = pRegion.getAnnotation(ARCAnnotation.class, "noploopswap");
			}
			loops.clear();
			loops2.clear();
			while(iter.hasNext()) {
				Object o = iter.next();
				if(o instanceof ForLoop)
					loops.add((Loop)o);
			}
			if(loops.size() < 2) {
				num_single++;
				continue;
			}	else if(!LoopTools.isPerfectNest((ForLoop)loops.get(0))) {
				num_non_perfect++;
				continue;
			}	else if(LoopTools.containsFunctionCall((ForLoop)loops.get(0))) {
				num_contain_func++;
				continue;
			} else {
				if( !extractTuningParameters && (noAnnot != null) ) {
					num_skipped++;
					continue;
				}
				target_loops++;
				Statement stm = ((ForLoop)loops.get(loops.size()-1)).getBody();
				List<ArrayAccess> arrays = new ArrayList();  // Arrays in loop body
				DepthFirstIterator iter2 = new DepthFirstIterator(stm);

				while(iter2.hasNext())
				{
					Object child = iter2.next();
					if(child instanceof ArrayAccess)
					{
						arrays.add((ArrayAccess)child);
					}
				}

				int r = 0,j,until = loops.size();
				int target_index = 0;
				boolean icFlag = true;
				List<Integer> rank;
				int rankSize;
				 
				if( loopSwapMode == 1 ) {
	                while(icFlag)
	                {
	                    Expression exp;
	                    expList.clear();
	                    for(j = 0; j < until; j++)
	                    {
	                        exp = LoopTools.getIndexVariable((ForLoop)loops.get(j));
	                        if(exp != null)
	                            expList.add(exp);
	                    }
	                    rank = getRank(arrays, expList, target_index);
	                    rankSize = rank.size();
	                    for(j = 0; j < rankSize; j++) 
	                    {
	                        r = getRank2(rank, expList, loops);
	                        rank.remove(rank.indexOf(r));
							int r1, r2;
							r1 = r;

	                        if(expList.size() < until) until = expList.size();
	                        for(int k = r+1; k < until; k++)
	                        {
	                            if(isLegal(loops, r, k))
	                            {
	                                swapLoop((ForLoop)loops.get(r), (ForLoop)loops.get(k));
	                                num_loop_interchange++;
	                                Collections.swap(expList, r, k);
	                                r = k;
	                                LoopSwapped = true;
	                            } else {
	                                break;
	                            }
	                        }        
	                        until = r;
	                        ///////////////////////////////////////////////////////////////////////////////////
	                        // FIXME: requested by SY Lee                                                    //
	                        // If the list "rank" consists of r1 and r2 (r1 < r2), and if getRank2() returns //
	                        // r1 first, the loop pointed by r2 will be moved upward by 1. Thus r2 in the    //
	                        // list "rank" should be changed with r2-1.                                      //
	                        ///////////////////////////////////////////////////////////////////////////////////
							for( int k = 0; k<rank.size()-1; k++) {
								r2 = rank.get(k);
								if( (r1 < r2) && (r2 <= r) ) {
									rank.set(k, r2-1);
								}
							}
	                    }
	                    target_index++;
	                    if(until == 0) icFlag = false;
	                }
				} else {
					////////////////////////////////////////////////////
					// loops2 contains nested loops in reverse order, //
					// from the innermost to the outermost.           //
					////////////////////////////////////////////////////
					for( j= until-1; j>=0; j-- ) {
						loops2.add(loops.get(j));
					}

					while(icFlag)
					{
						Expression exp;
						expList.clear();
						for(j = 0; j < until; j++)
						{
							exp = LoopTools.getIndexVariable((ForLoop)loops2.get(j));
							if(exp != null)
								expList.add(exp);
						}
						rank = getRank(arrays, expList, target_index);
						rankSize = rank.size();
						for(j = 0; j < rankSize; j++) 
						{
							r = getRank2(rank, expList, loops2);
							rank.remove(rank.indexOf(r));
							int r1, r2;
							r1 = r;

							if(expList.size() < until) until = expList.size();
							for(int k = r+1; k < until; k++)
							{
								if(isLegal(loops2, r, k))
								{
									swapLoop((ForLoop)loops2.get(r), (ForLoop)loops2.get(k));
									//num_loop_interchange++;
									Collections.swap(expList, r, k);
									r = k;
									LoopSwapped = true;

								} else {
									break;
								}
							}		
							until = r;
							///////////////////////////////////////////////////////////////////////////////////
							// FIXME: requested by SY Lee                                                    //
							// If the list "rank" consists of r1 and r2 (r1 < r2), and if getRank2() returns //
							// r1 first, the loop pointed by r2 will be moved downward by 1. Thus r2 in the  //
							// list "rank" should be changed with r2-1.                                      //
							///////////////////////////////////////////////////////////////////////////////////
							for( int k = 0; k<rank.size()-1; k++) {
								r2 = rank.get(k);
								if( (r1 < r2) && (r2 <= r) ) {
									rank.set(k, r2-1);
								}
							}
						}
						target_index++;
						if(until == 0) icFlag = false;
					}
				}
				if( LoopSwapped ) {
					num_loop_interchange++;
				} else {
					num_not_interchangeable++;
				}
			}
			if( extractTuningParameters && LoopSwapped ) {
				//"tuningparameter" clause is used only internally .
				ACCAnnotation tAnnot = pRegion.getAnnotation(ACCAnnotation.class, "tuningparameters");
				if( tAnnot == null ) {
					tAnnot = new ACCAnnotation("tuningparameters", "_directive");
					pRegion.annotate(tAnnot);
				}
				tAnnot.put("ploopswap", "_clause");
			}
		}

		PrintTools.println("# of OpenACC Loops: " + outer_loops.size(), debug_level);
		PrintTools.println("# of Not-in-parallel loops : " + num_not_in_parallel, debug_level);
		PrintTools.println("# of Single loops : " + num_single, debug_level);
		PrintTools.println("# of Non Perfect loops : " + num_non_perfect, debug_level);
		PrintTools.println("# of Function-containing loops : " + num_contain_func, debug_level);
		PrintTools.println("# of User-skipped Loops : " + num_skipped, debug_level);
		//PrintTools.println("# of Candidate loops : " + target_loops, debug_level);
		PrintTools.println("# of Interchanged loops : " + num_loop_interchange, debug_level);
		PrintTools.println("# of Not-Interchanged loops : " + num_not_interchangeable, debug_level);

		return;
	}
	
	/**
	 * Check legality of loop interchange between rsrc and rtarget. 
	 * Both rsrc and rtarget are in nest loops and rsrc is inner than rtarget                                         
	 * FIXME: this method can not detect loop-carried dependency in the cases
	 * where index expressions are indirectly used such as the following:
	 *     for (i = LBi; i <= UBi; i++) {
     *         for (k = LBk; k <= UBk; k++) {
     *             k1 = k  + 1; 
     *             A[i][k1] = A[i][k] - B[i][k1]; 
     *         }
     *     }
	 * 
	 * @param reverseNest nested loops in reverse order, from innermost to the outermost
	 * @param rsrc source loop id in the reverseNest list
	 * @param rtarget target loop id in the reverseNest list
	 * @return true if it is legal to swap two loops, rsrc and rtarget
	 */
	public boolean isLegal(LinkedList<Loop> reverseNest, int rsrc, int rtarget)
	{
		int i, j, next;
		int src, target;
		DDGraph ddg;
		String str;
		ArrayList<DependenceVector> dpv;
		DependenceVector dd;
		//////////////////////////////////////////////////////
		// Nested loops from the outermost to the innermost //
		//////////////////////////////////////////////////////
		LinkedList<Loop> nest = new LinkedList<Loop>();
		if( loopSwapMode == 1 ) {
			nest.addAll(reverseNest);
			src = rsrc;
			target = rtarget;
		} else {
			j = reverseNest.size();
			for(i=j-1; i>=0; i--) {
				nest.add(reverseNest.get(i));
			}
			src = j-rsrc-1;
			target = j-rtarget-1;
		}
		
		ddg = program.getDDGraph();
		dpv = ddg.getDirectionMatrix(nest);

		if(src == target) return true;
		///////////////////////////////////////////////////////////
		// Swap src and target to make src refer to a loop outer //
		// than a target loop.                                   //
		///////////////////////////////////////////////////////////
		if(src > target) {
			i = src;
			src = target;
			target = i;
		}
		
		ForLoop srcLoop = (ForLoop)nest.get(src);
		ForLoop targetLoop = (ForLoop)nest.get(src);
		////////////////////////////////////////////////////////////////
		// If both loops contain acc loop directive, both loop should //
		// contain only independent clause, which is safe but maybe   //
		// too conservative.                                          //
		////////////////////////////////////////////////////////////////
		if( !srcLoop.containsAnnotation(ACCAnnotation.class, "loop") || 
				srcLoop.containsAnnotation(ACCAnnotation.class, "seq") ||
				srcLoop.containsAnnotation(ACCAnnotation.class, "gang") ||
				srcLoop.containsAnnotation(ACCAnnotation.class, "worker") ||
				srcLoop.containsAnnotation(ACCAnnotation.class, "vector") ||
				srcLoop.containsAnnotation(ACCAnnotation.class, "collapse") ) {
			return false;
		}
		if( !targetLoop.containsAnnotation(ACCAnnotation.class, "loop") || 
				targetLoop.containsAnnotation(ACCAnnotation.class, "seq") ||
				targetLoop.containsAnnotation(ACCAnnotation.class, "gang") ||
				targetLoop.containsAnnotation(ACCAnnotation.class, "worker") ||
				targetLoop.containsAnnotation(ACCAnnotation.class, "vector") ||
				targetLoop.containsAnnotation(ACCAnnotation.class, "collapse") ) {
			return false;
		}
		
		//////////////////////////////////////////////////////////////
		// Check whether index variable for the outer loop (src) is //
		// used in initial statements or condition expressions of   //
		// the loops residing between src and target.               //
		//////////////////////////////////////////////////////////////
		Expression srcIndex = LoopTools.getIndexVariable(srcLoop);
		if( srcIndex == null ) {
			//Can't identify the index variable of the src loop; 
			//return false conservatively.
			return false;
		} else {
			for( i=src+1; i<=target; i++ ) {
				ForLoop tLoop = (ForLoop)nest.get(i);
				Statement initStmt = tLoop.getInitialStatement();
				if( IRTools.containsExpression(initStmt, srcIndex) ) {
					return false;
				}
				Expression condExp = tLoop.getCondition();
				if( IRTools.containsExpression(condExp, srcIndex) ) {
					return false;
				}
			}
		}

		for(i = 0; i < dpv.size(); i++)
		{
			dd = dpv.get(i);
			str = dd.toString();
			for(j = 0; j < str.length(); j++)
			{
				if(j == src) next = target;
				else if(j == target) next = src;
				else next = j;

				if(next < str.length()) {
					if(str.charAt(next) == '>') return false;
					if(str.charAt(next) == '<') break;
				}
			}
		}

		return true;
	}

}
