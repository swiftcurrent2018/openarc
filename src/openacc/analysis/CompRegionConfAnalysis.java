package openacc.analysis;

import cetus.analysis.AnalysisPass;
import cetus.analysis.LoopTools;
import cetus.hir.*;
import cetus.exec.Driver;
import openacc.hir.*;

import java.util.HashSet;
import java.util.List;
import java.util.LinkedList;
import java.util.ArrayList;

/**
 * This pass analyzes iteration sizes of worksharing loops and decides compute region configuration (gang size and worker size)
 * if not explicitly specified. If maxNumGangs is specified, the number of gangs will be min(maxNumGangs, calculatedNumGangs).
 * For each outermost gang/worker loop in Kernels loop, gangconf/workerconf clause will be added where the numbers of gangs/workers 
 * for each gang/worker loop are stored in the reverse order (the innermost loop first).
 * For each outermost gang/worker loop in Kernels loop, totalnumgangs/totalnumworkers clauses will be also added.
 * 
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class CompRegionConfAnalysis extends AnalysisPass {
	private String pass_name = "[CompRegionConfAnalysis]";
	private int defaultNumWorkers = 64;
	private int maxNumGangs = 0;
	private int systemMaxNumGangs = 65535; //Maximum number of gangs that CUDA system allows.
	private int maxNumWorkers = 512;
	private int OPENARC_ARCH = 0;

	public CompRegionConfAnalysis(Program program, int target) {
		super(program);
		OPENARC_ARCH = target;
	}

	@Override
	public String getPassName() {
		return pass_name;
	}

	@Override
	public void start() {
		String value = Driver.getOptionValue("defaultNumWorkers");
		if( value != null ) {
			defaultNumWorkers = Integer.valueOf(value).intValue();
		}
		value = Driver.getOptionValue("maxNumGangs");
		if( value != null ) {
			maxNumGangs = Integer.valueOf(value).intValue();
		}
		value = Driver.getOptionValue("maxNumWorkers");
		if( value != null ) {
			maxNumWorkers = Integer.valueOf(value).intValue();
		}
		value = Driver.getOptionValue("CUDACompCapability");
		if( OPENARC_ARCH == 0 ) {
			double CUDACompCapability = 1.1;
			if( value != null ) {
				CUDACompCapability = Double.valueOf(value).doubleValue();
			}
			if( CUDACompCapability >= 2.0 ) {
				if( maxNumWorkers > 1024 ) {
					maxNumWorkers = 1024;
				}
			} else {
				if( maxNumWorkers > 512 ) {
					maxNumWorkers = 512;
				}
			}
		}
		List<ACCAnnotation>  cRegionAnnots = AnalysisTools.collectPragmas(program, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
		List<ACCAnnotation> parallelRegions = new ArrayList<ACCAnnotation>();
		List<ACCAnnotation> kernelsRegions = new ArrayList<ACCAnnotation>();
		List<ACCAnnotation> seqKernelRegions = new ArrayList<ACCAnnotation>();
		if( cRegionAnnots != null ) {
			for( ACCAnnotation cAnnot : cRegionAnnots ) {
				if( cAnnot.containsKey("parallel") ) {
					parallelRegions.add(cAnnot);
				} else {
					kernelsRegions.add(cAnnot);
					if( cAnnot.containsKey("seq") ) {
						Annotatable at = cAnnot.getAnnotatable();
						if( AnalysisTools.ipContainPragmas(at, ACCAnnotation.class, ACCAnnotation.parallelWorksharingClauses, false, null) ) {
							seqKernelRegions.add(cAnnot);
						}
					}
				}
			}
			//Calculate number of gangs and workers for each kernels region if missing.
			calKernelsLoopConfiguration(kernelsRegions);
			//Calculate number of gangs and workers for each parallel region if missing.
			calParallelRegionConfConfiguration(parallelRegions);
			//Calculate number of gangs and workers for each seq kernel loop containing inner gang/worker loops.
			calSeqKernelRegionConfConfiguration(seqKernelRegions);
		}
	}
	
	/**Calculate number of gangs and workers for each kernels region if missing.
	 * At the end of this analysis, (1) each gang loop contains its iteration size ("iterspace") in an internal annotation,
	 * (2) gang(num_gangs) is added to each gang loop in Kernels regions if missed, and (3) worker(num_workers) is added to
	 * each worker loop in kernels regions if missed (for 2D worker loops, default value is 16 for each, and for 3D worker loops,
	 * default value is 8 for each.
	 * 
	 * @param kernelsRegions
	 */
	private void calKernelsLoopConfiguration( List<ACCAnnotation> kernelsRegions ) {
		List<FunctionCall> funcCallList = IRTools.getFunctionCalls(program);
		//Handle kernels regions.
		for( ACCAnnotation cAnnot : kernelsRegions ) {
			Annotatable at = cAnnot.getAnnotatable();
			if( !(at instanceof ForLoop) ) {
				Procedure pProc = IRTools.getParentProcedure(at);
				Tools.exit("[ERROR] Internal error in CompRegionConfAnalysis(); non-loop kernels region is found.\n" +
						"Enclosing procedure: " + pProc.getSymbolName() + "\nOpenACC annotation: " + cAnnot);
			}
			List<ACCAnnotation> gangAnnots = AnalysisTools.ipCollectPragmas(at, ACCAnnotation.class, "gang", null);
			if( gangAnnots == null ) {
				Procedure pProc = IRTools.getParentProcedure(at);
				Tools.exit("[ERROR] Internal error in CompRegionConfAnalysis(); no gang clause is found in a kernels region.\n" +
						"Enclosing procedure: " + pProc.getSymbolName() + "\nOpenACC annotation: " + cAnnot);
			} else {
				//Handle gang loops in the Kernels loops.
				for( ACCAnnotation gAnnot : gangAnnots ) {
					ForLoop gLoop = (ForLoop)gAnnot.getAnnotatable();
					Object gVal = gAnnot.get("gang");
					Expression numGangs;
					Expression iterspace = calIterSpace(gLoop, at, gAnnot);
					if( iterspace == null ) {
						Tools.exit("[ERROR in CompRegionConfAnalysis.calKernelLoopConfiguration()] error in calculating iteration " +
								"space for the following gang loop; exit\n" + 
								gLoop + "\n");

					}
					//Store iteration space size into an internal annotation.
					Annotation iAnnot = gLoop.getAnnotation(ACCAnnotation.class, "internal");
					if( iAnnot == null ) {
						iAnnot = new ACCAnnotation("internal", "_directive");
						iAnnot.setSkipPrint(true);
						gLoop.annotate(iAnnot);
					}
					iAnnot.put("iterspace", iterspace);
					//Check whether current gang loop contains worker clause.
					ACCAnnotation wAnnot = gLoop.getAnnotation(ACCAnnotation.class, "worker");
					if( wAnnot == null ) {
						numGangs = iterspace;
					} else {
						Object wObj = wAnnot.get("worker");
						Expression numWorkers;
						if( wObj instanceof Expression ) {
							numWorkers = (Expression)wObj;
						} else {
							int wCnt = 1;
							if( AnalysisTools.ipFindFirstPragmaInParent(gLoop, ACCAnnotation.class, "worker", funcCallList, null) != null ) {
								wCnt++;
							}
							if( AnalysisTools.ipContainPragmas(gLoop.getBody(), ACCAnnotation.class, "worker", null)) {
								wCnt++;
							}
							if( wCnt == 1 ) {
								numWorkers = new IntegerLiteral(defaultNumWorkers);
							} else if( wCnt == 2 ) {
								numWorkers = new IntegerLiteral(16);
							} else {
								numWorkers = new IntegerLiteral(8);
							}
							wAnnot.put("worker", numWorkers);
						}
						//numGangs = Symbolic.divide(iterspace, numWorkers);
						if( (iterspace instanceof IntegerLiteral) && (numWorkers instanceof IntegerLiteral) ) {
							int gsize = (int)Math.ceil( ((double)((IntegerLiteral)iterspace).getValue())/
									((double)((IntegerLiteral)numWorkers).getValue()) );
							numGangs = new IntegerLiteral(gsize);
						} else {
							if( (numWorkers instanceof IntegerLiteral) && (((IntegerLiteral)numWorkers).getValue() == 1) ) {
								numGangs = iterspace.clone();
							} else {
								List<Specifier> specs = new ArrayList<Specifier>(1);
								specs.add(Specifier.FLOAT);
								Expression floatsize1 = new Typecast(specs, iterspace.clone());
								Expression floatsize2;
								if( numWorkers instanceof IntegerLiteral ) {
									floatsize2 = new FloatLiteral((double)((IntegerLiteral)numWorkers.clone()).getValue(), "F");
								} else {
									specs = new ArrayList<Specifier>(1);
									specs.add(Specifier.FLOAT);
									floatsize2 = new Typecast(specs, numWorkers.clone());
								}
								FunctionCall ceilfunc = new FunctionCall(new NameID("ceil"));
								Expression argExp = Symbolic.divide(floatsize1, floatsize2); 
								ceilfunc.addArgument(argExp);
								//numGangs = ceilfunc;
								specs = new ArrayList<Specifier>(1);
								specs.add(Specifier.INT);
								numGangs = new Typecast(specs, ceilfunc);
							}
						}
					}
					// If maxNumGangs is set, the actual number of gangs should be min(numGangs, maxNumGangs).
					if( maxNumGangs > 0 ) {
						numGangs = Symbolic.simplify( 
								new MinMaxExpression(true, numGangs, new IntegerLiteral(maxNumGangs)));
					}
					///////////////////////////////////////////////////////////////////////////////
					// If symplified max expression has more than two children (ex: max(a,b,c)), //
					// convert it to max expression with two children (ex: max(max(a,b),c).      //
					///////////////////////////////////////////////////////////////////////////////
					List<MinMaxExpression> maxExpList = IRTools.getExpressionsOfType(numGangs, MinMaxExpression.class);
					while( !maxExpList.isEmpty() ) {
						Expression oldMaxExp = maxExpList.remove(maxExpList.size()-1);
						Expression maxExp = oldMaxExp;
						List<Traversable> children = (List<Traversable>)maxExp.getChildren();
						if( children.size() > 2 ) {
							maxExp = ((Expression)children.get(0)).clone();
							for( int i=1; i<children.size(); i++ ) {
								maxExp = new MinMaxExpression(false, maxExp, ((Expression)children.get(i)).clone());
							}
							oldMaxExp.swapWith(maxExp);
						}
					}
					//Number of gangs is not explicitly specified.
					if( !(gVal instanceof Expression) ) {
						gAnnot.put("gang", numGangs);
					}
				}
				//Add gangconf/totalnumgangs clause to each outermost gang loop.
				for( ACCAnnotation gAnnot : gangAnnots ) {
					ForLoop gLoop = (ForLoop)gAnnot.getAnnotatable();
					Traversable tt = gLoop.getParent();
					boolean outermostloop = true;
					while( tt != null ) {
						if( (tt instanceof Annotatable) && ((Annotatable)tt).containsAnnotation(ACCAnnotation.class, "gang") ) {
							outermostloop = false;
							break;
						}
						tt = tt.getParent();
					}
					if( outermostloop ) {
						List<ForLoop> nestedGLoops = AnalysisTools.findDirectlyNestedLoopsWithClause(gLoop, "gang");
						List<Expression> tExpList = new ArrayList<Expression>(3);
						for( ForLoop tLoop : nestedGLoops ) {
							ACCAnnotation iAnnot = tLoop.getAnnotation(ACCAnnotation.class, "gang");
							Expression tExp = iAnnot.get("gang");
							if( tExp != null ) {
								tExpList.add(tExp);
							}
						}
						if( nestedGLoops.size() != tExpList.size() ) {
							Tools.exit("[ERROR in CompRegionConfAnalysis.calKernelLoopConfiguration()] gang clause does not have" +
									"an argument, which should have been added either by a user or by a compiler; exit\n" + 
									gLoop + "\n");
						} else {
							ACCAnnotation iAnnot = gLoop.getAnnotation(ACCAnnotation.class, "internal");
							if( iAnnot == null ) {
								iAnnot = new ACCAnnotation("internal", "_directive");
								iAnnot.setSkipPrint(true);
								gLoop.annotate(iAnnot);
							}
							List<Expression> rExpList = new ArrayList<Expression>(tExpList.size());
							for( int i=tExpList.size()-1; i>=0; i--) {
								Expression tExp = tExpList.get(i);
								if( (tExp instanceof IntegerLiteral) && 
										(((IntegerLiteral)tExp).getValue() > systemMaxNumGangs) ) {
									Procedure pProc = IRTools.getParentProcedure(at);
/*									Tools.exit("[ERROR in CompRegionConfAnalysis.calKernelLoopConfiguration()] the number of gangs (" +
											tExp.toString() + ") in the following compute region is bigger than the system-supported maximum size (" + 
											systemMaxNumGangs + "); either put explicit smaller gang size or increase worker size to meet the system limit; exit\n" + 
											"Enclosing procedure: " + pProc.getSymbolName() + "\nOpenACC annotation: " + cAnnot + "\n");*/
									PrintTools.println("[WARNING in CompRegionConfAnalysis.calKernelLoopConfiguration()] the number of gangs (" +
											tExp.toString() + ") in the following compute region is bigger than the system-supported maximum size (" + 
											systemMaxNumGangs + "); the system-supported maximum size will be used for the number of gangs.\n" + 
											"Enclosing procedure: " + pProc.getSymbolName() + "\nOpenACC annotation: " + cAnnot + "\n", 0);
									tExp = new IntegerLiteral(systemMaxNumGangs);
									ForLoop ttloop = nestedGLoops.get(i);
									ACCAnnotation ttAnnot = ttloop.getAnnotation(ACCAnnotation.class, "gang");
									ttAnnot.put("gang", tExp.clone());
								}
								rExpList.add(tExp.clone());
							}
							iAnnot.put("gangconf", rExpList);
							Expression totalNumGangs = null;
							for( Expression exp : rExpList ) {
								if( totalNumGangs == null ) {
									totalNumGangs = exp.clone();
								} else {
									totalNumGangs = Symbolic.multiply(totalNumGangs, exp.clone());
								}
							}
							iAnnot.put("totalnumgangs", totalNumGangs);
						}
					}
				}
			}
			//Handle worker loops.
			List<ACCAnnotation> workerAnnots = AnalysisTools.ipCollectPragmas(at, ACCAnnotation.class, "worker", null);
			if( workerAnnots != null ) {
				for( ACCAnnotation wAnnot : workerAnnots ) {
					Object wObj = wAnnot.get("worker");
					if( !(wObj instanceof Expression) ) {
						ForLoop wLoop = (ForLoop)wAnnot.getAnnotatable();
						Expression numWorkers = null;
						int wCnt = 1;
						if( AnalysisTools.ipFindFirstPragmaInParent(wLoop, ACCAnnotation.class, "worker", funcCallList, null) != null ) {
							wCnt++;
						}
						if( AnalysisTools.ipContainPragmas(wLoop.getBody(), ACCAnnotation.class, "worker", null)) {
							wCnt++;
						}
						if( wCnt == 1 ) {
							numWorkers = new IntegerLiteral(defaultNumWorkers);
						} else if( wCnt == 2 ) {
							numWorkers = new IntegerLiteral(16);
						} else {
							numWorkers = new IntegerLiteral(8);
						}
						wAnnot.put("worker", numWorkers);
					}
				}
				//Add workerconf clause to each outermost worker loop.
				for( ACCAnnotation wAnnot : workerAnnots ) {
					ForLoop wLoop = (ForLoop)wAnnot.getAnnotatable();
					Traversable tt = wLoop.getParent();
					boolean outermostloop = true;
					while( tt != null ) {
						if( (tt instanceof Annotatable) && ((Annotatable)tt).containsAnnotation(ACCAnnotation.class, "worker") ) {
							outermostloop = false;
							break;
						}
						tt = tt.getParent();
					}
					if( outermostloop ) {
						List<ForLoop> nestedWLoops = AnalysisTools.findDirectlyNestedLoopsWithClause(wLoop, "worker");
						List<Expression> tExpList = new ArrayList<Expression>(3);
						for( ForLoop tLoop : nestedWLoops ) {
							ACCAnnotation iAnnot = tLoop.getAnnotation(ACCAnnotation.class, "worker");
							Expression tExp = iAnnot.get("worker");
							if( tExp != null ) {
								tExpList.add(tExp);
							}
						}
						if( nestedWLoops.size() != tExpList.size() ) {
							Tools.exit("[ERROR in CompRegionConfAnalysis.calKernelLoopConfiguration()] worker clause does not have" +
									"an argument, which should have been added either by a user or by a compiler; exit\n" + 
									wLoop + "\n");
						} else {
							ACCAnnotation iAnnot = wLoop.getAnnotation(ACCAnnotation.class, "internal");
							if( iAnnot == null ) {
								iAnnot = new ACCAnnotation("internal", "_directive");
								iAnnot.setSkipPrint(true);
								wLoop.annotate(iAnnot);
							}
							List<Expression> rExpList = new ArrayList<Expression>(tExpList.size());
							for( int i=tExpList.size()-1; i>=0; i--) {
								rExpList.add(tExpList.get(i).clone());
							}
							iAnnot.put("workerconf", rExpList);
							Expression totalNumWorkers = null;
							for( Expression exp : rExpList ) {
								if( totalNumWorkers == null ) {
									totalNumWorkers = exp.clone();
								} else {
									totalNumWorkers = Symbolic.multiply(totalNumWorkers, exp.clone());
								}
							}
							iAnnot.put("totalnumworkers", totalNumWorkers);
							if( totalNumWorkers instanceof IntegerLiteral ) {
								long tNumWorkers = ((IntegerLiteral)totalNumWorkers).getValue();
								if( tNumWorkers > maxNumWorkers ) {
									Procedure pProc = IRTools.getParentProcedure(at);
									if( (OPENARC_ARCH == 0) && (maxNumWorkers == 512) && (tNumWorkers <= 1024) ) {
										PrintTools.println("\n[WARNING in CompRegionConfAnalsys()] total number of workers (" + tNumWorkers
												+ ") in the following worker loop seems to be bigger than the allowed limit (" + maxNumWorkers
												+ ")\nEnclosing Procedure: " + pProc.getSymbolName() + "\nEnclosing ACCAnnotation: " + cAnnot + "\n", 0);
										PrintTools.println("If the compute capability of the target CUDA device is 2.x, this should be OK; " +
												"otherwise, user should reduce the total number of workers manually.", 0);
									} else {
										Tools.exit("[ERROR in CompRegionConfAnalsys()] total number of workers (" + tNumWorkers
												+ ") in the following worker loop is bigger than the allowed limit (" + maxNumWorkers
												+ ")\nEnclosing Procedure: " + pProc.getSymbolName() + "\nEnclosing ACCAnnotation: " 
												+ cAnnot + "\nTo increase the maximum number of workers, use maxNumWorkers option.\n");
									}
								}
							}
						}
					}
				}
			}
		}
	}
	
	/**Calculate number of gangs and workers for each parallel region if missing.
	 * For now, num_gangs is calculated by multiplying all inner gang loop sizes (physically 1D mapping).
	 * If any of inner gang loop contains worker clause, the aggregated iteration space size is
	 * divided by the num_workers.
	 * At the end of this analysis, (1) each gang loop contains its iteration size ("iterspace") in an internal annotation 
	 * and (2) num_gangs(num_gangs) is added to the parallel regions if missed.
	 * 
	 * @param parallelRegions
	 */
	private void calParallelRegionConfConfiguration( List<ACCAnnotation> parallelRegions ) {
		//Handle parallel regions.
		for( ACCAnnotation cAnnot : parallelRegions ) {
			Annotatable at = cAnnot.getAnnotatable();
			Expression num_workers = null;
			if( cAnnot.containsKey("num_workers") ) {
				num_workers = ((Expression)cAnnot.get("num_workers")).clone();
			} else {
				List<ACCAnnotation> workerAnnots = AnalysisTools.ipCollectPragmas(at, ACCAnnotation.class, "worker", null);
				if( (workerAnnots == null) || (workerAnnots.isEmpty()) ) {
					num_workers = new IntegerLiteral(1);
				} else {
					num_workers = new IntegerLiteral(defaultNumWorkers);
				}
				cAnnot.put("num_workers", num_workers.clone());
			}
			List<ForLoop> outermostGangLoops = new ArrayList<ForLoop>();
			List<ACCAnnotation> gangAnnots = AnalysisTools.ipCollectPragmas(at, ACCAnnotation.class, "gang", null);
			if( (gangAnnots == null) || (gangAnnots.isEmpty()) ) {
				//Commented out
				//Procedure pProc = IRTools.getParentProcedure(at);
				//Tools.exit("[ERROR] Internal error in CompRegionConfAnalysis(); no gang clause is found in a parallel region.\n" +
				//		"Enclosing procedure: " + pProc.getSymbolName() + "\nOpenACC annotation: " + cAnnot);
				if( !cAnnot.containsKey("num_gangs") ) {
					cAnnot.put("num_gangs", new IntegerLiteral(1));
				}
			} else {
				//Handle gang loops in the parallel region.
				for( ACCAnnotation gAnnot : gangAnnots ) {
					ForLoop gLoop = (ForLoop)gAnnot.getAnnotatable();
					//Number of gangs is not explicitly specified.
					Expression iterspace = calIterSpace(gLoop, at, gAnnot);
					//Store iteration space size into an internal annotation.
					Annotation iAnnot = gLoop.getAnnotation(ACCAnnotation.class, "internal");
					if( iAnnot == null ) {
						iAnnot = new ACCAnnotation("internal", "_directive");
						iAnnot.setSkipPrint(true);
						gLoop.annotate(iAnnot);
					}
					iAnnot.put("iterspace", iterspace);
					//Store outermost gang loops into outermostGangLoops list.
					boolean isInnerGangLoop = false;
					Traversable tt = gLoop.getParent();
					while (tt != null ) {
						if( (tt instanceof Annotatable) && ((Annotatable)tt).containsAnnotation(ACCAnnotation.class, "gang") ) {
							isInnerGangLoop = true;
							break;
						} else {
							tt = tt.getParent();
						}
					}
					if( !isInnerGangLoop ) {
						outermostGangLoops.add(gLoop);
					}
				}
				Expression num_gangs = null;
				if( !cAnnot.containsKey("num_gangs") ) {
					//Calculate num_gangs for each outermost gang loop.
					//For now, num_gangs is calculated by multiplying all inner gang loop sizes (physically 1D mapping).
					//If any of inner gang loop contains worker clause, the aggregated iteration space size is
					//divided by the num_workers.
					for( ForLoop oGLoop : outermostGangLoops ) {
						Expression iterspace = null;
						gangAnnots = IRTools.collectPragmas(oGLoop, ACCAnnotation.class, "gang");
						boolean containWorker = false;
						for( ACCAnnotation gAnnot : gangAnnots ) {
							ForLoop gLoop = (ForLoop)gAnnot.getAnnotatable();
							Annotation iAnnot = gLoop.getAnnotation(ACCAnnotation.class, "iterspace");
							if( iAnnot != null ) {
								Expression tSize = (Expression)iAnnot.get("iterspace");
								if( iterspace == null ) {
									iterspace = tSize;
								} else {
									iterspace = Symbolic.multiply(iterspace, tSize);
								}
							}
							if( gLoop.containsAnnotation(ACCAnnotation.class, "worker") ) {
								containWorker = true;
							}
						}
						Expression tnum_gangs = null;
						if( containWorker ) {
							tnum_gangs = Symbolic.divide(iterspace, num_workers);
							if( (iterspace instanceof IntegerLiteral) && (num_workers instanceof IntegerLiteral) ) {
								int gsize = (int)Math.ceil( ((double)((IntegerLiteral)iterspace).getValue())/
										((double)((IntegerLiteral)num_workers).getValue()) );
								tnum_gangs = new IntegerLiteral(gsize);
							} else {
								if( (num_workers instanceof IntegerLiteral) && (((IntegerLiteral)num_workers).getValue() == 1) ) {
									tnum_gangs = iterspace.clone();
								} else {
									List<Specifier> specs = new ArrayList<Specifier>(1);
									specs.add(Specifier.FLOAT);
									Expression floatsize1 = new Typecast(specs, iterspace.clone());
									Expression floatsize2;
									if( num_workers instanceof IntegerLiteral ) {
										floatsize2 = new FloatLiteral((double)((IntegerLiteral)num_workers.clone()).getValue(), "F");
									} else {
										specs = new ArrayList<Specifier>(1);
										specs.add(Specifier.FLOAT);
										floatsize2 = new Typecast(specs, num_workers.clone());
									}
									FunctionCall ceilfunc = new FunctionCall(new NameID("ceil"));
									Expression argExp = Symbolic.divide(floatsize1, floatsize2); 
									ceilfunc.addArgument(argExp);
									//numGangs = ceilfunc;
									specs = new ArrayList<Specifier>(1);
									specs.add(Specifier.INT);
									tnum_gangs = new Typecast(specs, ceilfunc);
								}
							}
						} else {
							tnum_gangs = iterspace;
						}
						//Find the maximum gang loop size.
						if( num_gangs == null ) {
							num_gangs = tnum_gangs;
						} else {
							num_gangs = Symbolic.simplify( 
									new MinMaxExpression(false, num_gangs, tnum_gangs));
						}
					}
					if( num_gangs != null ) {
						// If maxNumGangs is set, the actual number of gangs should be min(num_gangs, maxNumGangs).
						if( maxNumGangs > 0 ) {
							num_gangs = Symbolic.simplify( 
									new MinMaxExpression(true, num_gangs, new IntegerLiteral(maxNumGangs)));
						}
						///////////////////////////////////////////////////////////////////////////////
						// If symplified max expression has more than two children (ex: max(a,b,c)), //
						// convert it to max expression with two children (ex: max(max(a,b),c).      //
						///////////////////////////////////////////////////////////////////////////////
						List<MinMaxExpression> maxExpList = IRTools.getExpressionsOfType(num_gangs, MinMaxExpression.class);
						while( !maxExpList.isEmpty() ) {
							Expression oldMaxExp = maxExpList.remove(maxExpList.size()-1);
							Expression maxExp = oldMaxExp;
							List<Traversable> children = (List<Traversable>)maxExp.getChildren();
							if( children.size() > 2 ) {
								maxExp = ((Expression)children.get(0)).clone();
								for( int i=1; i<children.size(); i++ ) {
									maxExp = new MinMaxExpression(false, maxExp, ((Expression)children.get(i)).clone());
								}
								oldMaxExp.swapWith(maxExp);
							}
						}
					} else {
						num_gangs = new IntegerLiteral(1);
					}
					//Put the maximum gang loop size as num_gang clause value.
					cAnnot.put("num_gangs", num_gangs);
					if( (num_gangs instanceof IntegerLiteral) && 
							(((IntegerLiteral)num_gangs).getValue() > systemMaxNumGangs) ) {
						Procedure pProc = IRTools.getParentProcedure(at);
/*						Tools.exit("[ERROR in CompRegionConfAnalysis.calParallelRegionConfiguration()] the number of gangs (" +
								num_gangs.toString() + ") in the following compute region is bigger than the system-supported maximum size (" + 
								systemMaxNumGangs + "); either put explicit smaller gang size or increase worker size to meet the system limt; exit\n" + 
								"Enclosing procedure: " + pProc.getSymbolName() + "\nOpenACC annotation: " + cAnnot + "\n");*/
						PrintTools.println("[WARNING in CompRegionConfAnalysis.calParallelRegionConfiguration()] the number of gangs (" +
								num_gangs.toString() + ") in the following compute region is bigger than the system-supported maximum size (" + 
								systemMaxNumGangs + "); the system-supported maximum size will be used for the number of gang.\n" + 
								"Enclosing procedure: " + pProc.getSymbolName() + "\nOpenACC annotation: " + cAnnot + "\n", 0);
						num_gangs = new IntegerLiteral(systemMaxNumGangs);
						cAnnot.put("num_gangs", num_gangs);
					}
				}
			}
		}
	}
	
	/**
	 * At the end of this analysis, (1) each seq kernel loop contains its gang configuration ("gangconf") and worker configuration
	 * ("workerconf") in an internal annotation are added to the parallel regions if missed.
	 * 
	 * @param seqKernelRegions
	 */
	private void calSeqKernelRegionConfConfiguration( List<ACCAnnotation> seqKernelRegions ) {
		//Handle seq kernel regions.
		for( ACCAnnotation cAnnot : seqKernelRegions ) {
			Annotatable at = cAnnot.getAnnotatable();
			Annotation iAnnot = at.getAnnotation(ACCAnnotation.class, "internal");
			if( iAnnot == null ) {
				iAnnot = new ACCAnnotation("internal", "_directive");
				iAnnot.setSkipPrint(true);
				at.annotate(iAnnot);
			}
			List<ForLoop> outermostGangLoops = new ArrayList<ForLoop>();
			List<ACCAnnotation> gangAnnots = AnalysisTools.ipCollectPragmas(at, ACCAnnotation.class, "gang", null);
			if( gangAnnots == null ) {
					Procedure pProc = IRTools.getParentProcedure(at);
					Tools.exit("[ERROR] Internal error in CompRegionConfAnalysis(); no gang clause is found in a kernel region.\n" +
							"Enclosing procedure: " + pProc.getSymbolName() + "\nOpenACC annotation: " + cAnnot + "\n");
			} else {
				for( ACCAnnotation gAnnot : gangAnnots ) {
					ForLoop gLoop = (ForLoop)gAnnot.getAnnotatable();
					Traversable tt = gLoop.getParent();
					boolean outermostloop = true;
					while( tt != null ) {
						if( (tt instanceof Annotatable) && ((Annotatable)tt).containsAnnotation(ACCAnnotation.class, "gang") ) {
							outermostloop = false;
							break;
						}
						tt = tt.getParent();
					}
					if( outermostloop ) {
						outermostGangLoops.add(gLoop);
					}
				}
				Expression tGangdim = null;
				Expression tTotalNumGangs = null;
				List<Expression> tGangConfList = new LinkedList<Expression>();
				for( ForLoop oLoop : outermostGangLoops ) {
					ACCAnnotation tAnnot = oLoop.getAnnotation(ACCAnnotation.class, "internal");
					List<Expression> gangConfList = tAnnot.get("gangconf");
					Expression totalNumGangs = tAnnot.get("totalnumgangs");
					Expression gangdim = tAnnot.get("gangdim");
					if( tGangdim == null ) {
						tTotalNumGangs = totalNumGangs;
						tGangdim = gangdim;
						for( Expression dim : gangConfList ) {
							tGangConfList.add(dim.clone());
						}
					} else if( tGangdim != gangdim ) {
						Procedure pProc = IRTools.getParentProcedure(at);
						Tools.exit("[ERROR] if seq kernel loop contains multiple inner gang loops, each gang loop should have the" +
								" same dimension; exit\nEnclosing procedure: " + pProc.getSymbolName() + "\nOpenACC annotation: " 
								+ cAnnot +"\n" );
					} else {
						int i = 0;
						for( Expression dim : gangConfList ) {
							Expression num_gangs = tGangConfList.get(i);
							num_gangs = Symbolic.simplify( 
									new MinMaxExpression(false, num_gangs, dim));
							tGangConfList.set(i, num_gangs);
							i++;
						}
						tTotalNumGangs = Symbolic.simplify( 
									new MinMaxExpression(false, tTotalNumGangs, totalNumGangs));
					}
				}
				iAnnot.put("gangconf", tGangConfList);
				iAnnot.put("totalnumgangs", tTotalNumGangs);
				
				Expression tTotalNumWorkers = null;
				List<ForLoop> outermostWorkerLoops = new ArrayList<ForLoop>();
				List<ACCAnnotation> workerAnnots = AnalysisTools.ipCollectPragmas(at, ACCAnnotation.class, "worker", null);
				for( ACCAnnotation wAnnot : workerAnnots ) {
					ForLoop wLoop = (ForLoop)wAnnot.getAnnotatable();
					Traversable tt = wLoop.getParent();
					boolean outermostloop = true;
					while( tt != null ) {
						if( (tt instanceof Annotatable) && ((Annotatable)tt).containsAnnotation(ACCAnnotation.class, "worker") ) {
							outermostloop = false;
							break;
						}
						tt = tt.getParent();
					}
					if( outermostloop ) {
						outermostWorkerLoops.add(wLoop);
					}
				}
				Expression tworkerdim = null;
				List<Expression> tWorkerConfList = new LinkedList<Expression>();
				for( ForLoop oLoop : outermostWorkerLoops ) {
					ACCAnnotation tAnnot = oLoop.getAnnotation(ACCAnnotation.class, "internal");
					List<Expression> workerConfList = tAnnot.get("workerconf");
					Expression totalNumWorkers = tAnnot.get("totalnumworkers");
					Expression workerdim = tAnnot.get("workerdim");
					if( tworkerdim == null ) {
						tTotalNumWorkers = totalNumWorkers;
						tworkerdim = workerdim;
						for( Expression dim : workerConfList ) {
							tWorkerConfList.add(dim.clone());
						}
					} else if( tworkerdim != workerdim ) {
						Procedure pProc = IRTools.getParentProcedure(at);
						Tools.exit("[ERROR] if seq kernel loop contains multiple inner worker loops, each worker loop should have the" +
								" same dimension; exit\nEnclosing procedure: " + pProc.getSymbolName() + "\nOpenACC annotation: " 
								+ cAnnot +"\n" );
					} else {
						int i = 0;
						for( Expression dim : workerConfList ) {
							Expression num_workers = tWorkerConfList.get(i);
							num_workers = Symbolic.simplify( 
									new MinMaxExpression(false, num_workers, dim));
							tWorkerConfList.set(i, num_workers);
							i++;
						}
						tTotalNumWorkers = Symbolic.simplify( 
									new MinMaxExpression(false, tTotalNumWorkers, totalNumWorkers));
					}
				}
				iAnnot.put("workerconf", tWorkerConfList);
				iAnnot.put("totalnumworkers", tTotalNumWorkers);
			}
		}
	}
	
	private Expression calIterSpace(ForLoop gLoop, Annotatable at, ACCAnnotation gAnnot) {
						List<ForLoop> gangLoops = new ArrayList<ForLoop>();
						//gangLoops contains loops constituting gang loop iteration space.
						gangLoops.add(gLoop);
						ACCAnnotation collapseAnnot = gLoop.getAnnotation(ACCAnnotation.class, "collapse");
						if( collapseAnnot != null ) {
							int collapseLevel = (int)((IntegerLiteral)collapseAnnot.get("collapse")).getValue();
							if( collapseLevel > 1 ) {
								if( !AnalysisTools.extendedPerfectlyNestedLoopChecking(gLoop, collapseLevel, gangLoops, null)) {
									Procedure pProc = IRTools.getParentProcedure(at);
									Tools.exit("[ERROR] Internal error in CompRegionConfAnalysis(); Collapse clause is applicable only to perfectly nested loops.\n" +
											"Enclosing procedure: " + pProc.getSymbolName() + "\nOpenACC annotation: " + gAnnot);
								}
							}
						}
						//Calculate the maximumm size of gang loop iteration spaces, which will be the number of gangs if no worker clause exists. 
						//If worker clause exists, number of gangs * number of workers will be maximum size.
						Expression iterspace = null;
						for( ForLoop ploop : gangLoops ) {
							// check for a canonical loop
							if ( !LoopTools.isCanonical(ploop) ) {
								Procedure pProc = IRTools.getParentProcedure(at);
								Tools.exit("[ERROR in CompRegionConfAnalysis()] a worksharing loop is not " +
										"a canonical loop; compiler can not determine iteration space of " +
										"the following loop: \nEnclosing procedure: " + pProc.getSymbolName() + "\nOpenACC Annotation: " + gAnnot );
							}
							// check whether loop stride is 1.
							Expression incr = LoopTools.getIncrementExpression(ploop);
							boolean increasingOrder = true;
							if( incr instanceof IntegerLiteral ) {
								long IntIncr = ((IntegerLiteral)incr).getValue();
								if( IntIncr < 0 ) {
									increasingOrder = false;
								}
								if( Math.abs(IntIncr) != 1 ) {
									Procedure pProc = IRTools.getParentProcedure(at);
									Tools.exit("[ERROR in CompRegionConfAnalysis()] A worksharing loop with a stride > 1 is found;" +
											"current implmentation can not handle the following loop: \nEnclosing procedure: " + pProc.getSymbolName() + 
											"\nOpenACC Annotation: " + gAnnot);
								}
							} else {
								Procedure pProc = IRTools.getParentProcedure(at);
								Tools.exit("[ERROR in CompRegionConfAnalysis()] The stride of a worksharing loop is not constant;" +
										"current implmentation can not handle the following loop: \nEnclosing procedure: " + pProc.getSymbolName() + 
										"\nOpenACC Annotation: " + gAnnot);

							}
							// identify the loop index variable 
							//Expression ivar = LoopTools.getIndexVariable(ploop);
							Expression lb = LoopTools.getLowerBoundExpression(ploop);
							Expression ub = LoopTools.getUpperBoundExpression(ploop);
							Expression tSize = null;
							if( increasingOrder ) {
								tSize = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
							} else {
								tSize = Symbolic.add(Symbolic.subtract(lb,ub),new IntegerLiteral(1));
							}
							if( iterspace == null ) {
								iterspace = tSize;
							} else {
								iterspace = Symbolic.multiply(iterspace, tSize);
							}
						}
						return iterspace;
	}
}
