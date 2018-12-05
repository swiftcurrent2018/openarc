/**
 * @author Seyong Lee <lees2@ornl.gov> and Jacob Lambert <jlambert@cs.uoregon.edu>
 *         Future Technologies Group, Oak Ridge National Laboratory 
 */
package openacc.transforms;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import openacc.analysis.ACCAnalysis;
import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;
import openacc.hir.ACCAnnotation;
import openacc.hir.ARCAnnotation;
import cetus.analysis.LoopTools;
import cetus.hir.Annotation;
import cetus.hir.AccessSymbol;
import cetus.hir.Annotatable;
import cetus.hir.ArrayAccess;
import cetus.hir.ArraySpecifier;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.ConditionalExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.DFIterator;
import cetus.hir.Expression;
import cetus.hir.SimpleExpression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.IDExpression;
import cetus.hir.IRTools;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.MinMaxExpression;
import cetus.hir.NameID;
import cetus.hir.PragmaAnnotation;
import cetus.hir.PrintTools;
import cetus.hir.Program;
import cetus.hir.PseudoSymbol;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTools;
import cetus.hir.Symbolic;
import cetus.hir.Tools;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import cetus.transforms.TransformPass;
import cetus.hir.Traversable;

/**
 * @author f6l
 *
 */
public class SlidingWindowTransformation extends TransformPass {

  /**
   * @param program
   */
  public SlidingWindowTransformation(Program program) {
    super(program);
  }

  /* (non-Javadoc)
   * @see cetus.transforms.TransformPass#getPassName()
   */
  @Override
    public String getPassName() {
      return "[SlidingWindowTransformation]";
    }

  /* (non-Javadoc)
   * @see cetus.transforms.TransformPass#start()
   */
  @Override
    public void start() {

      /* Iterate over each window annotation */
      List<ARCAnnotation>  windowAnnots = IRTools.collectPragmas(program, ARCAnnotation.class, "window");
      for( ARCAnnotation tAnnot : windowAnnots ) {
        Annotatable wAt = tAnnot.getAnnotatable();
        ACCAnalysis.updateSymbolsInACCAnnotations(wAt, null);

        if ( !(wAt instanceof ForLoop) ) {
          Tools.exit("[ERROR in SlidingWindowTransformation] Sliding-window transformation is applicable only "
              + "to a for-loop, but the following transform window directive is attached to a non-loop statement; exit!\n"
              + "ARC annotation: " + tAnnot + AnalysisTools.getEnclosingContext(wAt));
        }

        ForLoop fLoop = (ForLoop)wAt;
        CompoundStatement fBody = null;

        int collapseLevel = 0;
        ArrayList<Expression> indexVariables = new ArrayList<Expression>();
        ArrayList<Expression> iterspaceList = new ArrayList<Expression>();
        List<ForLoop> nestedLoops = new ArrayList<ForLoop>();

        if( fLoop.containsAnnotation(ACCAnnotation.class, "collapse") ) {
          ACCAnnotation collapseAnnot = fLoop.getAnnotation(ACCAnnotation.class, "collapse");
          collapseLevel = (int)((IntegerLiteral)collapseAnnot.get("collapse")).getValue();
          nestedLoops = new ArrayList<ForLoop>();
          AnalysisTools.extendedPerfectlyNestedLoopChecking(fLoop, collapseLevel, nestedLoops, null);

          // collapse index variables
          indexVariables.add(LoopTools.getIndexVariable(fLoop));

          // iter space
          Expression lb = LoopTools.getLowerBoundExpression(fLoop);
          Expression ub = LoopTools.getUpperBoundExpression(fLoop);
          Expression itrSpace = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
          iterspaceList.add(itrSpace);

          for( ForLoop currLoop : nestedLoops ) {
            // collapse index variables
            indexVariables.add(LoopTools.getIndexVariable(currLoop));

            // iter space
            lb = LoopTools.getLowerBoundExpression(currLoop);
            ub = LoopTools.getUpperBoundExpression(currLoop);
            itrSpace = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
            iterspaceList.add(itrSpace);
          }

          /* Replace nested loops with single collapsed loop */
          CollapseTransformation.collapseLoopHead(fLoop, true);

          fBody = (CompoundStatement)fLoop.getBody();
        } 
        else if( fLoop.containsAnnotation(ACCAnnotation.class, "gang") ) {
          nestedLoops = AnalysisTools.findDirectlyNestedLoopsWithClause(fLoop, "gang");
          ForLoop innerLoop = nestedLoops.get(nestedLoops.size()-1);
          fBody = (CompoundStatement)innerLoop.getBody();
        } 
        else {
          nestedLoops.add(fLoop);
          fBody = (CompoundStatement)fLoop.getBody();
        }

        Statement firstExeStatement = AnalysisTools.getFirstExecutableStatement(fBody);
        Expression LB = LoopTools.getLowerBoundExpressionNS(fLoop);
        Expression UB = LoopTools.getUpperBoundExpressionNS(fLoop);
        Expression iter = LoopTools.getIndexVariable(fLoop);
        Expression condition = fLoop.getCondition();
        Expression step = LoopTools.getIncrementExpression(fLoop);

        if( !(step instanceof IntegerLiteral) ) {
          PrintTools.println("[WARNING in SlidingWindowTransformation] the OpenARC transform window directive can be applied "
              + "only to a for-loop with constant step size; the following directive will be skipped!\n"
              + "ARC annotation: " + tAnnot + AnalysisTools.getEnclosingContext(wAt), 0);
          continue;
        } 

        long stepValue =  ((IntegerLiteral)step).getValue();
        if( stepValue != 1 ) {
          PrintTools.println("[WARNING in SlidingWindowTransformation] the OpenARC transform window directive can be applied "
              + "only to a for-loop with step size of 1; the following directive will be skipped!\n"
              + "ARC annotation: " + tAnnot + AnalysisTools.getEnclosingContext(wAt), 0);
          continue;
        }

        /* Parse annotation parameters */
        List<Object> windowarglist = tAnnot.get("window");
        if( (windowarglist == null) || (windowarglist.size() != 2) ) {
          Tools.exit("[ERROR in SlidingWindowTransformation] incorrect number of arguments for window clause "
              + "of the following directive; exit!\n"
              + "ARC annotation: " + tAnnot + AnalysisTools.getEnclosingContext(wAt));
        }

        // input array
        SubArray input_subarray = (SubArray)windowarglist.get(0);
        Expression input_array = input_subarray.getArrayName();
        Symbol iSym = AnalysisTools.subarrayToSymbol(input_subarray, false);

        if( input_subarray.getArrayDimension() <= 0 ) {
          Tools.exit("[ERROR in SlidingWindowTransformation] "
              + "Input array dimensions and length must be provided, ex. "
              + "\"window(input[0:N], output)\" "
              + "in the following directive; exit!\n"
              + "ARC annotation: " + tAnnot + AnalysisTools.getEnclosingContext(wAt));
        }

        // [DEBUG] for now, we assume that the input array is 1D.
        Expression iarray_LB = input_subarray.getStartIndices().get(0);
        Expression iarray_Length = input_subarray.getLengths().get(0);
        if( ( iarray_LB == null ) || ( iarray_Length == null ) ) {
          Tools.exit("[ERROR in SlidingWindowTransformation] dimension information "
              + "of the input array argument for window clause "
              + "of the following directive is missing; exit!\n"
              + "ARC annotation: " + tAnnot + AnalysisTools.getEnclosingContext(wAt));
        }
        
        // output array
        SubArray output_subarray = (SubArray)windowarglist.get(1);
        Expression output_array = output_subarray.getArrayName();
        Symbol oSym = AnalysisTools.subarrayToSymbol(output_subarray, false);

        /* Record loop unroll factor */
        int unroll_factor = 1;

        List<PragmaAnnotation> pragmaList = fLoop.getAnnotations(PragmaAnnotation.class);
        fLoop.removeAnnotations(PragmaAnnotation.class);

        for (PragmaAnnotation pragmaAnnot : pragmaList) {
          String keySet[] = pragmaAnnot.toString().split(" ");

          int index = 0;
          boolean unroll_bool = false;
          for( String tKey : keySet ) {
            // get unroll value
            if (tKey.equals("unroll"))  {
              if (index == keySet.length - 1)
                Tools.exit("[ERROR in SlidingWindowTransformation: Annotation #pragma unroll must have" +
                    " an associated unroll factor");
              unroll_factor = Integer.parseInt(keySet[index + 1]);
              unroll_bool = true;
            }
            index++;
          }

          // replace non-unroll pragmas
          if (!unroll_bool)
            fLoop.annotate(pragmaAnnot);

        }

        if (input_array.equals(output_array) && unroll_factor > 1) {
          Tools.exit("[ERROR in SlidingWindowTransformation] Loop unrolling not supported when input array and output array are the same. Please remove unrolling clause or change array.");
        }
        
        // verify that the unroll factor divides either the input size (1D)  or the column size (2D)
        // DEBUG: Unfinished
        //  if ( (unroll_factor % input_size) != 0) 
        //    Tools.Exit("[ERROR in SlidingWindoTransformation] Unroll factor must be a factor of the iteration space 

        /* Create a compound statement that will enclose the transformed statements. */
        CompoundStatement cStmt = new CompoundStatement();

        /* Add #pragma ivdep to independent loops */
        boolean ivdep = false;
        if( fLoop.containsAnnotation(ACCAnnotation.class, "independent") ) {
          Annotation ivPragma = new PragmaAnnotation("ivdep"); 
          fLoop.annotateBefore(ivPragma.clone());
        }

        /* Generate iteration expression */
        /* For 1D loops, this is the iteration (iter)
         * For higher-dimensional loops, this is expanded (r*C + c for 2D) */
        Expression iterExp = iter.clone();
        if (collapseLevel > 1) {
          for (int i = 0; i < collapseLevel; ++i) {
            Expression nExp = indexVariables.get(i).clone();
            for (int j = i + 1; j < collapseLevel; ++j) {
              nExp = new BinaryExpression(nExp, BinaryOperator.MULTIPLY, iterspaceList.get(j).clone());
            }
            if (i == 0) 
              iterExp = nExp.clone(); 
            else
              iterExp = new BinaryExpression(iterExp, BinaryOperator.ADD, nExp.clone());
          }
        }

        /* Calculate Window Size and Target element */ 
        // Unroll loops in the body 
        Statement fBody_unrolled = FPGASpecificTools.UnrollLoopsInRegion(fBody.clone());

        // collect window index expressions
        List<Expression> windowExpList = new ArrayList<Expression>();
        List<ArrayAccess> arrayAccessList = IRTools.getExpressionsOfType(fBody_unrolled, ArrayAccess.class);
        if( arrayAccessList == null ) {
          Tools.exit("[ERROR in SlidingWindowTransformation]: No array accesses in window loop");
        }

        for( ArrayAccess aAccess : arrayAccessList ) {
          if( aAccess.getArrayName().equals(input_array) ) {
            List<Expression> window_indices = aAccess.getIndices();
            for( Expression wExp : window_indices) {
              Expression nExp = wExp.clone();

              // remove iteration index
              if (nExp.equals(iterExp)) {
                nExp = new IntegerLiteral(0);
              } else if (IRTools.containsExpression(nExp, iterExp)) {
                IRTools.replaceAll(nExp, iterExp, new IntegerLiteral(0)); 
              } else {
                nExp = new BinaryExpression(nExp, BinaryOperator.SUBTRACT, iterExp.clone());
              }

              nExp = Symbolic.simplify(nExp);

              windowExpList.add(nExp);
            }
          }
        }

        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
        for (Expression wExp : windowExpList) {

          // verify that sliding-window access is constant
          if (wExp.getClass() != IntegerLiteral.class) {
            Tools.exit("[ERROR in SlidingWindowTransformation] The following input array index expressions "  + 
                "cannot be resolved to a constant:\n  " + wExp.toString()); 
          }

          long wVal = ((IntegerLiteral) wExp).getValue();

          if (wVal > max) max = (int) wVal;
          if (wVal < min) min = (int) wVal;
        }

        if (max < 0) Tools.exit("max < 0");
        if (min > 0) Tools.exit("min > 0");

        int nbd_size_int = max - min + 1;
        int target_int = 0 - min;

        //System.out.println("max: " + max);
        //System.out.println("min: " + min);
        //System.out.println("nbd_size: " + nbd_size_int); 
        //System.out.println("target: " + target_int); 

        Expression target = new IntegerLiteral(target_int);
        Expression nbd_size = new IntegerLiteral(nbd_size_int);

        /* Define unroll pragma */
        PragmaAnnotation unrollpragma = new PragmaAnnotation("unroll");

        ////////////////////////////////////////
        // Base transformation (no unrolling) //
        ////////////////////////////////////////
        if (unroll_factor == 1) {

          /* Calculate window array size */
          Expression sw_size = Symbolic.simplify((Expression) new IntegerLiteral(nbd_size_int));

          /* Create and 0-initialize sliding window array */
          List<Specifier> removeSpecs = new ArrayList<Specifier>();
          removeSpecs.add(Specifier.STATIC);
          removeSpecs.add(Specifier.CONST);
          removeSpecs.add(Specifier.EXTERN);
          List<Specifier> typeSpecs = new ArrayList<Specifier>();
          Symbol IRSym = iSym;
          if( iSym instanceof PseudoSymbol ) {
            IRSym = ((PseudoSymbol)iSym).getIRSymbol();
            Symbol tSym = iSym;
            while( tSym instanceof AccessSymbol ) {
              tSym = ((AccessSymbol)tSym).getMemberSymbol();
            }
            typeSpecs.addAll(((VariableDeclaration)tSym.getDeclaration()).getSpecifiers());
          } else {
            typeSpecs.addAll(((VariableDeclaration)IRSym.getDeclaration()).getSpecifiers());
          }
          typeSpecs.removeAll(removeSpecs);
          String symNameBase = null;
          if( iSym instanceof AccessSymbol) {
            symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)iSym);
          } else {
            symNameBase = iSym.getSymbolName();
          }
          String windowSymName = "sw__" + symNameBase;
          List tindices = new LinkedList();

          tindices.add(sw_size.clone());

          ArraySpecifier aspec = new ArraySpecifier(tindices);
          List tailSpecs = new ArrayList(1);
          tailSpecs.add(aspec);
          VariableDeclarator wSym_declarator = new VariableDeclarator(new NameID(windowSymName), tailSpecs);
          Identifier sw = new Identifier(wSym_declarator);
          VariableDeclaration wSym_decl = new VariableDeclaration(typeSpecs,
              wSym_declarator);
          cStmt.addDeclaration(wSym_decl);

          /* Create a new temp index variable. */
          Symbol indexSym = SymbolTools.getSymbolOf(iter);
          List<Specifier> typeSpecs2 = new ArrayList<Specifier>();
          typeSpecs2.addAll(indexSym.getTypeSpecifiers());
          symNameBase = indexSym.getSymbolName();
          Identifier nIndex = TransformTools.getTempScalar(cStmt, typeSpecs2, "w__" + symNameBase, 0);

          /* Initialize the sliding window array. */
          Expression expr1 = new AssignmentExpression(nIndex, AssignmentOperator.NORMAL,
              new IntegerLiteral(0));
          Statement expStmt = new ExpressionStatement(expr1);
          expr1 = new BinaryExpression((Identifier)nIndex.clone(), BinaryOperator.COMPARE_LT,
              sw_size.clone());
          Expression expr2 = new UnaryExpression(
              UnaryOperator.POST_INCREMENT, (Identifier)nIndex.clone());
          CompoundStatement forBody = new CompoundStatement();
          Expression expr3 = new AssignmentExpression(new ArrayAccess(sw.clone(), nIndex.clone()), AssignmentOperator.NORMAL,
              new IntegerLiteral(0)); 
          Statement bStmt = new ExpressionStatement(expr3);
          forBody.addStatement(bStmt);
          ForLoop wLoop = new ForLoop(expStmt, expr1, expr2, forBody);
          wLoop.annotate(unrollpragma.clone());
          cStmt.addStatement(wLoop);

          /* Initialize slidwing window by redefining loop header */ 
          /* NOTE: Omitted if TARGET == NBD_SIZE - 1*/

          // i = TARGET - (NBD_SIZE - 1);
          // NOTE: Same as -(read_offset)
          if (target_int != nbd_size_int - 1) {
            expr1 = new BinaryExpression(target.clone(), BinaryOperator.SUBTRACT, new IntegerLiteral(nbd_size_int - 1));
            expr1 = Symbolic.simplify(expr1);
            expr1 = new AssignmentExpression(iter.clone(), AssignmentOperator.NORMAL, expr1);

            Statement initStmt = new ExpressionStatement(expr1);
            fLoop.getInitialStatement().swapWith(initStmt);
          }


          /* Identify LHS accesses to input_array */
          /* NOTE: I think this may only be needed if input_array == output_array */

          List<AssignmentExpression> assignList = IRTools.getExpressionsOfType(fBody, AssignmentExpression.class);
          ArrayList<ArrayAccess> LHSarrayAccessList = new ArrayList<ArrayAccess>(); 
          ArrayList<Statement> LHSstatementList = new ArrayList<Statement>(); 

          if ( assignList != null) {
            for (AssignmentExpression assignExp : assignList) {
              if (assignExp.getLHS().getClass() == ArrayAccess.class) {
                ArrayAccess LHSaAccess = ((ArrayAccess) assignExp.getLHS());

                if (LHSaAccess.getArrayName().equals(input_array)) {
                  LHSarrayAccessList.add(LHSaAccess);

                }
              }
            }
          }

          /* replace iter with TARGET */
          /* NOTE: This is done either by                                             */
          /*  1. direct substitution      array[iter] -> array[TARGET], or            */  
          /*  2. addition/subtraction     array[ijk]  -> array[ijk - iter + TARGET]   */        

          // replacement in each input array access
          arrayAccessList = IRTools.getExpressionsOfType(fBody, ArrayAccess.class);
          if( arrayAccessList != null ) {
            for( ArrayAccess aAccess : arrayAccessList ) {
              if( aAccess.getArrayName().equals(input_array) ) {
                List<Expression> oldindices = aAccess.getIndices();
                List<Expression> newindices = new ArrayList<Expression>(oldindices.size());
                for( Expression oExp : oldindices ) {
                  Expression nExp = oExp.clone();

                  Expression nExpSim = Symbolic.simplify(nExp);
                  Expression iterExpSim = Symbolic.simplify(iterExp);

                  // handles input[i] case
                  if (nExp.equals(iterExp)) {
                    //System.out.println("Immediate MATCH");
                    nExp = target.clone();
                  }
                  else if (IRTools.containsExpression(nExp, iterExp)) {
                    //System.out.println("Replace MATCH");
                    IRTools.replaceAll(nExp, iterExp, target.clone());
                  }
                  else if (IRTools.containsExpression(nExpSim, iterExpSim)) {
                    //System.out.println("Simplify MATCH");
                    IRTools.replaceAll(nExpSim, iterExpSim, target.clone());

                    nExp = nExpSim;
                  }
                  else {
                    System.out.println("[WARNING in SlidingWindowTransformation] All input array index expressions " + 
                        "must be of the form:\n  required: " + iterExp.toString() + 
                        " + CONST\n  found: " + nExp.toString());
                    System.out.println(" Performing addition/subtraction instead of direct replacement\n");
                    nExp = new BinaryExpression(nExp, BinaryOperator.SUBTRACT, iterExp.clone());
                    nExp = new BinaryExpression(nExp, BinaryOperator.ADD, target.clone());
                  }

                  nExp = Symbolic.simplify(nExp);

                  newindices.add(nExp);
                }
                ArrayAccess nAccess = new ArrayAccess(sw.clone(), newindices);
                aAccess.swapWith(nAccess);
                aAccess.setParent(nAccess.getParent());
              }
            }
          }

          /* Reinsert LHS input_array accesses, updating their values based on the sliding window. */
          /* NOTE: I think this may only be needed if input_array == output_array */

          for (ArrayAccess LHSaAccess : LHSarrayAccessList) {
            // generate index expression list for RHS
            List<Expression> LHSindices = LHSaAccess.getIndices();
            List<Expression> RHSindices = new ArrayList<Expression>(LHSindices.size());
            for( Expression lhsExp : LHSindices ) {
              Expression nExp = lhsExp.clone();

              Expression nExpSim = Symbolic.simplify(nExp);
              Expression iterExpSim = Symbolic.simplify(iterExp);

              // handles input[i] case
              if (nExp.equals(iterExp)) {
                //System.out.println("Immediate MATCH");
                nExp = target.clone();
              }
              else if (IRTools.containsExpression(nExp, iterExp)) {
                //System.out.println("Replace MATCH");
                IRTools.replaceAll(nExp, iterExp, target.clone());
              }
              else if (IRTools.containsExpression(nExpSim, iterExpSim)) {
                //System.out.println("Simplify MATCH");
                IRTools.replaceAll(nExpSim, iterExpSim, target.clone());

                nExp = nExpSim;
              }
              else {
                System.out.println("[WARNING in SlidingWindowTransformation] All input array index expressions " + 
                    "must be of the form:\n  required: " + iterExp.toString() + 
                    " + CONST\n  found: " + nExp.toString());
                System.out.println(" Performing addition/subtraction instead of direct replacement\n");
                nExp = new BinaryExpression(nExp, BinaryOperator.SUBTRACT, iterExp.clone());
                nExp = new BinaryExpression(nExp, BinaryOperator.ADD, target.clone());
              }

              nExp = Symbolic.simplify(nExp);
              RHSindices.add(nExp);
            }
            ArrayAccess RHSaAccess = new ArrayAccess(sw.clone(), RHSindices);

            // generate assignment statement and LHS statement
            Statement stmt = new ExpressionStatement(new AssignmentExpression(LHSaAccess.clone(),
                  AssignmentOperator.NORMAL, RHSaAccess.clone()));

            // add statement to loop conditional
            CompoundStatement refCmpStmt = IRTools.getAncestorOfType(LHSaAccess, CompoundStatement.class);
            Statement refStmt = IRTools.getAncestorOfType(LHSaAccess, Statement.class); 

            refCmpStmt.addStatementAfter(refStmt, stmt);
          }

          /* Enclose loop body in a conditional to omit computation in intilization iterations */
          /* NOTE: Ommitted if TARGET == NBD_SIZE - 1 */
          CompoundStatement fBody_cond = new CompoundStatement();
          IfStatement ifStmt;

          if (target_int != nbd_size_int - 1) {
            Expression cond = new BinaryExpression(iter.clone(), BinaryOperator.COMPARE_GE, new IntegerLiteral(0));
            ifStmt = new IfStatement(cond, fBody.clone()); 

            fBody_cond.swapWith(fBody);
            fBody_cond.addStatement(ifStmt);
          }
          else {
            fBody_cond = fBody;
          }

          firstExeStatement = AnalysisTools.getFirstExecutableStatement(fBody_cond);


          /*  Slide the window each iteration */
          Expression biexp = null;

          // i = 0
          expr1 = new AssignmentExpression(nIndex.clone(), AssignmentOperator.NORMAL,
              new IntegerLiteral(0));
          expStmt = new ExpressionStatement(expr1);

          // i < nbd_size - 1
          expr1 = new BinaryExpression((Identifier)nIndex.clone(), BinaryOperator.COMPARE_LT, new IntegerLiteral(nbd_size_int - 1)); 

          // i ++
          expr2 = new UnaryExpression(
              UnaryOperator.POST_INCREMENT, (Identifier)nIndex.clone());
          forBody = new CompoundStatement();

          // sw[i] = sw[i + 1]
          expr3 = new AssignmentExpression(new ArrayAccess(sw.clone(), nIndex.clone()), AssignmentOperator.NORMAL,
              new ArrayAccess(sw.clone(), new BinaryExpression(nIndex.clone(), BinaryOperator.ADD, new IntegerLiteral(1) )));

          bStmt = new ExpressionStatement(expr3);
          forBody.addStatement(bStmt);
          wLoop = new ForLoop(expStmt, expr1, expr2, forBody);
          wLoop.annotate(unrollpragma.clone());

          if( firstExeStatement == null ) {
            fBody_cond.addStatement(wLoop);
          } else {
            fBody_cond.addStatementBefore(firstExeStatement, wLoop);
          }

          /* Read value from input array to sliding window */
          Identifier read_offset = null;

          read_offset = TransformTools.getTempScalar(fBody_cond, typeSpecs2, "read_offset__" + symNameBase, 0);

          // int read_offset = iter + (NBD_SIZE - 1) - TARGET;
          expr1 = new BinaryExpression(iter.clone(), BinaryOperator.ADD, new IntegerLiteral(nbd_size_int - 1));
          expr1 = new BinaryExpression(expr1, BinaryOperator.SUBTRACT, target.clone());

          expr1 = new AssignmentExpression(read_offset.clone(), AssignmentOperator.NORMAL, expr1);
          expStmt = new ExpressionStatement(expr1);

          if( firstExeStatement == null ) {
            fBody_cond.addStatement(expStmt);
          } else {
            fBody_cond.addStatementBefore(firstExeStatement, expStmt);
          }

          // (read_offset >= 0 && read_offset < N)
          expr1 = new BinaryExpression(read_offset.clone(), BinaryOperator.COMPARE_GE, new IntegerLiteral(0));
          expr2 = new BinaryExpression(read_offset.clone(), BinaryOperator.COMPARE_LT, iarray_Length.clone());
          expr3 = new BinaryExpression(expr1, BinaryOperator.LOGICAL_AND, expr2);

          //  sw[NBD_SIZE - 1] = input[read_offset];
          expr1 = new AssignmentExpression(new ArrayAccess(sw.clone(), new IntegerLiteral(nbd_size_int - 1)),
              AssignmentOperator.NORMAL,
              new ArrayAccess(input_array.clone(), read_offset.clone()));

          expStmt = new ExpressionStatement(expr1);
          ifStmt = new IfStatement(expr3, expStmt);

          //  sw[NBD_SIZE - 1] = 0 
          expr1 = new AssignmentExpression(new ArrayAccess(sw.clone(), new IntegerLiteral(nbd_size_int - 1)),
              AssignmentOperator.NORMAL, new IntegerLiteral(0) );

          expStmt = new ExpressionStatement(expr1);
          CompoundStatement elseBody = new CompoundStatement();
          elseBody.addStatement(expStmt);
          ifStmt.setElseStatement(elseBody);

          if( firstExeStatement == null ) {
            fBody_cond.addStatement(ifStmt);
          } else {
            fBody_cond.addStatementBefore(firstExeStatement, ifStmt);
          }

          /* Replace the target loop (fLoop) with the new, enclosing compound statement (cStmt), */
          /* and add the target loop into the compound statement.                                */
          cStmt.swapWith(fLoop);
          cStmt.addStatement(fLoop);

          /* Add initalizations before loop */
          /* DEBUG: Could this code be factored out into a function like collapseLoopTail(...) */
          if (collapseLevel > 1) {
            Expression accforIndex = indexVariables.get(0).clone();
            Expression indexExp = new BinaryExpression(target.clone(), BinaryOperator.SUBTRACT, new IntegerLiteral(nbd_size_int - 1));
            indexExp = Symbolic.simplify(indexExp);

            int i = collapseLevel-1;
            while (i > 0) {
              Expression tIndex = indexVariables.get(i).clone();

              if (i == (collapseLevel-1)) {
                expr1 = new BinaryExpression(indexExp.clone(), BinaryOperator.MODULUS,
                    iterspaceList.get(i).clone());
              } else {
                expr1 = new BinaryExpression(accforIndex.clone(), BinaryOperator.MODULUS,
                    iterspaceList.get(i).clone());
              }
              //[DEBUG] disabled for debugging
              //expr1 = Symbolic.simplify(expr1);

              if (target_int == nbd_size_int - 1) expr1 = new IntegerLiteral(0);

              Statement stmt = new ExpressionStatement(new AssignmentExpression(tIndex.clone(),
                    AssignmentOperator.NORMAL, expr1));

              cStmt.addStatementBefore(fLoop, stmt);

              if( i == (collapseLevel-1) ) {
                expr1 = new BinaryExpression(indexExp.clone(), BinaryOperator.DIVIDE,
                    iterspaceList.get(i).clone());
              } else {
                expr1 = new BinaryExpression(accforIndex.clone(), BinaryOperator.DIVIDE,
                    iterspaceList.get(i).clone());
              }
              //[DEBUG] disabled for debugging
              //expr1 = Symbolic.simplify(expr1);

              if (target_int == nbd_size_int - 1) expr1 = new IntegerLiteral(0);

              stmt = new ExpressionStatement(new AssignmentExpression(accforIndex.clone(),
                    AssignmentOperator.NORMAL, expr1));

              cStmt.addStatementBefore(fLoop, stmt);

              i--;
            }
          }

          /* Add collapse counter variable incrementations */
          /* DEBUG: Could this code be factored out into a function like collapseLoopTail(...) */
          fBody = (CompoundStatement)fLoop.getBody();
          int i = 1;
          while (i < collapseLevel) {
            Expression tVar = indexVariables.get(i).clone();
            Expression nVar = indexVariables.get(i-1).clone();
            Statement stmt;

            // x = x + 1
            if (i == collapseLevel-1) {
              expr1 = new BinaryExpression(tVar.clone(), BinaryOperator.ADD, new IntegerLiteral(1));
              stmt = new ExpressionStatement(new AssignmentExpression(tVar.clone(),
                    AssignmentOperator.NORMAL, expr1));
              fBody.addStatement(stmt);
            }

            // y = (x == X) ? y + 1 : y
            Expression expr_true = new BinaryExpression(nVar.clone(), BinaryOperator.ADD,
                new IntegerLiteral(1));
            Expression expr_cond = new BinaryExpression(tVar.clone(), BinaryOperator.COMPARE_EQ,
                iterspaceList.get(i).clone());
            expr1 = new ConditionalExpression(expr_cond, expr_true, nVar.clone());
            stmt = new ExpressionStatement(new AssignmentExpression(nVar.clone(),
                  AssignmentOperator.NORMAL, expr1));
            fBody.addStatement(stmt);

            // x = (x == X) ? 0 : x; 
            expr_cond = new BinaryExpression(tVar.clone(), BinaryOperator.COMPARE_EQ,
                iterspaceList.get(i).clone());
            expr1 = new ConditionalExpression(expr_cond, new IntegerLiteral(0), tVar.clone());
            stmt = new ExpressionStatement(new AssignmentExpression(tVar.clone(),
                  AssignmentOperator.NORMAL, expr1));
            fBody.addStatement(stmt);

            i++;
          }
        }

        ///////////////////////////////////////////
        // Multi transformation (loop unrolling) //
        ///////////////////////////////////////////

        if (unroll_factor > 1) {

          /* Calculate window size */
          Expression ssize = new IntegerLiteral(unroll_factor);
          Expression sw_size = Symbolic.simplify(new IntegerLiteral(nbd_size_int + unroll_factor - 1));

          /* Create and 0-initialize sliding window array */
          List<Specifier> removeSpecs = new ArrayList<Specifier>();
          removeSpecs.add(Specifier.STATIC);
          removeSpecs.add(Specifier.CONST);
          removeSpecs.add(Specifier.EXTERN);
          List<Specifier> typeSpecs = new ArrayList<Specifier>();
          Symbol IRSym = iSym;
          if( iSym instanceof PseudoSymbol ) {
            IRSym = ((PseudoSymbol)iSym).getIRSymbol();
            Symbol tSym = iSym;
            while( tSym instanceof AccessSymbol ) {
              tSym = ((AccessSymbol)tSym).getMemberSymbol();
            }
            typeSpecs.addAll(((VariableDeclaration)tSym.getDeclaration()).getSpecifiers());
          } else {
            typeSpecs.addAll(((VariableDeclaration)IRSym.getDeclaration()).getSpecifiers());
          }
          typeSpecs.removeAll(removeSpecs);
          String symNameBase = null;
          if( iSym instanceof AccessSymbol) {
            symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)iSym);
          } else {
            symNameBase = iSym.getSymbolName();
          }
          String windowSymName = "sw__" + symNameBase;
          List tindices = new LinkedList();

          tindices.add(sw_size.clone());

          ArraySpecifier aspec = new ArraySpecifier(tindices);
          List tailSpecs = new ArrayList(1);
          tailSpecs.add(aspec);
          VariableDeclarator wSym_declarator = new VariableDeclarator(new NameID(windowSymName), tailSpecs);
          Identifier sw = new Identifier(wSym_declarator);
          VariableDeclaration wSym_decl = new VariableDeclaration(typeSpecs,
              wSym_declarator);
          cStmt.addDeclaration(wSym_decl);

          /* Create a new temp index variable. */
          Symbol indexSym = SymbolTools.getSymbolOf(iter);
          List<Specifier> typeSpecs2 = new ArrayList<Specifier>();
          typeSpecs2.addAll(indexSym.getTypeSpecifiers());
          symNameBase = indexSym.getSymbolName();
          Identifier nIndex = TransformTools.getTempScalar(cStmt, typeSpecs2, "w__" + symNameBase, 0);

          /* Initialize the sliding window array. */
          Expression expr1 = new AssignmentExpression(nIndex, AssignmentOperator.NORMAL,
              new IntegerLiteral(0));
          Statement expStmt = new ExpressionStatement(expr1);
          expr1 = new BinaryExpression((Identifier)nIndex.clone(), BinaryOperator.COMPARE_LT,
              sw_size.clone());
          Expression expr2 = new UnaryExpression(
              UnaryOperator.POST_INCREMENT, (Identifier)nIndex.clone());
          CompoundStatement forBody = new CompoundStatement();
          Expression expr3 = new AssignmentExpression(new ArrayAccess(sw.clone(), nIndex.clone()), AssignmentOperator.NORMAL,
              new IntegerLiteral(0)); 
          Statement bStmt = new ExpressionStatement(expr3);
          forBody.addStatement(bStmt);
          ForLoop wLoop = new ForLoop(expStmt, expr1, expr2, forBody);
          wLoop.annotate(unrollpragma.clone());
          cStmt.addStatement(wLoop);

          /* Initialize slidwing window by redefining loop header */ 
          /* NOTE: Partially omitted if TARGET == NBD_SIZE - 1 */

          if (target_int != nbd_size_int - 1) {
            // i = TARGET - (NBD_SIZE - 1);
            expr1 = new BinaryExpression(target.clone(), BinaryOperator.SUBTRACT, new IntegerLiteral(nbd_size_int - 1));
            expr1 = new AssignmentExpression(iter.clone(), AssignmentOperator.NORMAL, expr1);

            Statement initStmt = new ExpressionStatement(expr1);
            fLoop.getInitialStatement().swapWith(initStmt);
          }

          // i += ssize
          expr1 = new AssignmentExpression(iter.clone(), AssignmentOperator.ADD, ssize.clone());
          fLoop.getStep().swapWith(expr1);


          /* Identify LHS accesses to input_array */
          /* NOTE: I think this may only be needed if input_array == output_array */
          List<AssignmentExpression> assignList = IRTools.getExpressionsOfType(fBody, AssignmentExpression.class);
          ArrayList<ArrayAccess> LHSarrayAccessList = new ArrayList<ArrayAccess>(); 
          ArrayList<Statement> LHSstatementList = new ArrayList<Statement>(); 

          if ( assignList != null) {
            for (AssignmentExpression assignExp : assignList) {
              if (assignExp.getLHS().getClass() == ArrayAccess.class) {
                ArrayAccess LHSaAccess = ((ArrayAccess) assignExp.getLHS());

                if (LHSaAccess.getArrayName().equals(input_array)) {
                  LHSarrayAccessList.add(LHSaAccess);
                }
              }
            }
          }
          /* replace iter with TARGET+ss */
          /* NOTE: This is done either by                                                  */
          /*  1. direct substitution      array[iter] -> array[TARGET + ss], or            */  
          /*  2. addition/subtraction     array[ijk]  -> array[ijk - iter + TARGET + ss]   */        

          // replace iter with iter+ss
          Identifier ssIndex = TransformTools.getTempScalar(cStmt, typeSpecs2, "ss__" + symNameBase, 0);
          Expression inner_iterExp;
          if (collapseLevel == 0) 
            inner_iterExp = iter;
          else
            inner_iterExp = indexVariables.get(indexVariables.size() - 1);

          List<Expression> expressionList = IRTools.findExpressions(fBody, inner_iterExp);
          for (Expression oExp : expressionList) {
            // iter
            Expression nExp = oExp.clone();
            // iter + ss
            nExp = new BinaryExpression(nExp, BinaryOperator.ADD, ssIndex.clone());
            oExp.swapWith(nExp);
          }

          // replace iter with TARGET
          arrayAccessList = IRTools.getExpressionsOfType(fBody, ArrayAccess.class);
          if( arrayAccessList != null ) {
            for( ArrayAccess aAccess : arrayAccessList ) {
              if( aAccess.getArrayName().equals(input_array) ) {
                List<Expression> oldindices = aAccess.getIndices();
                List<Expression> newindices = new ArrayList<Expression>(oldindices.size());
                for( Expression oExp : oldindices ) {
                  Expression nExp = oExp.clone();

                  Expression nExpSim = Symbolic.simplify(nExp);
                  Expression iterExpSim = Symbolic.simplify(iterExp);

                  if (nExp.equals(iterExp)) {
                    //System.out.println("Immediate MATCH");
                    nExp = target.clone();
                  }
                  else if (IRTools.containsExpression(nExp, iterExp)) {
                    //System.out.println("Replace MATCH");
                    IRTools.replaceAll(nExp, iterExp, target.clone());
                  }
                  else if (IRTools.containsExpression(nExpSim, iterExpSim)) {
                    //System.out.println("Simplify MATCH");
                    IRTools.replaceAll(nExpSim, iterExpSim, target.clone());

                    nExp = nExpSim;
                  }
                  else {
                    System.out.println("[WARNING in SlidingWindowTransformation] All input array index expressions " + 
                        "must be of the form:\n  required: " + iterExp.toString() + 
                        " + CONST\n  found: " + nExp.toString());
                    System.out.println(" Performing addition/subtraction instead of direct replacement\n");
                    nExp = new BinaryExpression(nExp, BinaryOperator.SUBTRACT, iterExp.clone());
                    nExp = new BinaryExpression(nExp, BinaryOperator.ADD, target.clone());
                  }

                  nExp = Symbolic.simplify(nExp);

                  newindices.add(nExp);
                }
                ArrayAccess nAccess = new ArrayAccess(sw.clone(), newindices);
                aAccess.swapWith(nAccess);
                aAccess.setParent(nAccess.getParent());
              }
            }
          }

          /* Reinsert LHS input_array accesses, updating their values based on the sliding window */
          for (ArrayAccess LHSaAccess : LHSarrayAccessList) {
            // generate index expression list for RHS
            List<Expression> LHSindices = LHSaAccess.getIndices();
            List<Expression> RHSindices = new ArrayList<Expression>(LHSindices.size());
            for( Expression lhsExp : LHSindices ) {
              Expression nExp = lhsExp.clone();

              Expression nExpSim = Symbolic.simplify(nExp);
              Expression iterExpSim = Symbolic.simplify(iterExp);

              // handles input[i] case
              if (nExp.equals(iterExp)) {
                //System.out.println("Immediate MATCH");
                nExp = target.clone();
              }
              else if (IRTools.containsExpression(nExp, iterExp)) {
                //System.out.println("Replace MATCH");
                IRTools.replaceAll(nExp, iterExp, target.clone());
              }
              else if (IRTools.containsExpression(nExpSim, iterExpSim)) {
                //System.out.println("Simplify MATCH");
                IRTools.replaceAll(nExpSim, iterExpSim, target.clone());

                nExp = nExpSim;
              }
              else {
                System.out.println("[WARNING in SlidingWindowTransformation] All input array index expressions " + 
                    "must be of the form:\n  required: " + iterExp.toString() + 
                    " + CONST\n  found: " + nExp.toString());
                System.out.println(" Performing addition/subtraction instead of direct replacement\n");
                nExp = new BinaryExpression(nExp, BinaryOperator.SUBTRACT, iterExp.clone());
                nExp = new BinaryExpression(nExp, BinaryOperator.ADD, target.clone());
              }

              nExp = Symbolic.simplify(nExp);
              RHSindices.add(nExp);
            }
            ArrayAccess RHSaAccess = new ArrayAccess(sw.clone(), RHSindices);

            // generate assignment statement and LHS statement 
            Statement stmt = new ExpressionStatement(new AssignmentExpression(LHSaAccess.clone(),
                  AssignmentOperator.NORMAL, RHSaAccess.clone()));

            // add statement to loop conditional
            CompoundStatement refCmpStmt = IRTools.getAncestorOfType(LHSaAccess, CompoundStatement.class);
            Statement refStmt = IRTools.getAncestorOfType(LHSaAccess, Statement.class); 

            refCmpStmt.addStatementAfter(refStmt, stmt);
          }

          /* Enclose loop body in a conditional to omit computation in intilization iterations */
          /* DEBUG: This can be ommitted if TARGET == SW_BASE_SIZE */

          // for (int ss = 0; ss < ssize; ++ss) 
          expr1 = new AssignmentExpression(ssIndex.clone(), AssignmentOperator.NORMAL, new IntegerLiteral(0));
          expStmt = new ExpressionStatement(expr1);
          expr1 = new BinaryExpression((Identifier)ssIndex.clone(), BinaryOperator.COMPARE_LT, ssize.clone());
          expr2 = new UnaryExpression(UnaryOperator.POST_INCREMENT, (Identifier)ssIndex.clone());

          ForLoop ssLoop = new ForLoop(expStmt, expr1, expr2, fBody.clone());
          ssLoop.annotate(unrollpragma.clone());

          CompoundStatement fBody_mod = new CompoundStatement();
          fBody_mod.swapWith(fBody);

          Expression cond;
          IfStatement ifStmt;
          if (target_int != nbd_size_int - 1) {
            // if (iter > 0)
            cond = new BinaryExpression(iter.clone(), BinaryOperator.COMPARE_GE, new IntegerLiteral(0));
            ifStmt = new IfStatement(cond, ssLoop); 

            fBody_mod.addStatement(ifStmt);
          }
          else {
            fBody_mod.addStatement(ssLoop);
          }

          firstExeStatement = AnalysisTools.getFirstExecutableStatement(fBody_mod);


          /*  Slide the window each iteration */
          Expression biexp = null;

          // i = 0
          expr1 = new AssignmentExpression(nIndex.clone(), AssignmentOperator.NORMAL,
              new IntegerLiteral(0));
          expStmt = new ExpressionStatement(expr1);

          // i < nbd_size - 1
          expr1 = new BinaryExpression((Identifier)nIndex.clone(), BinaryOperator.COMPARE_LT, new IntegerLiteral(nbd_size_int - 1)); 

          // i ++
          expr2 = new UnaryExpression(
              UnaryOperator.POST_INCREMENT, (Identifier)nIndex.clone());
          forBody = new CompoundStatement();

          // sw[i] = sw[i + ssize]
          expr3 = new AssignmentExpression(new ArrayAccess(sw.clone(), nIndex.clone()), AssignmentOperator.NORMAL,
              new ArrayAccess(sw.clone(), new BinaryExpression(nIndex.clone(), BinaryOperator.ADD, ssize.clone() )));

          bStmt = new ExpressionStatement(expr3);
          forBody.addStatement(bStmt);
          wLoop = new ForLoop(expStmt, expr1, expr2, forBody);
          wLoop.annotate(unrollpragma.clone());

          if( firstExeStatement == null ) {
            fBody_mod.addStatement(wLoop);
          } else {
            fBody_mod.addStatementBefore(firstExeStatement, wLoop);
          }

          /* Read value from input array to sliding window */
          Identifier read_offset = null;

          read_offset = TransformTools.getTempScalar(fBody_mod, typeSpecs2, "read_offset__" + symNameBase, 0);

          // int read_offset = iter + NBD_SZIE - 1 - TARGET;
          expr1 = new BinaryExpression(iter.clone(), BinaryOperator.ADD, new IntegerLiteral(nbd_size_int - 1));
          expr1 = new BinaryExpression(expr1, BinaryOperator.SUBTRACT, target.clone());

          expr1 = new AssignmentExpression(read_offset.clone(), AssignmentOperator.NORMAL, expr1);
          expStmt = new ExpressionStatement(expr1);

          if( firstExeStatement == null ) {
            fBody_mod.addStatement(expStmt);
          } else {
            fBody_mod.addStatementBefore(firstExeStatement, expStmt);
          }

          // (read_offset + ss >= 0 && read_offset + ss < N)
          expr1 = new BinaryExpression(read_offset.clone(), BinaryOperator.ADD, ssIndex.clone());
          expr1 = new BinaryExpression(expr1, BinaryOperator.COMPARE_GE, new IntegerLiteral(0));

          expr2 = new BinaryExpression(read_offset.clone(), BinaryOperator.ADD, ssIndex.clone());
          expr2 = new BinaryExpression(read_offset.clone(), BinaryOperator.COMPARE_LT, iarray_Length.clone());

          expr3 = new BinaryExpression(expr1, BinaryOperator.LOGICAL_AND, expr2);

          //  sw[NBD_SIZE - 1 + ss] = input[read_offset + ss];
          expr1 = new BinaryExpression(new IntegerLiteral(nbd_size_int - 1), BinaryOperator.ADD,  ssIndex.clone());
          expr1 = new ArrayAccess(sw.clone(), expr1);

          expr2 = new BinaryExpression(read_offset.clone(), BinaryOperator.ADD,  ssIndex.clone());
          expr2 = new ArrayAccess(input_array.clone(), expr2);

          expr1 = new AssignmentExpression(expr1, AssignmentOperator.NORMAL, expr2);

          expStmt = new ExpressionStatement(expr1);
          ifStmt = new IfStatement(expr3, expStmt);

          //  sw[NBD_SIZE - 1 + ss] = 0 
          expr1 = new BinaryExpression(new IntegerLiteral(nbd_size_int - 1), BinaryOperator.ADD,  ssIndex.clone());
          expr1 = new AssignmentExpression(new ArrayAccess(sw.clone(), expr1),
              AssignmentOperator.NORMAL, new IntegerLiteral(0) );

          expStmt = new ExpressionStatement(expr1);
          CompoundStatement elseBody = new CompoundStatement();
          elseBody.addStatement(expStmt);
          ifStmt.setElseStatement(elseBody);

          // enclose in ssize for loop
          expr1 = new AssignmentExpression(ssIndex.clone(), AssignmentOperator.NORMAL, new IntegerLiteral(0));
          expStmt = new ExpressionStatement(expr1);
          expr1 = new BinaryExpression((Identifier)ssIndex.clone(), BinaryOperator.COMPARE_LT, ssize.clone());
          expr2 = new UnaryExpression(UnaryOperator.POST_INCREMENT, (Identifier)ssIndex.clone());

          ssLoop = new ForLoop(expStmt, expr1, expr2, ifStmt);
          ssLoop.annotate(unrollpragma.clone());

          if( firstExeStatement == null ) {
            fBody_mod.addStatement(ssLoop);
          } else {
            fBody_mod.addStatementBefore(firstExeStatement, ssLoop);
          }

          /* Replace the target loop (fLoop) with the new, enclosing compound statement (cStmt), */
          /* and add the target loop into the compound statement.                                */
          cStmt.swapWith(fLoop);
          cStmt.addStatement(fLoop);

          /* Add initalizations before loop */
          /* DEBUG: Could this code be factored out into a function like collapseLoopTail(...) */
          if (collapseLevel > 1) {
            Expression accforIndex = indexVariables.get(0).clone();
            Expression indexExp = new BinaryExpression(target.clone(), BinaryOperator.SUBTRACT, new IntegerLiteral(nbd_size_int - 1));

            int i = collapseLevel-1;
            while (i > 0) {
              Expression tIndex = indexVariables.get(i).clone();

              if (i == (collapseLevel-1)) {
                expr1 = new BinaryExpression(indexExp.clone(), BinaryOperator.MODULUS,
                    iterspaceList.get(i).clone());
              } else {
                expr1 = new BinaryExpression(accforIndex.clone(), BinaryOperator.MODULUS,
                    iterspaceList.get(i).clone());
              }

              if (target_int == nbd_size_int - 1) expr1 = new IntegerLiteral(0);

              Statement stmt = new ExpressionStatement(new AssignmentExpression(tIndex.clone(),
                    AssignmentOperator.NORMAL, expr1));

              cStmt.addStatementBefore(fLoop, stmt);

              if( i == (collapseLevel-1) ) {
                expr1 = new BinaryExpression(indexExp.clone(), BinaryOperator.DIVIDE,
                    iterspaceList.get(i).clone());
              } else {
                expr1 = new BinaryExpression(accforIndex.clone(), BinaryOperator.DIVIDE,
                    iterspaceList.get(i).clone());
              }

              if (target_int == nbd_size_int - 1) expr1 = new IntegerLiteral(0);

              stmt = new ExpressionStatement(new AssignmentExpression(accforIndex.clone(),
                    AssignmentOperator.NORMAL, expr1));

              cStmt.addStatementBefore(fLoop, stmt);

              i--;
            }

          }

          fBody = (CompoundStatement)fLoop.getBody();
          firstExeStatement = AnalysisTools.getFirstExecutableStatement(fBody);

          /* Define ss values array */
          IRSym = oSym;
          if( oSym instanceof PseudoSymbol ) {
            IRSym = ((PseudoSymbol)oSym).getIRSymbol();
            Symbol tSym = oSym;
            while( tSym instanceof AccessSymbol ) {
              tSym = ((AccessSymbol)tSym).getMemberSymbol();
            }
          }
          symNameBase = null;
          if( oSym instanceof AccessSymbol) {
            symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)oSym);
          } else {
            symNameBase = oSym.getSymbolName();
          }
          String valuesSymName = "values__ss__" + symNameBase;

          tindices = new LinkedList();
          tindices.add(ssize.clone());
          aspec = new ArraySpecifier(tindices);
          tailSpecs = new ArrayList(1);
          tailSpecs.add(aspec);

          VariableDeclarator vSym_declarator = new VariableDeclarator(new NameID(valuesSymName), tailSpecs);
          Identifier values_ss = new Identifier(vSym_declarator);
          VariableDeclaration vSym_decl = new VariableDeclaration(typeSpecs, vSym_declarator);
          fBody.addDeclaration(vSym_decl);

          /* Replace reads from output[] with reads from values[] */
          arrayAccessList = IRTools.getExpressionsOfType(fBody_mod, ArrayAccess.class);
          if ( arrayAccessList != null ) {
            for ( ArrayAccess aAccess : arrayAccessList ) {
              if ( aAccess.getArrayName().equals(output_array) ) {

                int swap = 0;
                // check if aAccess is on the RHS of an expression, which indicates that this 
                // specific aAccess is a read from the output[] (and not a write to output[])
                // NOTE: This is done by traversing up the tree. When we find an assignment 
                // expression, we compare the RHS of the expression with the traversed expression 
                // DEBUG: There may be a cleaner/better/easier way to do this
                Traversable aTrav = aAccess;
                while (true) {
                  Traversable aTravParent = aTrav.getParent();
                  if (aTravParent == null) {
                    swap = 1;
                    System.out.println("No parent assignment " + aAccess.toString());
                    break;
                  }
                  else if (aTravParent.getClass() == AssignmentExpression.class) {
                    if (aTrav.equals(((AssignmentExpression)aTravParent).getRHS())) 
                    {
                      System.out.println("RHS: " + aAccess.toString());
                      swap = 0;
                    }
                    else
                      swap = 1;
                    break;
                  }
                  aTrav = aTravParent;
                }

                // DEBUG: oldindicies should equal iter, we should probably check
                List<Expression> oldindices = aAccess.getIndices();

                if (swap == 1) {
                  ArrayAccess nAccess = new ArrayAccess(values_ss.clone(), ssIndex.clone());
                  aAccess.swapWith(nAccess);
                  aAccess.setParent(nAccess.getParent());
                }

              }
            }
          }

          /* Reassign values[] to output[] */
          // output[iter + ss] = values[ss]
          expr1 = new BinaryExpression(iter.clone(), BinaryOperator.ADD, ssIndex.clone());
          expr1 = new ArrayAccess(output_array.clone(), expr1);
          expr2 = new ArrayAccess(values_ss.clone(), ssIndex.clone());
          expr3 = new AssignmentExpression(expr1, AssignmentOperator.NORMAL, expr2);

          ExpressionStatement loopBody = new ExpressionStatement(expr3);

          // for (int ss = 0; ss < ssize; ++ss) 
          expr1 = new AssignmentExpression(ssIndex.clone(), AssignmentOperator.NORMAL, new IntegerLiteral(0));
          expStmt = new ExpressionStatement(expr1);
          expr1 = new BinaryExpression((Identifier)ssIndex.clone(), BinaryOperator.COMPARE_LT, ssize.clone());
          expr2 = new UnaryExpression(UnaryOperator.POST_INCREMENT, (Identifier)ssIndex.clone());

          ssLoop = new ForLoop(expStmt, expr1, expr2, loopBody.clone()); 
          ssLoop.annotate(unrollpragma.clone());

          // if (iter > 0)
          cond = new BinaryExpression(iter.clone(), BinaryOperator.COMPARE_GE, new IntegerLiteral(0));
          ifStmt = new IfStatement(cond, ssLoop); 

          fBody.addStatement(ifStmt);

          /* Add collapse counter variable incrementations */
          /* DEBUG: Could this code be factored out into a function like collapseLoopTail(...) */
          int i = 1;
          while (i < collapseLevel) {
            Expression tVar = indexVariables.get(i).clone();
            Expression nVar = indexVariables.get(i-1).clone();
            Statement stmt;

            // x = x + sszie
            if (i == collapseLevel-1) {
              expr1 = new BinaryExpression(tVar.clone(), BinaryOperator.ADD, ssize.clone());
              stmt = new ExpressionStatement(new AssignmentExpression(tVar.clone(),
                    AssignmentOperator.NORMAL, expr1));
              fBody.addStatement(stmt);
            }

            // y = (x == X) ? y + 1 : y
            Expression expr_true = new BinaryExpression(nVar.clone(), BinaryOperator.ADD,
                new IntegerLiteral(1));
            Expression expr_cond = new BinaryExpression(tVar.clone(), BinaryOperator.COMPARE_EQ,
                iterspaceList.get(i).clone());
            expr1 = new ConditionalExpression(expr_cond, expr_true, nVar.clone());
            stmt = new ExpressionStatement(new AssignmentExpression(nVar.clone(),
                  AssignmentOperator.NORMAL, expr1));
            fBody.addStatement(stmt);

            // x = (x == X) ? 0 : x; 
            expr_cond = new BinaryExpression(tVar.clone(), BinaryOperator.COMPARE_EQ,
                iterspaceList.get(i).clone());
            expr1 = new ConditionalExpression(expr_cond, new IntegerLiteral(0), tVar.clone());
            stmt = new ExpressionStatement(new AssignmentExpression(tVar.clone(),
                  AssignmentOperator.NORMAL, expr1));
            fBody.addStatement(stmt);

            i++;
          }
        } 

        /* Move OpenACC annotations in the target loop into the compound statement */
        ACCAnnotation internalAnnot = null;
        ACCAnnotation move_annot = new ACCAnnotation();
        ACCAnnotation keep_annot = new ACCAnnotation();
        List<ACCAnnotation> ACCList = fLoop.getAnnotations(ACCAnnotation.class);
        fLoop.removeAnnotations(ACCAnnotation.class);

        for( ACCAnnotation tOldAnnot : ACCList ) {
          Set<String> keySet = tOldAnnot.keySet();
          for( String tKey : keySet ) {
              // move all data clauses up
            if( ACCAnnotation.dataClauses.contains(tKey) ) {
              move_annot.put(tKey, tOldAnnot.get(tKey));
              // move all parallel clauses up
            } else if( tKey.equals("parallel") || tKey.equals("firstprivate") ||
                tKey.equals("private") ||
                tKey.equals("num_gangs") || tKey.equals("num_workers") ) {
              move_annot.put(tKey, tOldAnnot.get(tKey));
              // change and move kernels annotations into parallel annotations
            } else if( tKey.equals("kernels") ) {
              move_annot.put("parallel", tOldAnnot.get(tKey));
              // move gang and worker clauses
            } else if( tKey.equals("num_gangs") ) {
              move_annot.put("num_gangs", tOldAnnot.get(tKey));
            } else if( tKey.equals("num_workers") ) {
              move_annot.put("num_workers", tOldAnnot.get(tKey));
              // move other clauses
            } else if( tKey.equals("async") || tKey.equals("wait") || tKey.equals("if") ) {
              move_annot.put(tKey, tOldAnnot.get(tKey));
              // keep loop-specific annotations
            } else if( tKey.equals("loop") ) {
              keep_annot.put(tKey, tOldAnnot.get(tKey));
              keep_annot.put("seq", "_clause"); //this loop should be executed sequentially.
            }
          }
          if( tOldAnnot.containsKey("internal") ) {
            internalAnnot = tOldAnnot;
          }
        }
        if( nestedLoops.size() > 1 ) {
          for( ForLoop tLoop : nestedLoops ) {
            tLoop.removeAnnotations(ACCAnnotation.class);
          }
        }
        if( internalAnnot != null ) {
          cStmt.annotate(internalAnnot);
        }
        cStmt.annotate(move_annot);
        if( keep_annot.size() > 1 ) {
        	fLoop.annotate(keep_annot);
        }
      }

    }

}

