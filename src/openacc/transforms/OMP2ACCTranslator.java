/**
 * 
 */
package openacc.transforms;

import cetus.analysis.Reduction;
import cetus.hir.*;
import cetus.transforms.TransformPass;
import openacc.analysis.ParserTools;
import openacc.analysis.SubArray;
import openacc.hir.ACCAnnotation;

import java.util.*;

/**
 * @author Putt Sakdhnagool <psakdhna@purdue.edu>
 *
 */
public class OMP2ACCTranslator extends TransformPass {
    private enum OmpRegion
    {
        Target,
        Teams,
        Distribute,
        Parallel_For,
        Parallel,
        SIMD
    }

    protected String pass_name = "[OMP2ACCTranslator]";
    protected Program program;
    //Main refers either a procedure containing acc_init() call or main() procedure if no explicit acc_init() call exists.
    protected Procedure main;

    //Target information
    private boolean targetRegion = false;
    private Expression deviceExpr = null;

    private Stack<Traversable> parentStack = new Stack<Traversable>();
    private Stack<OmpRegion> regionStack = new Stack<OmpRegion>();

    public OMP2ACCTranslator(Program prog) {
        super(prog);
        program = prog;
    }

    @Override
    public String getPassName() {
        return pass_name;
    }

    @Override
    public void start() 
    {
        convertCritical2Reduction();

        targetRegion = false;
        DFIterator<Annotatable> iter = new DFIterator<Annotatable>(program, Annotatable.class);
        while (iter.hasNext()) 
        {
            Annotatable at = iter.next();
            if(parentStack.size() > 0)
            {
                /**
                 * Check if new annotatable has same parent as the current region.
                 * If it's true then the iterator exit the region.
                 */
                if(parentStack.contains(at.getParent()))
                {
                    //PrintTools.println(at.toString(), 0);

                    while(parentStack.size() > 0 && parentStack.peek() != at.getParent())
                    {
                        PrintTools.println("Exit " + regionStack.peek() + " region", 2);
                        parentStack.pop();
                        regionStack.pop();
                    }

                    while(parentStack.size() > 0 && parentStack.peek() == at.getParent())
                    {
                        PrintTools.println("Exit " + regionStack.peek() + " region", 2);
                        parentStack.pop();
                        regionStack.pop();
                    }

                }
            }

            List<OmpAnnotation> pragmas = at.getAnnotations(OmpAnnotation.class);
            for (int i = 0; i < pragmas.size(); i++) 
            {
                OmpAnnotation pragma = pragmas.get(i);
                if(pragma.containsKey("declare"))
                {
                    continue;
                }
                if(pragma.containsKey("target") && pragma.containsKey("update"))
                {
                    //parse_target_pragma(pragma);
                    continue;
                }

                if(pragma.containsKey("target"))
                {
                    //Enter Target region
                    parentStack.push(at.getParent());
                    regionStack.push(OmpRegion.Target);
                    PrintTools.println("Enter " + regionStack.peek() + " region", 2);

                    if(at.getAnnotation(OmpAnnotation.class, "teams") == null)
                    {
                        FlatIterator<Statement> flatIterator = new FlatIterator<Statement>(at);
                        while (flatIterator.hasNext())
                        {
                            Statement stmt = flatIterator.next();
                            OmpAnnotation stmtAnnot = stmt.getAnnotation(OmpAnnotation.class, "teams");
                            if(stmtAnnot == null || !(stmtAnnot.containsKey("teams")))
                            {
                                Tools.exit("[Error] Target construct must contain no statements or directives outside of the teams construct.\n Error at: \n" + stmt);
                                //throw new RuntimeException("Target construct must contain no statements or directives outside of the teams construct.\n" + child);
                            }
                        }
                    }
                    parse_target_pragma(pragma);
                }

                if(regionStack.contains(OmpRegion.Target)) {
                    if (pragma.containsKey("teams")) {
                        //Enter Teams region
                        parentStack.push(at.getParent());
                        regionStack.push(OmpRegion.Teams);
                        PrintTools.println("Enter " + regionStack.peek() + " region", 2);
                        parse_teams_pragma(pragma);
                    }

                    if (pragma.containsKey("distribute")) {
                        //Enter Distribute region
                        parentStack.push(at.getParent());
                        regionStack.push(OmpRegion.Distribute);
                        PrintTools.println("Enter " + regionStack.peek() + " region",2);
                        parse_distribute_pragma(pragma);
                    } else if (pragma.containsKey("parallel") && pragma.containsKey("for")) {
                        //Enter Parallel For region
                        parentStack.push(at.getParent());
                        regionStack.push(OmpRegion.Parallel_For);
                        PrintTools.println("Enter " + regionStack.peek() + " region", 2);
                        parse_parallel_for_pragma(pragma);
                    } else if (pragma.containsKey("simd")) {
                        //Enter SIMD region
                        parentStack.push(at.getParent());
                        regionStack.push(OmpRegion.SIMD);
                        PrintTools.println("Enter " + regionStack.peek() + " region", 2);
                        parse_simd_pragma(pragma);
                    }

                    if (pragma.containsKey("atomic")) {
                        parse_atomic(pragma);
                    }
                }
            }
        }
    }

    private void parse_atomic(OmpAnnotation pragma)
    {        
        ACCAnnotation accAnnot = new ACCAnnotation();
        accAnnot.put("atomic", "true");

        if(pragma.containsKey("type"))
        {
            accAnnot.put((String)pragma.get("type"), "true");
        }
        else
        {
            accAnnot.put("update", "true");           
        }
        pragma.getAnnotatable().annotate(accAnnot);
    }

    private void parse_simd_pragma(OmpAnnotation pragma)
    {
        ACCAnnotation accAnnot = new ACCAnnotation();
        accAnnot.put("loop", "true");
        accAnnot.put("vector", "true");

        pragma.getAnnotatable().annotate(accAnnot);
    }

    private void parse_parallel_for_pragma(OmpAnnotation pragma)
    {
        Set<SubArray> privateSet = new HashSet<SubArray>();
        Set<SubArray> firstprivateSet = new HashSet<SubArray>();

        ACCAnnotation accAnnot = new ACCAnnotation();
        accAnnot.put("loop", "true");

	//Add implicit teams construct
	if(!regionStack.contains(OmpRegion.Teams))
	        accAnnot.put("parallel", "true");

        accAnnot.put("worker", "true");

        if(pragma.containsKey("simd"))
            accAnnot.put("vector", "true");

        if(pragma.containsKey("private"))
            parseSet((Set<String>)pragma.get("private"), privateSet);
        if(pragma.containsKey("firstprivate"))
            parseSet((Set<String>)pragma.get("firstprivate"), firstprivateSet);

        if(privateSet.size() > 0)
            accAnnot.put("private", privateSet);
        if(firstprivateSet.size() > 0)
            accAnnot.put("firstprivate", firstprivateSet);
        if(pragma.containsKey("collapse"))
            accAnnot.put("collapse", ParserTools.strToExpression((String) pragma.get("collapse")));

        pragma.getAnnotatable().annotate(accAnnot);
    }

    private void parse_distribute_pragma(OmpAnnotation pragma)
    {
        Set<SubArray> privateSet = new HashSet<SubArray>();
        Set<SubArray> firstprivateSet = new HashSet<SubArray>();

        ACCAnnotation accAnnot = new ACCAnnotation();
        accAnnot.put("loop", "true");
        accAnnot.put("gang", "true");

        if (pragma.containsKey("parallel") && pragma.containsKey("for"))
            accAnnot.put("worker", "true");

        if(pragma.containsKey("simd"))
            accAnnot.put("vector", "true");

        if(pragma.containsKey("private"))
            parseSet((Set<String>)pragma.get("private"), privateSet);
        if(pragma.containsKey("firstprivate"))
            parseSet((Set<String>)pragma.get("firstprivate"), firstprivateSet);

        if(privateSet.size() > 0)
            accAnnot.put("private", privateSet);
        if(firstprivateSet.size() > 0)
            accAnnot.put("firstprivate", firstprivateSet);
        if(pragma.containsKey("collapse"))
            accAnnot.put("collapse", ParserTools.strToExpression((String) pragma.get("collapse")));

        //TODO: dist_schedule clause

        pragma.getAnnotatable().annotate(accAnnot);
    }

    private void parse_teams_pragma(OmpAnnotation pragma)
    {
        Set<SubArray> privateSet = new HashSet<SubArray>();
        Set<SubArray> firstprivateSet = new HashSet<SubArray>();
        Set<SubArray> copySet = new HashSet<SubArray>();

        Expression numTeamsExpr = null;
        Expression threadLimitExpr = null;

        String defaultExpr = null;

        if(pragma.containsKey("num_teams")) {
            String numTeamsStr = pragma.get("num_teams");
            numTeamsExpr = ParserTools.strToExpression(numTeamsStr);
        }

        if(pragma.containsKey("thread_limit")) {
            String threadLimitStr = pragma.get("thread_limit");
            threadLimitExpr = ParserTools.strToExpression(threadLimitStr);
        }

        if(pragma.containsKey("default")) {
            String defaultStr = pragma.get("default");
            if(defaultStr.compareTo("none") == 0)
            {
                defaultExpr = "none";
            }
        }

        if(pragma.containsKey("private"))
            parseSet((Set<String>)pragma.get("private"), privateSet);
        if(pragma.containsKey("firstprivate"))
            parseSet((Set<String>)pragma.get("firstprivate"), firstprivateSet);
        if(pragma.containsKey("shared"))
            parseSet((Set<String>)pragma.get("shared"), copySet);

        ACCAnnotation accAnnot = new ACCAnnotation();
        accAnnot.put("parallel", "true");
        if(copySet.size() > 0)
            accAnnot.put("copy", copySet);
        if(privateSet.size() > 0)
            accAnnot.put("private", privateSet);
        if(firstprivateSet.size() > 0)
            accAnnot.put("firstprivate", firstprivateSet);
        if(numTeamsExpr != null)
            accAnnot.put("num_gangs", numTeamsExpr);
        if(threadLimitExpr != null)
            accAnnot.put("num_workers", threadLimitExpr);
        if(defaultExpr != null)
            accAnnot.put("default", defaultExpr);

        //TODO:Add reduction clause

        pragma.getAnnotatable().annotate(accAnnot);
    }

    private void parse_target_pragma(OmpAnnotation pragma)
    {
        Expression condition = null;

        Set<SubArray> createSet = new HashSet<SubArray>();
        Set<SubArray> copyinSet = new HashSet<SubArray>();
        Set<SubArray> copyoutSet = new HashSet<SubArray>();

	//Handle condition
        if(pragma.containsKey("if"))
        {
            String condStr = pragma.get("if");
            condition =  ParserTools.strToExpression(condStr);
        }

        if(pragma.containsKey("device"))
        {
            //device clause
            String deviceStr = pragma.get("device");
            deviceExpr = ParserTools.strToExpression(deviceStr);
            ((CompoundStatement)pragma.getAnnotatable().getParent()).addStatementBefore(
                    (Statement)pragma.getAnnotatable(),
                    new AnnotationStatement(new ACCAnnotation("device", deviceExpr)));
        }

        //Handle Data Transfer

        if(pragma.containsKey("alloc"))
            parseSet((Set<String>)pragma.get("alloc"), createSet);

        if(pragma.containsKey("to"))
            parseSet((Set<String>)pragma.get("to"), copyinSet);

        if(pragma.containsKey("from"))
            parseSet((Set<String>)pragma.get("from"), copyoutSet);

        if(pragma.containsKey("tofrom"))
        {
            parseSet((Set<String>)pragma.get("tofrom"), copyoutSet);
            parseSet((Set<String>)pragma.get("tofrom"), copyinSet);
        }

        if((createSet.size() > 0) || (copyinSet.size() > 0) || (copyoutSet.size() > 0))
        {
            ACCAnnotation newAnnot = new ACCAnnotation();
//            newAnnot.put("enter", "true");
            newAnnot.put("data", "true");
            if(createSet.size() > 0) 
                newAnnot.put("create", createSet);
            if(copyinSet.size() > 0) 
                newAnnot.put("copyin", copyinSet);
            if(copyoutSet.size() > 0) 
                newAnnot.put("copyout", copyoutSet);
            if(condition != null)
                newAnnot.put("if", condition);

	    pragma.getAnnotatable().annotate(newAnnot);

//            ((CompoundStatement) pragma.getAnnotatable().getParent()).addStatementBefore(
//                    (Statement) pragma.getAnnotatable(),
//                    new AnnotationStatement(newAnnot));
        }

//        if(copyoutSet.size() > 0) {
//            ACCAnnotation newAnnot = new ACCAnnotation();
//            newAnnot = new ACCAnnotation();
//            newAnnot.put("exit", "true");
//            newAnnot.put("data", "true");
//            newAnnot.put("copyout", copyoutSet);
//            if(condition != null)
//                newAnnot.put("if", condition);
//            ((CompoundStatement) pragma.getAnnotatable().getParent()).addStatementAfter(
//                    (Statement) pragma.getAnnotatable(),
//                    new AnnotationStatement(newAnnot));
//        }
    }

    private void parseSet(Set<String> orig, Set dest)
    {
        for(String s: orig)
        {
            dest.add(new SubArray(new NameID(s)));
        }
    }

    /**
     * [Convert critical sections into reduction sections]
     * For each critical section in a parallel region,
     *     if the critical section is a kind of reduction form, necessary reduction
     *     clause is added to the annotation of the enclosing parallel region, and
     *     the original critical construct is commented out.
     * A critical section is considered as a reduction form if reduction variables recognized
     * by Reduction.analyzeStatement2() are the only shared variables modified in the
     * critical section.
     * [CAUTION] Cetus compiler can recognize array reduction, but the array reduction
     * is not supported by standard OpenMP compilers. Therefore, below conversion may
     * not be handled correctly by other OpenMP compilers.
     * [FIXME] Reduction.analyzeStatement2() returns a set of reduction variables as expressions,
     * but this method converts them into a set of symbols. This conversion loses some information
     * and thus complex reduction expressions such as a[0][i] and a[i].b can not be handled properly;
     * current translator supports only simple scalar or array variables.
     */
    public void convertCritical2Reduction()
    {
        List<OmpAnnotation> ompPAnnots = IRTools.collectPragmas(program, OmpAnnotation.class, "parallel");
		PrintTools.println(ompPAnnots.toString(), 0);
        Reduction redAnalysis = new Reduction(program);
        for (OmpAnnotation omp_annot : ompPAnnots)
        {
            Statement pstmt = (Statement)omp_annot.getAnnotatable();
            HashSet<Symbol> shared_set = (HashSet<Symbol>)omp_annot.get("shared");
            HashMap pRedMap = (HashMap)omp_annot.get("reduction");
            List<OmpAnnotation> ompCAnnots = IRTools.collectPragmas(pstmt, OmpAnnotation.class, "critical");
		PrintTools.println(ompCAnnots.toString(), 0);
            for (OmpAnnotation cannot : ompCAnnots)
            {
                boolean foundError = false;
                Statement cstmt = (Statement)cannot.getAnnotatable();
                Set<Symbol> definedSymbols = DataFlowTools.getDefSymbol(cstmt);
                HashSet<Symbol> shared_subset = new HashSet<Symbol>();
                shared_subset.addAll(shared_set);
                Map<String, Set<Expression>> reduce_map = redAnalysis.analyzeStatement2(cstmt);
		PrintTools.println(reduce_map.toString(), 0);
                Map<String, Set<Symbol>> reduce_map2 = new HashMap<String, Set<Symbol>>();
                if (!reduce_map.isEmpty())
                {
                    // Remove reduction variables from shared_subset.
                    for (String ikey : (Set<String>)(reduce_map.keySet())) {
                        if( foundError ) {
                            break;
                        }
                        Set<Expression> tmp_set = (Set<Expression>)reduce_map.get(ikey);
                        HashSet<Symbol> redSet = new HashSet<Symbol>();
                        for (Expression iexp : tmp_set) {
                            //Symbol redSym = findsSymbol(shared_set, iexp.toString());
                            Symbol redSym = SymbolTools.getSymbolOf(iexp);
                            if( redSym != null ) {
                                if( redSym instanceof VariableDeclarator ) {
                                    shared_subset.remove(redSym);
                                    redSet.add(redSym);
                                } else {
                                    PrintTools.println("[INFO in convertCritical2Reduction()] the following expression has reduction pattern" +
                                            " but not handled by current translator: " + iexp, 0);
                                    //Skip current critical section.
                                    foundError = true;
                                    break;

                                }
                            } else {
                                PrintTools.println("[WARNING in convertCritical2Reduction()] found unrecognizable reduction expression (" +
                                        iexp+")", 0);
                                //Skip current critical section.
                                foundError = true;
                                break;
                            }
                        }
                        reduce_map2.put(ikey, redSet);
                    }
                    //If error is found, skip current critical section.
                    if( foundError ) {
                        continue;
                    }
                    //////////////////////////////////////////////////////////////////////
                    // If shared_subset and definedSymbols are disjoint,                //
                    // it means that reduction variables are the only shared variables  //
                    // defined in the critical section.                                 //
                    //////////////////////////////////////////////////////////////////////
                    if( Collections.disjoint(shared_subset, definedSymbols) ) {
                        if( pRedMap == null ) {
                            pRedMap = new HashMap();
                            omp_annot.put("reduction", pRedMap);
                        }
                        for (String ikey : (Set<String>)(reduce_map2.keySet())) {
                            Set<Symbol> tmp_set = (Set<Symbol>)reduce_map2.get(ikey);
                            HashSet<Symbol> redSet = (HashSet<Symbol>)pRedMap.get(ikey);
                            if( redSet == null ) {
                                redSet = new HashSet<Symbol>();
                                pRedMap.put(ikey, redSet);
                            }
                            redSet.addAll(tmp_set);
                        }
                        // Remove omp critical annotation and add comment annotation.
                        CommentAnnotation comment = new CommentAnnotation(cannot.toString());
                        AnnotationStatement comment_stmt = new AnnotationStatement(comment);
                        CompoundStatement parent = (CompoundStatement)cstmt.getParent();
                        parent.addStatementBefore(cstmt, comment_stmt);
                        cstmt.removeAnnotations(OmpAnnotation.class);
                    }
                }
            }
        }
    }

}
