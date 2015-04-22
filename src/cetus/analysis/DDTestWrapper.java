package cetus.analysis;

import cetus.exec.Driver;
import cetus.hir.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.LinkedList;
import java.util.List;
import java.util.Arrays;
import java.util.Iterator;

/**
 * Wrapper framework for executing specific data-dependence test on array
 * subscripts
 */
public class DDTestWrapper {
    // Store the array accesses for which this wrapper will perform dependence
    // testing
    private DDArrayAccessInfo acc1, acc2;
    // Common eligible nest for the two accesses which will provide the
    // dependence vector
    private LinkedList<Loop> loop_nest;
    // Loop Info for all loops pertaining to these two accesses
    private HashMap<Loop, LoopInfo> loop_info;
    // Multiple dependence tests can be used for testing
    private static final int DDTEST_BANERJEE = 1;
    private static final int DDTEST_RANGE = 2;
    private static final int DDTEST_OMEGA = 3;
    // Get Commandline input for which test must be run,
    // default = DDTEST_BANERJEE
    private int ddtest_type;
    private static final Cache<List<Statement>, Boolean>
            reachable_stmts = new Cache<List<Statement>, Boolean>();
    private static final Cache<Loop, CFGraph>
            body_graph = new Cache<Loop, CFGraph>();

    /**
    * Constructs a new test wrapper with the specified pair of array accesses
    * and loop information.
    */
    public DDTestWrapper(DDArrayAccessInfo a1,
                         DDArrayAccessInfo a2,
                         LinkedList<Loop> loop_nest,
                         HashMap<Loop, LoopInfo> loopInfo) {
        // Array accesses and their information required for testing
        this.acc1 = a1;
        this.acc2 = a2;
        this.loop_nest = loop_nest;
        this.loop_info = loopInfo;
        this.ddtest_type = Integer.parseInt(Driver.getOptionValue("ddt"));
    }
    
    /**
     * Accepts two access pairs, partitions their subscripts, performs
     * dependence testing and constructs a set of dependence vectors if
     * dependence exists
     * @param DVset
     * @return true if dependence exists
     */
    public boolean testAccessPair(ArrayList<DependenceVector> DVset) {
        boolean dependence_result = true;
        if (ddtest_type == DDTEST_OMEGA) {
            dependence_result = testAllSubscriptsTogether(DVset);
        }
        // Add other dependence tests here when required
        // else if ...
        // By default, use Banerjee DDTEST_BANERJEE = 1
        else if (ddtest_type == DDTEST_BANERJEE ||
                 ddtest_type == DDTEST_RANGE) {
            dependence_result = testSubscriptBySubscript(DVset);
        }
        if (ddtest_type == DDTEST_RANGE && !DVset.isEmpty()) {
            RangeTest.compressDV(DVset);
        }
        return dependence_result;
    }

    private boolean
            testAllSubscriptsTogether(ArrayList<DependenceVector> DVset) {
        DDTest ddtest = null;
        ArrayList<DependenceVector> returned_DVset;
        if (ddtest_type == DDTEST_OMEGA) {
            // Currently, the Omega Test inclusion in Cetus is on hold. This 
            // section of code is not meant to be used by Cetus users. 
            // Remove the warning and exit call once Omega test inclusion 
            // is complete 
            // ddtest = new OmegaTest(getAcc1(),
            //                        getAcc2(),
            //                        getCommonEnclosingLoops(),
            //                        getAllLoopsInfo());
            throw new UnsupportedOperationException(
                    "OMEGA test is not supported now");
        }
        // else if .. add other whole array access tests here
        returned_DVset = testAllDependenceVectors(ddtest);
        if (returned_DVset.size() == 0) {
            return false;
        } else {
            // Merge returned set with input DVset
            mergeVectorSets(DVset, returned_DVset);
            return true;
        }
    }

    private boolean
            testSubscriptBySubscript(ArrayList<DependenceVector> DVset) {
        ArrayList<SubscriptPair> subscriptPairs;
        ArrayAccess access1 = acc1.getArrayAccess();  //(ArrayAccess)acc1;
        ArrayAccess access2 = acc2.getArrayAccess();  //(ArrayAccess)acc2;
        Statement stmt1 = acc1.getParentStatement();
        Statement stmt2 = acc2.getParentStatement();
        // test dependency only if two array accesses with the same array
        // symbol id and have the same dimension
        if (access1.getNumIndices() == access2.getNumIndices()) {
            // Obtain single subscript pairs and while doing so, check if the
            // subscripts are affine
            int dimensions = access1.getNumIndices();
            subscriptPairs = new ArrayList<SubscriptPair>(dimensions);
            for (int dim = 0; dim < dimensions; dim++) {
                SubscriptPair pair = new SubscriptPair(
                        access1.getIndex(dim),
                        access2.getIndex(dim),
                        stmt1,
                        stmt2,
                        loop_nest,
                        loop_info);
                subscriptPairs.add(dim, pair);
            }
            // Partition the subscript pairs - currently ignore effects of
            // coupled subscripts
            List<List<SubscriptPair>> parts = getPartition(subscriptPairs);
            for (int i = 0; i < parts.size(); i++) {
                boolean depExists = testPartition(parts.get(i), DVset);
                if (!depExists) {
                    return false;
                }
            }
        } else {
            // For arrays with different dimensions that are said to be
            // aliased, conservatively assume dependence in all directions
            // with respect to enclosing loops
            DependenceVector dv =
                    new DependenceVector(loop_nest);
            if (!DVset.contains(dv)) {
                DVset.add(dv);
            }
        }
        // Dependence exists
        return true;
    }

    /**
    * Returns a partitioned list from the given list of subscript pairs.
    * The resulting partition keeps coupled subscripts in the same group.
    * TODO: This method should replace getSubscriptPartitions in the future
    * once coupled subscripts are handled effectively.
    */
    private List<List<SubscriptPair>> getPartition(List<SubscriptPair> pairs){
/*
        PrintTools.printlnStatus(2, "For ", acc1, " and ", acc2);
*/
        List<List<SubscriptPair>> ret =
                new ArrayList<List<SubscriptPair>>(pairs.size());
        for (int i = 0; i < pairs.size(); i++) {
            SubscriptPair pair = pairs.get(i);
            List<SubscriptPair> group = new ArrayList<SubscriptPair>();
            group.add(pair);
            if (pair.getComplexity() == 0) {
                ret.add(0, group);
            } else {
                ret.add(group);
            }
        }
        for (int i = 0; i < loop_nest.size(); i++) {
            Symbol index = LoopTools.getLoopIndexSymbol(
                           loop_nest.get(i));
            List<SubscriptPair> new_parts = null;
            Iterator<List<SubscriptPair>> iter = ret.iterator();
            while (iter.hasNext()) {
                List<SubscriptPair> parts = iter.next();
                boolean has_index = false;
                for (int j = 0; j < parts.size(); j++) {
                    Expression sub1 = parts.get(j).getSubscript1();
                    Expression sub2 = parts.get(j).getSubscript2();
                    if (IRTools.containsSymbol(sub1, index) ||
                        IRTools.containsSymbol(sub2, index)) {
                        has_index = true;
                        break;
                    }
                }
                if (has_index) {
                    if (new_parts == null) {
                        new_parts = parts;
                    } else {
                        new_parts.addAll(parts);
                        iter.remove();
                    }
                }
            }
        }
/*
        if (PrintTools.getVerbosity() > 1) {
            StringBuilder sb = new StringBuilder(80);
            for (int i = 0; i < ret.size(); i++) {
                sb.append("    Partition#").append(i).append(": {");
                sb.append("(").append(ret.get(i).get(0).getSubscript1());
                sb.append(",").append(ret.get(i).get(0).getSubscript2());
                sb.append(")");
                for (int j = 1; j < ret.get(i).size(); j++) {
                    sb.append(", (").append(ret.get(i).get(j).getSubscript1());
                    sb.append(",").append(ret.get(i).get(j).getSubscript2());
                    sb.append(")");
                }
                sb.append("}").append(PrintTools.line_sep);
            }
            PrintTools.printlnStatus(0, sb);
        }
*/
        return ret;
    }

    // Caution: call this only after all subscriptPairs are found
    private List<Set<SubscriptPair>>
            getSubscriptPartitions(List<SubscriptPair> subscriptPairs) {
        // for now they are all separable
        LinkedList<Set<SubscriptPair>>
                partitions = new LinkedList<Set<SubscriptPair>>();
        // this may look redundant now, but all the partitions are singletons 
        // containing a SubscriptPair, in the future, a more elaborate 
        // partition algorithm will be incorporated along with a coupled
        // subscript test
        PrintTools.println("getSubscriptPartitions: subscriptPairs.size()=" +
                           subscriptPairs.size(), 2);
        for (int i = 0; i < subscriptPairs.size(); i++) {
            SubscriptPair pair = subscriptPairs.get(i);
            Set<SubscriptPair> new_partition = new HashSet<SubscriptPair>();
            new_partition.add(pair);
            // In order to test simpler ZIV subscripts first, we add them
            // to the beginning of the partition list
            // ------------------------------------------------------
            if (pair.getComplexity() == 0) {
                partitions.addFirst(new_partition);
            } else {
                partitions.addLast(new_partition);
            }
            // ------------------------------------------------------
        }
        return partitions;
    }

    /**
    * Invokes dependence test for the given list of partition.
    * TODO: better handling of coupled subscripts. For now, local dv list is
    * intersected before being merged with incoming direction vectors.
    */
    private boolean testPartition(List<SubscriptPair> partition,
                                  List<DependenceVector> dvs) {
        boolean ret = false;
        List<DependenceVector> dv = new ArrayList<DependenceVector>();
        for (int i = 0; i < partition.size(); i++) {
            SubscriptPair pair = partition.get(i);
            List<DependenceVector> dv2 = new ArrayList<DependenceVector>();
            int complexity = pair.getComplexity();
            if (complexity == 0) {
                PrintTools.printlnStatus(2, "** calling testZIV");
                ret |= testZIV(pair, dv2);
            } else {
                PrintTools.printlnStatus(2, "** calling testMIV: Complexity=",
                                         complexity);
                ret |= testMIV(pair, dv2);
            }
            mergeVectorSets(dv, dv2);
        }
        if (ret) {
            mergeVectorSets(dvs, dv);
        }
        return ret;
    }

    private boolean testSeparableSubscripts(Set<SubscriptPair> partition,
                                            List<DependenceVector> DVset) {
        boolean depExists;
        // iterate over partitions and get singletons
        List<DependenceVector> DV = new ArrayList<DependenceVector>();
        // get the first (AND ONLY) element
        SubscriptPair pair = partition.iterator().next();
        switch (pair.getComplexity()) {
        case 0:
            PrintTools.println("** calling testZIV", 2);
            depExists = testZIV(pair, DV);
            break;
        case 1:
        default:
            PrintTools.println("** calling testMIV: Complexity=" +
                    pair.getComplexity(), 2);
            depExists = testMIV(pair, DV);
            break;
        }
        if (!depExists) {
            return depExists;
        } else {
            this.mergeVectorSets(DVset, DV);
            return true;
        }
    }

    /**
     * For each vector in DVSet, replicate it ||DV|| times, merge each replica
     * with one vector in DV (DV is the set of vectors returned by the
     * dependence test) Add the new merged vector back to DVSet only if it is a
     * valid vector
     */
    private void mergeVectorSets(List<DependenceVector> dvs,
                                 List<DependenceVector> other) {
        if (!dvs.isEmpty()) {
            int size = dvs.size(), osize = other.size();
            while (size-- > 0) {
                DependenceVector dv = dvs.remove(0);
                for (int i = 0; i < osize; i++) {
                    DependenceVector dv1 = new DependenceVector(dv);
                    dv1.mergeWith(other.get(i));
                    if (dv1.isValid()) {
                        dvs.add(dv1);
                    }
                }
            }
        } else {
            dvs.addAll(other);
        }
        return;
    }

    private boolean testZIV(SubscriptPair pair,
                            List<DependenceVector> DV) {
        Expression subscript1 = pair.getSubscript1();
        Expression subscript2 = pair.getSubscript2();
        Expression expr_diff = Symbolic.subtract(subscript1, subscript2);
        if (expr_diff instanceof IntegerLiteral) {
            IntegerLiteral diff = (IntegerLiteral)expr_diff;
            if (diff.getValue() == 0) {
                // Need to assign all possible combinations of DVs to this
                // subscript pair
                DependenceVector dv = new DependenceVector(loop_nest);
                //for (Loop l : pair.getEnclosingLoopsList()) {
                //  dv.setDirection(l, DependenceVector.equal);
                //}
                DV.add(dv);
                return true;
            } else {
                return false;
            }
        } else {
            // Difference in expressions is symbolic, conservatively return
            // true
            DependenceVector dv = new DependenceVector(loop_nest);
            DV.add(dv);
            return true;
        }
    }

    /**
     * Having collected all information related to subscripts and enclosing
     * loops, this is the function that will call the dependence test for MIV
     * (and currently SIV) subscripts.
     */
    private boolean testMIV(SubscriptPair pair,
                            List<DependenceVector> dependence_vectors) {
        DDTest ddtest = null;
        ArrayList<DependenceVector> new_dv;
        if (ddtest_type == DDTEST_OMEGA) {
            // ERROR, how did we get here?
            PrintTools.println("Error in data dependence testing", 0);
            Tools.exit(0);
        }
        // Add other subscript by subscript dependence tests here when required
        // else if ...
        // By default, use Banerjee
        else if (ddtest_type == DDTEST_RANGE) {
            ddtest = RangeTest.getInstance(pair);
        } else if (ddtest_type == DDTEST_BANERJEE) {
            ddtest = new BanerjeeTest(pair);
        }
        if (ddtest.isTestEligible()) {
            new_dv = testAllDependenceVectors(ddtest);
            if (new_dv.size() == 0) {
                return false;
            } else {
                dependence_vectors.addAll(new_dv);
                return true;
            }
        } else {
            DependenceVector dv = new DependenceVector(loop_nest);
            dependence_vectors.add(dv);
            return true;
        }
    }

    /** 
     * Test all combinations of dependence vectors for the enclosing loop nest,
     * prune on direction vectors for which no dependence exists.
     */
    private ArrayList<DependenceVector>
            testAllDependenceVectors(DDTest ddtest) {
        ArrayList<DependenceVector> dv_list = new ArrayList<DependenceVector>();
        //create vector dv=(*,...,*);
        DependenceVector dv = new DependenceVector(loop_nest); 
        // test dependence vector tree starting at (*,*,*,....) vector
        if (ddtest.testDependence(dv)) {
            // Test entire tree only if dependence exists in the any(*)
            // direction
            testTree(ddtest, dv, 0, dv_list);
        }
        return dv_list;
    }

    // Tests if acc2 is reachable from acc1 within the body of the innermost
    // common loop. This test needs to be done for loop-independent
    // dependences.
    private Boolean accIsReachable() {
        Boolean ret;
        Statement stmt1 = acc1.getParentStatement();
        Statement stmt2 = acc2.getParentStatement();
        if (stmt1 == stmt2) {
            // In a same statement, consider the arc from a read to a write
            // a reachable arc. e.g, a[i] = a[i] + ...
            ret = (acc1.getAccessType() == DDArrayAccessInfo.read_type &&
                   acc2.getAccessType() == DDArrayAccessInfo.write_type);
        } else {
            ret = reachable_stmts.get(Arrays.asList(stmt1, stmt2));
            if (ret == null) {
                Loop inner = loop_nest.getLast();
                CFGraph cfg = body_graph.get(inner);
                if (cfg == null) {
                    cfg = new CFGraph(inner.getBody());
                    body_graph.put(inner, cfg);
                }
                DFANode node1 = cfg.getNodeWith("stmt", stmt1);
                DFANode node2 = cfg.getNodeWith("stmt", stmt2);
                if (node1 == null || node2 == null) {
                    ret = true; // conservative but safe answer
                } else {
                    ret = cfg.isReachable(node1, node2);
                }
                reachable_stmts.put(Arrays.asList(stmt1, stmt2), ret);
            }
        }
        return ret;
    }

    private void testTree(DDTest ddtest,
                          DependenceVector dv,
                          int pos,
                          ArrayList<DependenceVector> dv_list) {
        // Test the entire tree of dependence vectors, prune if dependence
        // doesn't exist at a given level i.e. don't explore the tree further
        for (int dir = DependenceVector.less;
                dir <= DependenceVector.greater; dir++) {
            Loop loop = loop_nest.get(pos);
            dv.setDirection(loop, dir);
            if (ddtest.testDependence(dv)) {
                DependenceVector dv_clone = new DependenceVector(dv);
                // Add to dependence vector list only if it does not contain
                // the 'any' (*) direction for all given loops
                if (!((dv_clone.getDirectionVector()).
                        containsValue(DependenceVector.any))) {
                    // Skips access pairs that do not follow the definition of
                    // data dependence.
                    if (!dv.isEqual() || accIsReachable()) {
                        dv_list.add(dv_clone);
                    }
                }
                // Dependence exists, hence test the child tree rooted at
                // current dv
                if ((pos + 1) < loop_nest.size()) {
                    testTree(ddtest, dv, pos + 1, dv_list);
                }
            }
            dv.setDirection(loop, DependenceVector.any);
        }
        return;
    }
}
