package cetus.application;

import cetus.hir.FunctionCall;
import java.util.HashMap;
import java.util.Map;

/**
 * This class provides a way to enrol the information on the third party functions
 * which may affect the reaching definition analysis.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public class ThirdPartyLibrary {

    private static Map<String, int[]> modIdxMap = new HashMap<String, int[]>();

    /**
     * Adds function names with parameter indices for side effect parameters
     * 
     * @param modIdxMap a map for <function_name, array_of_side_effect_parameter>
     */
    public static void addAll(Map<String, int[]> modIdxMap) {
        ThirdPartyLibrary.modIdxMap.putAll(modIdxMap);
    }

    /**
     * Adds a function name which has side effects and an array of indices on which the side effect parameters are
     * 
     * @param fName the function name
     * @param modIndices the array of indices for side effect parameters, which starts from '0'
     */
    public static void add(String fName, int[] modIndices) {
        ThirdPartyLibrary.modIdxMap.put(fName, modIndices);
    }

    /**
     * Used by def-use/use-def chain computation
     * 
     * @param fcall
     * @return
     */
    public static boolean contains(FunctionCall fcall) {
        return (ThirdPartyLibrary.modIdxMap.get(fcall.getName().toString()) != null);
    }

    /**
     * Used by def-use/use-def chain computation
     * 
     * @param fcall
     * @return
     */
    public static boolean hasSideEffectOnParameter(FunctionCall fcall) {
        if (!contains(fcall)) {
            return false;
        }
        return (modIdxMap.get(fcall.getName().toString()) != null);
    }

    /**
     * Used by def-use/use-def chain computation
     * 
     * @param fcall
     * @return
     */
    public static int[] getSideEffectParamIndices(FunctionCall fcall) {
        if (hasSideEffectOnParameter(fcall) == false) {
            return null;
        }
        return modIdxMap.get(fcall.getName().toString());
    }
}
