package openacc.analysis;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.lang.Cloneable;
import java.util.ArrayList;
import java.util.List;
import cetus.hir.Expression;
import cetus.hir.PrintTools;

/**
* Represents OpenACC subarray, which is an array name followed by an extended array range specification in brackets,
* with start and length, such as arr[2:n].
* 
* @author Seyong Lee <lees2@ornl.gov>
*         Future Technologies Group, Oak Ridge National Laboratory
*/
public class SubArray implements Cloneable {

	private Expression array;
	private int dimension; //dimension of array; -1 if dimension is not-known
	                       //                     0 if scalar variable
	private ArrayList<Expression> startIndices;
	private ArrayList<Expression> lengths;
	private boolean printRange;
	
    /**
    * Creates a subarray with an array name and start and length expression
    *
    * @param arrayName An array name
    * @param start The expression of start index of the subarray
    * @param length The expression of the length of the subarray
    */
    public SubArray(Expression arrayName, Expression start, Expression length) {
    	array = arrayName;
    	startIndices = new ArrayList<Expression>();
    	lengths = new ArrayList<Expression>();
    	startIndices.add(start);
    	lengths.add(length);
    	dimension = 1;
    	printRange = true;
    }
    
    /**
    * Creates a subarray with an array name but no start or length expression
    *
    * @param arrayName An array name
    */
    public SubArray(Expression arrayName) {
    	array = arrayName;
    	startIndices = new ArrayList<Expression>();
    	lengths = new ArrayList<Expression>();
    	dimension = -1;
    	printRange = true;
    }
    
    /**
    * Creates a subarray with an array name with dimension size; useful to create subarray for scalar variable
    *
    * @param arrayName An array name
    * @param dimSize dimension size of this subarray
    */
    public SubArray(Expression arrayName, int dimSize) {
    	array = arrayName;
    	if( dimSize > 0 ) {
    		startIndices = new ArrayList<Expression>(dimSize);
    		lengths = new ArrayList<Expression>(dimSize);
    	} else {
    		startIndices = new ArrayList<Expression>();
    		lengths = new ArrayList<Expression>();
    	}
    	dimension = dimSize;
    	printRange = true;
    }
    
    /**
     * Set array name of this SubArray
     * 
     * @param aName array name Expression
     */
    public void setArrayName(Expression aName) {
    	array = aName;
    }

    /**
    * Inserts a new range (start, length) at the end of startIndices and lengths lists, respectively,
    * increasing the dimension of the subarray.
    *
    * @param range a new range list with start and length as its first and second elements, respectively
    */
    public void addRange(List<Expression> range) {
        startIndices.add(range.get(0));
        lengths.add(range.get(1));
        if( dimension <= 0 ) dimension = 1;
        else dimension++;
    }
    
    /**
    * Inserts a new range (start, length) at the index of startIndices and lengths lists, respectively.
    * 
    * @throws IndexOutOfBoundsException if <b>index</b> is out-of-range
    * @param index index to insert the new range
    * @param range a new range list with start and length as its first and second elements, respectively
    */
    public void setRange(int index, List<Expression> range) {
    	if( (dimension <= index) || (index < 0) ) throw new IndexOutOfBoundsException();
    	else {
    		startIndices.remove(index);
    		startIndices.add(index, range.get(0));
    		lengths.remove(index);
    		lengths.add(index, range.get(1));
    		//dimension = startIndices.size();
    	}
    }
    
    /**
    * Set start indices and lengths for each dimension of this subarray.
    * 
    * @throws IllegalArgumentException if the length of <b>startList</b> differs from that of <b>lengthList</b>
    * @param startList a list of start indices for each dimension
    * @param lengthList a list of length indices for each dimension
    */
    public void setRange(List<Expression> startList, List<Expression> lengthList) {
    	if( startList.size() != lengthList.size() ) throw new IllegalArgumentException();
    	else {
    		startIndices.clear();
    		lengths.clear();
    		startIndices.addAll(startList);
    		lengths.addAll(lengthList);
    		dimension = startList.size();
    	}
    }

    /** Returns a clone of the array access */
    @Override
    public SubArray clone() {
    	SubArray clonedSubArray = new SubArray(getArrayName().clone(), dimension);
    	if( dimension > 0 ) {
    		List<Expression> clonedStarts = new ArrayList<Expression>();
    		List<Expression> clonedLengths = new ArrayList<Expression>();
    		Expression tExp = null;
    		Expression clonedExp = null;
    		for(int i=0; i<dimension; i++) {
    			tExp = startIndices.get(i);
    			if( tExp == null ) {
    				clonedExp = null;
    			} else {
    				clonedExp = tExp.clone();
    			}
    			clonedStarts.add(clonedExp);
    			tExp = lengths.get(i);
    			if( tExp == null ) {
    				clonedExp = null;
    			} else {
    				clonedExp = tExp.clone();
    			}
    			clonedLengths.add(clonedExp);
    		}
    		clonedSubArray.setRange(clonedStarts, clonedLengths);
    	}
    	return clonedSubArray;
    }
    
    public boolean equals(Object o) {
    	if (o == null || this.getClass() != o.getClass()) {
    		return false;
    	}
    	List<Expression> oStartIndices = ((SubArray)o).getStartIndices();
    	List<Expression> oLengths = ((SubArray)o).getLengths();
    	Expression oArray = ((SubArray)o).getArrayName();
    	if (array == null) {
    		if (oArray != null) {
    			return false;
    		}
    	} else if( !array.equals(oArray) ) {
    			return false;
    	}
    	if (startIndices == null) {
    		if (oStartIndices != null) {
    			return false;
    		}
    	} else if( !startIndices.equals(oStartIndices) ) {
    			return false;
    	}
    	if (lengths == null) {
    		if (oLengths != null) {
    			return false;
    		}
    	} else if( !lengths.equals(oLengths) ) {
    			return false;
    	}
    	return true;
    }
    
    public int hashCode() {
        String s = toString();
        int h = 0;
        for (int i = 0; i < s.length(); i++) {
            h = 31 * h + s.charAt(i);
        }
        return h;
    }
    
    /**
     * Set whether to print subarray range or not
     * 
     * @param printOption value to set printRange member variable
     */
    public void setPrintRange(boolean printOption) {
    	printRange = printOption;
    }

    /**
    * Returns a string representation of the subarray.
    * 
    * @return the string representation.
    */
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        sb.append(getArrayName());
        if( printRange && (dimension > 0) ) {
        	for(int i=0; i<dimension; i++) {
        		sb.append("[");
        		Expression exp = startIndices.get(i);
        		if( exp == null ) {
        			sb.append(" ");
        		} else {
        			sb.append(exp.toString());
        		}
        		sb.append(":");
        		exp = lengths.get(i);
        		if( exp == null ) {
        			sb.append(" ");
        		} else {
        			sb.append(exp.toString());
        		}
        		sb.append("]");
        	}
        }
        return sb.toString();
    }

    /**
    * Returns the expression being indexed. Most often, the expression will be
    * an Identifier, but any expression that evaluates to an address is allowed.
    *
    * @return the expression being indexed.
    */
    public Expression getArrayName() {
        return array;
    }
    
    /**
     * Return the range of index's dimension.
     * 
     * @param index index of the dimension whose range will be returned
     * @return list of (startIndex, length) of the index's dimension
     */
    public List<Expression> getRange(int index) {
    	if( (dimension <= index) || (index < 0) ) throw new IndexOutOfBoundsException();
    	else {
    		List<Expression> range = new ArrayList<Expression>();
    		range.add(startIndices.get(index));
    		range.add(lengths.get(index));
    		return range;
    	}
    }
    
    /**
     * Return the list of start indices
     * 
     * @return list of start indices
     */
    public List<Expression> getStartIndices() {
    	return startIndices;
    }
    
    /**
     * Return the list of lengths
     * 
     * @return list of lengths
     */
    public List<Expression> getLengths() {
    	return lengths;
    }

    /**
    * Returns the number of index expressions used in this array access.
    *
    * @return the number of index expressions.
    */
    public int getArrayDimension() {
        return dimension;
    }

}
