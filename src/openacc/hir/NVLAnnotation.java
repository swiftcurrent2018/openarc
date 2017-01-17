package openacc.hir;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import cetus.hir.Annotatable;
import cetus.hir.PragmaAnnotation;
import cetus.hir.PrintTools;

/**
 * NVLAnnotation is used for internally representing NVL pragmas.
 * NVL pragmas are raw text right after parsing 
 * but converted to an internal annotation of this type.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov>
 *         Future TechnologiesGroup, Oak Ridge National Laboratory
 */
public class NVLAnnotation extends PragmaAnnotation
{
  private static final long serialVersionUID = 1L;

/**
 * NVL Annotation List.
 *
 * The following user directives are supported:
 * <p>     
 * #pragma nvl atomic \
 *   [heap(heap-pointer)] \
 *   [default(backup|backup_writeFirst|clobber|readonly)] \
 *   [backup(NVM-pointer)] [backup_writeFirst(NVM-pointer)] \
 *   [clobber(NVM-pointer)] [readonly(NVM-pointer)] \
 *   [mpiGroup(mpi-group-handle)]
 */

  // NVL directives
  public static final Set<String> nvl_directives =
    new HashSet<String>(Arrays.asList("atomic"));

  // Pragmas used without values
  private static final Set<String> no_values =
    new HashSet<String>();

  // Pragmas used with collection of values
  private static final Set<String> collection_values =
    new HashSet<String>(Arrays.asList("heap", "default",
                                      "backup", "backup_writeFirst",
                                      "clobber", "readonly", "mpiGroup"));

  // Pragmas used with optional value
  private static final Set<String> optional_values = 
    new HashSet<String>();

  //List used to set print orders.
  //Clauses not listed here may be printed in a random order.
  private static final List<String> print_order =
    new ArrayList<String>(Arrays.asList( "atomic", "heap", "default",
                                         "backup", "backup_writeFirst",
                                         "clobber", "readonly", "mpiGroup" ));

  /**
   * Constructs an empty NVL annotation.
   */
  public NVLAnnotation() {
    super("nvl");
  }

  /**
   * Check whether this annotation is applicable to the input Annotatable, <var>at</var>.
   * For stand-alone annotations, this will always return false.
   * 
   * @param at Annotatable to which this annotation will be attached.
   * @return true if this annotation is applicable to the input Annotatable <var>at</var>
   */
  public boolean isValidTo(Annotatable at) {
    return true;
  }

  /**
   * Returns the string representation of this annotation.
   *
   * @return the string representation.
   */
  public String toString() {
    if ( skip_print )
      return "";

    StringBuilder str = new StringBuilder(80);

    str.append(super.toString());

    Set<String> directiveSet = new HashSet<String>();
    directiveSet.addAll(keySet());
    directiveSet.remove("pragma");

    for( String key : print_order ) {
      if( directiveSet.contains(key) ) {
        printDirective(key, str);
        directiveSet.remove(key);
      }
    }
    if( !directiveSet.isEmpty() ) {
      for( String key: directiveSet ) {
        printDirective(key, str);
      }
    }

    return str.toString();
  }

  private void printDirective(String key, StringBuilder str) {
    if ( no_values.contains(key) )
      str.append(" "+key);
    else if ( collection_values.contains(key) ) {
      str.append(" "+key+"(");
      Object value = get(key);
      if ( value instanceof Collection )
        if ( value instanceof List ) {
          str.append(PrintTools.listToString((List)value, ", "));
        } else {
          str.append(PrintTools.collectionToString((Collection)value, ", "));
        }
      else // e.g., label
        str.append(value);
      str.append(")");
    }
    else if ( optional_values.contains(key) ) {
      str.append(" "+key);
      Object tVal = get(key);
      if ( tVal != null && !tVal.equals("true") && !tVal.equals("_directive") 
          && !tVal.equals("_clause") )
        str.append("("+tVal+")");
    }
    else {
      //If this annotation contains annotatable object as value, 
      //printing the value will cause infinite recursion; skip
      //printing annotatable object.
      Object tObj = get(key);
      if ( !(tObj instanceof Annotatable) ) {
      str.append(" "+key);
      if ( (tObj != null) && (!"true".equals(tObj) && !"_directive".equals(tObj) 
          && !"_clause".equals(tObj)) )
        str.append("("+tObj+")");
      }
    }
  }
}
