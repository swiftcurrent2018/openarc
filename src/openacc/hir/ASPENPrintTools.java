package openacc.hir;

import java.io.PrintWriter;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import cetus.hir.Symbol;

/**
* <b>ASPENPrintTools</b> provides tools that perform printing of collections of IR
* or debug messages.
*/
public final class ASPENPrintTools {

    // Short names for system properties
    public static final String line_sep = System.getProperty("line.separator");
    public static final String file_sep = System.getProperty("file.separator");
    public static final String path_sep = System.getProperty("path.separator");
    
    private ASPENPrintTools() {
    }

    /**
    * Prints a list of printable object to the specified print writer with a
    * separating stirng. If the list contains an object not printable, this
    * method throws a cast exception.
    * @param list the list of printable object.
    * @param w the target print writer.
    * @param sep the separating string.
    */
    public static void
            printListWithSeparator(List list, PrintWriter w, String sep) {
        if (list == null) {
            return;
        }
        int list_size = list.size();
        if (list_size > 0) {
            ((ASPENPrintable)list.get(0)).printASPENModel(w);
            for (int i = 1; i < list_size; i++) {
                w.print(sep);
                ((ASPENPrintable)list.get(i)).printASPENModel(w);
            }
        }
    }

    /**
    * Prints a list of printable object to the specified print writer with a
    * separating comma.
    *
    * @param list the list of printable object.
    * @param w the target print writer.
    */
    public static void printListWithComma(List list, PrintWriter w) {
        printListWithSeparator(list, w, ", ");
    }

    /**
    * Prints a list of printable object to the specified print writer with a
    * separating white space.
    *
    * @param list the list of printable object.
    * @param w the target print writer.
    */
    public static void printListWithSpace(List list, PrintWriter w) {
        printListWithSeparator(list, w, " ");
    }

    /**
    * Prints a list of printable object to the specified print writer without
    * any separating string.
    *
    * @param list the list of printable object.
    * @param w the target print writer.
    */
    public static void printList(List list, PrintWriter w) {
        printListWithSeparator(list, w, "");
    }

    /**
    * Prints a list of printable object to the specified print writer with a
    * separating new line character.
    *
    * @param list the list of printable object.
    * @param w the target print writer.
    */
    public static void printlnList(List list, PrintWriter w) {
        printListWithSeparator(list, w, line_sep);
        w.println("");
    }

    /**
    * Converts a collection of objects to a string with the given separator.
    * By default, the element of the collections are sorted alphabetically, and
    * any {@code Symbol} object is printed with its name.
    *
    * @param coll the collection to be converted.
    * @param separator the separating string.
    * @return the converted string.
    */
    public static String
            collectionToString(Collection coll, String separator) {
        String ret = "";
        if (coll == null || coll.isEmpty()) {
            return ret;
        }
        // Sort the collection first.
        TreeSet<String> sorted = new TreeSet<String>();
        for (Object o : coll) {
            if (o instanceof Symbol) {
                sorted.add(((Symbol)o).getSymbolName());
            } else if (o instanceof ASPENPrintable) {
                sorted.add(((ASPENPrintable)o).toASPENString());
            } else {
                sorted.add(o.toString());
            }
        }
        Iterator<String> iter = sorted.iterator();
        if (iter.hasNext()) {
            StringBuilder sb = new StringBuilder(80);
            sb.append(iter.next());
            while (iter.hasNext()) {
                sb.append(separator).append(iter.next());
            }
            ret = sb.toString();
        }
        return ret;
    }

    /**
    * Converts a list of objects to a string with the given separator.
    *
    * @param list the list to be converted.
    * @param separator the separating string.
    * @return the converted string.
    */
    public static String listToString(List list, String separator) {
        if (list == null || list.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder(80);
        String argStr; 
        Object o = list.get(0);
        if( o instanceof ASPENPrintable ) {
        	argStr = ((ASPENPrintable)o).toASPENString();
        } else {
        	argStr = o.toString();
        }
        sb.append(argStr);
        int list_size = list.size();
        for (int i = 1; i < list_size; i++) {
        	o = list.get(i);
        	if( o instanceof ASPENPrintable ) {
        		argStr = ((ASPENPrintable)o).toASPENString();
        	} else {
        		argStr = o.toString();
        	}
            sb.append(separator).append(argStr);
        }
        return sb.toString();
    }

    /** 
    * Converts a list of objects to a string. The difference from
    * {@code listToString} is that this method inserts the separating string
    * only if the heading string length is non-zero.
    * @param list the list to be converted.
    * @param separator the separating string.
    */
    public static String listToStringWithSkip(List list, String separator) {
        if (list == null || list.isEmpty()) {
            return "";
        }
        String prev; 
        Object o = list.get(0);
        if( o instanceof ASPENPrintable ) {
        	prev = ((ASPENPrintable)o).toASPENString();
        } else {
        	prev = o.toString();
        }
        StringBuilder sb = new StringBuilder(80);
        sb.append(prev);
        int list_size = list.size();
        for (int i = 1; i < list_size; i++) {
            if (prev.length() > 0) {
                sb.append(separator);
            }
            o = list.get(i);
            if( o instanceof ASPENPrintable ) {
            	prev = ((ASPENPrintable)o).toASPENString();
            } else {
            	prev = o.toString();
            }
            sb.append(prev);
        }
        return sb.toString();
    }
}
