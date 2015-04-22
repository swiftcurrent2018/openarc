package openacc.hir;

import java.io.PrintWriter;

public interface ASPENPrintable {

	void printASPENModel(PrintWriter o);	
	String toASPENString();

}
