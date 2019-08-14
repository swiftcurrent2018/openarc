/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

PROJECT:        Cetus C Parser
FILE:           NewCParser.g

AUTHOR:         Sang Ik Lee (sangik@purdue.edu)

DESCRIPTION:

        This file implements a C language parser that generates
        Cetus internal representation using public constructors in
        package cetus.hir
        The grammar support ANSI C language with some additions
        for GNU C extentions and few ISO C99 features.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/*
Copyright (c) 1998-2000, Non, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in
  the documentation and/or other materials provided with the
  distribution.

  All advertising materials mentioning features or use of this
  software must display the following acknowledgement:

    This product includes software developed by Non, Inc. and
    its contributors.

  Neither name of the company nor the names of its contributors
  may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COMPANY OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Copyright (c) Non, Inc. 1997 -- All Rights Reserved

PROJECT:        C Compiler
MODULE:         Parser
FILE:           stdc.g

AUTHOR:         John D. Mitchell (john@non.net), Jul 12, 1997

REVISION HISTORY:

        Name    Date            Description
        ----    ----            -----------
        JDM     97.07.12        Initial version.
        JTC     97.11.18        Declaration vs declarator & misc. hacking.
        JDM     97.11.20        Fixed:  declaration vs funcDef,
                                        parenthesized expressions,
                                        declarator iteration,
                                        varargs recognition,
                                        empty source file recognition,
                                        and some typos.


DESCRIPTION:

        This grammar supports the Standard C language.

        Note clearly that this grammar does *NOT* deal with
        preprocessor functionality (including things like trigraphs)
        Nor does this grammar deal with multi-byte characters nor strings
        containing multi-byte characters [these constructs are "exercises
        for the reader" as it were :-)].

        Please refer to the ISO/ANSI C Language Standard if you believe
        this grammar to be in error.  Please cite chapter and verse in any
        correspondence to the author to back up your claim.

TODO:

        - typedefName is commented out, needs a symbol table to resolve
        ambiguity.

        - trees

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
header
{
package cetus.base.grammars;
}

{
import java.io.*;
import antlr.CommonAST;
import antlr.DumpASTVisitor;
import java.util.*;
import cetus.hir.*;
import openacc.hir.CUDASpecifier;
import openacc.hir.OpenCLSpecifier;
import static cetus.hir.AttributeSpecifier.Attribute;
@SuppressWarnings({"unchecked", "cast"})
}

class NewCParser extends Parser;

options
{
k = 2;
exportVocab = NEWC;
// Copied following options from java grammar.
codeGenMakeSwitchThreshold = 2;
codeGenBitsetTestThreshold = 3;
}

{
Expression baseEnum = null,curEnum = null;
NewCLexer curLexer=null;
boolean isFuncDef = false;
boolean isExtern = false;
PreprocessorInfoChannel preprocessorInfoChannel = null;
SymbolTable symtab = null;
CompoundStatement curr_cstmt = null;
boolean hastypedef = false;
HashMap typetable = null;
List currproc = new ArrayList();
Declaration prev_decl = null;
boolean old_style_func = false;
HashMap func_decl_list = new HashMap();

public void getPreprocessorInfoChannel(PreprocessorInfoChannel preprocChannel)
{
  preprocessorInfoChannel = preprocChannel;
}

public void setLexer(NewCLexer lexer)
{
  curLexer=lexer;
  curLexer.setParser(this);
}

public NewCLexer getLexer()
{
  return curLexer;
}

/*
 * Retreive all buffered pragmas and comments up to this
 * token number
 */
public List getPragma(int a)
{
  return
      preprocessorInfoChannel.extractLinesPrecedingTokenNumber(new Integer(a));
}

/*
 * Add pragmas and line directives as PreAnnotation
 */
public void putPragma(Token sline, SymbolTable sym)
{
  List v  = null;
  // Get token number and get all buffered information
  v = getPragma(((CToken)sline).getTokenNumber());

  // Go though the list of pragmas and comments
  Pragma p = null;
  PreAnnotation anote = null;
  int vsize = v.size();
  for (int i = 0; i < vsize; i++) {
    p = (Pragma)v.get(i);
    anote = new PreAnnotation(p.str);
    anote.setPrintMethod(PreAnnotation.print_raw_method);
    // Add PreAnnotation as Statement in a block
    if (sym instanceof CompoundStatement)
      ((CompoundStatement)sym).addStatement(new DeclarationStatement(anote));
    // Add PreAnnotation in other SymbolTables
    else
      sym.addDeclaration(anote);
  }
}

// Suppport C++-style single-line comments?
public static boolean CPPComments = true;
public Stack symtabstack = new Stack();
public Stack typestack = new Stack();

public void enterSymtab(SymbolTable curr_symtab)
{
  symtabstack.push(symtab);
  typetable = new HashMap();
  typestack.push(typetable);
  symtab = curr_symtab;
}

public void exitSymtab()
{
  Object o = symtabstack.pop();
  if (o != null) {
    typestack.pop();
    typetable = (HashMap)(typestack.peek());
    symtab = (SymbolTable)o;
  }
}

public boolean isTypedefName(String name)
{
  //System.err.println("Typename "+name);
  int n = typestack.size()-1;
  Object d = null;
  while(n>=0) {
    d = ((HashMap)(typestack.get(n))).get(name);
    if (d != null )
      return true;
    n--;
  }
  if (name.equals("__builtin_va_list"))
    return true;
  // [NVL support added by Joel E. Denny]
  if (name.equals("__builtin_nvl_heap"))
    return true;

  //System.err.println("Typename "+name+" not found");
  return false;
}

int traceDepth = 0;

public void reportError(RecognitionException ex)
{
  try {
    System.err.println("Cetus Parsing Error: " + "Exception Type: "
        + ex.getClass().getName());
    System.err.println("Source: " + getLexer().lineObject.getSource()
        + " Line:" + ex.getLine() + " Column: " + ex.getColumn()
        + " token name:" + tokenNames[LA(1)]);
    ex.printStackTrace(System.err);
    Tools.exit(1);
  } catch (TokenStreamException e) {
    System.err.println("Cetus Parsing Error: "+ex);
    ex.printStackTrace(System.err);
    Tools.exit(1);
  }
}

public void reportError(String s)
{
  System.err.println("Cetus Parsing Error from String: " + s);
}

public void reportWarning(String s)
{
  System.err.println("Cetus Parsing Warning from String: " + s);
}

public void match(int t) throws MismatchedTokenException
{
  boolean debugging = false;
  if ( debugging ) {
    for (int x=0; x<traceDepth; x++)
      System.out.print(" ");
    try {
      System.out.println("Match(" + tokenNames[t] + ") with LA(1)="
          + tokenNames[LA(1)] + ((inputState.guessing>0)?
          " [inputState.guessing " + inputState.guessing + "]":""));
    } catch (TokenStreamException e) {
      System.out.println("Match("+tokenNames[t]+") "
          + ((inputState.guessing>0)?
          " [inputState.guessing "+ inputState.guessing + "]":""));
    }
  }
  try {
    if ( LA(1)!=t ) {
      if ( debugging ) {
        for (int x=0; x<traceDepth; x++)
          System.out.print(" ");
        System.out.println("token mismatch: "+tokenNames[LA(1)]
            + "!=" + tokenNames[t]);
      }
      throw new MismatchedTokenException
          (tokenNames, LT(1), t, false, getFilename());
    } else {
      // mark token as consumed -- fetch next token deferred until LA/LT
      consume();
    }
  } catch (TokenStreamException e) {
  }
}

public void traceIn(String rname)
{
  traceDepth += 1;
  for (int x=0; x<traceDepth; x++)
    System.out.print(" ");
  try {
    System.out.println("> "+rname+"; LA(1)==("+ tokenNames[LT(1).getType()]
        + ") " + LT(1).getText() + " [inputState.guessing "
        + inputState.guessing + "]");
  } catch (TokenStreamException e) {
  }
}

public void traceOut(String rname)
{
  for (int x=0; x<traceDepth; x++)
    System.out.print(" ");
  try {
    System.out.println("< "+rname+"; LA(1)==("+ tokenNames[LT(1).getType()]
        + ") " + LT(1).getText() + " [inputState.guessing "
        + inputState.guessing + "]");
  } catch (TokenStreamException e) {
  }
  traceDepth -= 1;
}

/* Normalizes switch statement by removing unncessary compound statement */
private void unwrapSwitch(SwitchStatement swstmt) {
    List<CompoundStatement> cstmts = (new DFIterator<CompoundStatement>(
            swstmt, CompoundStatement.class)).getList();
    cstmts.remove(0);
    for (int i = cstmts.size()-1; i >= 0; i--) {
        CompoundStatement cstmt = cstmts.get(i);
        if (cstmt.getParent() instanceof CompoundStatement &&
            !IRTools.containsClass(cstmt, VariableDeclaration.class)) {
            CompoundStatement parent = (CompoundStatement)cstmt.getParent();
            List<Traversable> children = cstmt.getChildren();
            while (!children.isEmpty()) {
                Statement child = (Statement)children.get(0);
                child.detach();
                parent.addStatementBefore(cstmt, child);
            }
            cstmt.detach();
        }
    }
}

}

/* TranslationUnit */
translationUnit [TranslationUnit init_tunit] returns [TranslationUnit tunit]
        {
            /* build a new Translation Unit */
            if (init_tunit == null)
                tunit = new TranslationUnit(getLexer().originalSource);
            else
                tunit = init_tunit;
            enterSymtab(tunit);
        }
        :
        ( externalList[tunit] )?  /* Empty source files are allowed.  */
        {exitSymtab();}
        ;


externalList [TranslationUnit tunit]
        //{boolean flag = true;}
        :
        (
        externalDef[tunit]
        )+
        ;


/* Declaration */
externalDef [TranslationUnit tunit]
{Declaration decl = null;}
        :
        ( "typedef" | declaration ) => decl=declaration
{
if (decl != null) {
  //PrintTools.printStatus("Adding Declaration: ",3);
  //PrintTools.printlnStatus(decl,3);
  tunit.addDeclaration(decl);
}
}
        |
        ( functionPrefix ) => decl=functionDef
{
//PrintTools.printStatus("Adding Declaration: ",3);
//PrintTools.printlnStatus(decl,3);
tunit.addDeclaration(decl);
}
        |
        decl=typelessDeclaration
{
//PrintTools.printStatus("Adding Declaration: ",3);
//PrintTools.printlnStatus(decl,3);
tunit.addDeclaration(decl);
}
        |
        asm_expr // not going to handle this now
        |
        esemi:SEMI // empty declaration - ignore it
        {putPragma(esemi,symtab);}
        ;


/* these two are here because GCC allows "cat = 13;" as a valid program! */
functionPrefix
{Declarator decl = null;}
        :
        (
        (functionDeclSpecifiers) => functionDeclSpecifiers
        |
        //epsilon
        )
        // Passing "null" could cause a problem
        decl = declarator
        ( declaration )* (VARARGS)? ( SEMI )*
        flcurly:LCURLY {putPragma(flcurly,symtab);}
        ;


/* Type Declaration */
typelessDeclaration returns [Declaration decl]
{decl=null; List idlist=null;}
        :
        idlist=initDeclList tdsemi:SEMI {putPragma(tdsemi,symtab);}
        /* Proper constructor missing */
{decl = new VariableDeclaration(new ArrayList(),idlist); }
        ;


// going to ignore this
asm_expr
{Expression expr1 = null;}
        :
        "asm"^ ("volatile")? asmlcurly:LCURLY {putPragma(asmlcurly,symtab);}expr1=expr RCURLY ( SEMI )+
        ;


/* Declaration */
declaration returns [Declaration bdecl]
{bdecl=null; List dspec=null; List idlist=null;}
        :
        dspec=declSpecifiers
        (
        // Pass specifier to add to Symtab
        idlist=initDeclList
        )?
{
if (idlist != null) {
  if (old_style_func) {
    Declarator d  = null;
    Declaration newdecl = null;
    int idlist_size = idlist.size();
    for (int i = 0; i < idlist_size; i++) {
      d = (Declarator)idlist.get(i);
      newdecl = new VariableDeclaration(dspec,d);
      func_decl_list.put(d.getID().toString(),newdecl);
    }
  bdecl = null;
  } else
    bdecl = new VariableDeclaration(dspec,idlist);
  prev_decl = null;
} else {
  // Looks like a forward declaration
  if (prev_decl != null) {
    bdecl = prev_decl;
    prev_decl = null;
  }
}
hastypedef = false;
}
        ( dsemi:SEMI )+
{
int sline = 0;
sline = dsemi.getLine();
putPragma(dsemi,symtab);
hastypedef = false;
}
        ;


/* Specifier List */
// The main type information
// [DEBUG by Seyong Lee] old one is deprecated.
declSpecifiers_old returns [List decls]
{decls = new ArrayList(); Specifier spec = null; Specifier temp=null;}
        :
        (
        // this loop properly aborts when it finds a non-typedefName ID MBZ
        options {warnWhenFollowAmbig = false;}
        :
        /* Modifier */
        spec = storageClassSpecifier
{decls.add(spec);}
        |
        /* Modifier */
        spec = typeQualifier
{decls.add(spec);}
        |
        /* SubType */
        ( "struct" | "union" | "enum" | typeSpecifier ) =>
        temp = typeSpecifier
{decls.add(temp);}
        // MinGW specific
        |
        attributeDecl
        |
        // VC++ specific
        vcDeclSpec
        )+
        ;
        
/* Specifier List */
// The main type information
// [DEBUG by Seyong Lee] new definition of declSpecifiers
declSpecifiers returns [List decls]
{decls = new ArrayList(); Specifier spec = null; Specifier temp=null;}
        :
        (
        // this loop properly aborts when it finds a non-typedefName ID MBZ
        options {warnWhenFollowAmbig = false;}
        :
        ( 
        // List of non-type specifiers
        /* Modifier */
        spec = storageClassSpecifier
{decls.add(spec);}
        |
        /* Modifier */
        spec = typeQualifier
{decls.add(spec);}
        |
        attributeDecl
        |
        // VC++ specific
        vcDeclSpec
        )*
        
        //There should exist at least one type specifier.
        /* SubType */
        ( "struct" | "union" | "enum" | typeSpecifier ) =>
        temp = typeSpecifier
{decls.add(temp);}

		(
		// List of non-typeDefName specifiers
        /* Modifier */
        spec = storageClassSpecifier
{decls.add(spec);}
        |
        /* Modifier */
        spec = typeQualifier
{decls.add(spec);}
        |
        /* SubType */
        ( "struct" | "union" | "enum" | typeSpecifier_noTypeDefName ) =>
        temp = typeSpecifier_noTypeDefName
{decls.add(temp);}
        // MinGW specific
        |
        attributeDecl
        |
        // VC++ specific
        vcDeclSpec
        )*
        )
        ;


/*********************************
 * Specifiers                    *
 *********************************/


storageClassSpecifier returns [Specifier cspec]
{cspec= null;}
        :
        "auto"
{cspec = Specifier.AUTO;}
        |
        "register"
{cspec = Specifier.REGISTER;}
        |
        scstypedef:"typedef"
{cspec = Specifier.TYPEDEF; hastypedef = true; putPragma(scstypedef,symtab);}
        |
        cspec = functionStorageClassSpecifier
        |
        // [Support for GCC extension __thread added by Joel E. Denny]
        "__thread"
{cspec = Specifier.THREAD;}
        ;


functionStorageClassSpecifier returns [Specifier type]
{type= null;}
        :
        "extern"
{type= Specifier.EXTERN;}
        |
        "static"
{type= Specifier.STATIC;}
        |
        "inline"
{type= Specifier.INLINE;}
        ;


// [NVL support added by Joel E. Denny]
typeQualifier returns [Specifier tqual]
{tqual=null;}
        :
        "const"
{tqual = Specifier.CONST;}
        |
        "volatile"
{tqual = Specifier.VOLATILE;}
        |
        "restrict"
{tqual = Specifier.RESTRICT;}
        |
        "__nvl__"
{tqual = Specifier.NVL;}
        |
        "__nvl_wp__"
{tqual = Specifier.NVL_WP;}
        ;


// A Type Specifier (basic type and user type)
/***************************************************
 * Should a basic type be an int value or a class ? *
 ****************************************************/
 //[DEBUG by Seyong Lee] typeName is replaced by typeName2
typeSpecifier returns [Specifier types]
{
types = null;
String tname = null;
Expression expr1 = null;
List tyname = null;
boolean typedefold = false;
}
        :
{typedefold = hastypedef; hastypedef = false;}
        (
        "void"
{types = Specifier.VOID;}
        |
        "char"
{types = Specifier.CHAR;}
        |
        "short"
{types = Specifier.SHORT;}
        |
        "int"
{types = Specifier.INT;}
        |
        "long"
{types = Specifier.LONG;}
        |
        "float"
{types = Specifier.FLOAT;}
        |
        "double"
{types = Specifier.DOUBLE;}
        |
        "signed"
{types = Specifier.SIGNED;}
        |
        "unsigned"
{types = Specifier.UNSIGNED;}
        /* C99 built-in type support */
        |
        "_Bool"
{types = Specifier.CBOOL;}
        |
        "_Complex"
{types = Specifier.CCOMPLEX;}
        |
        "_Imaginary"
{types = Specifier.CIMAGINARY;}
		|
        "__thread"
{types = Specifier.THREAD;}
		|
        "__nvl__"
{types = Specifier.NVL;}
		|
        "__nvl_wp__"
{types = Specifier.NVL_WP;}
		|
        "int8_t"
{types = Specifier.INT8_T;}
		|
        "uint8_t"
{types = Specifier.UINT8_T;}
		|
        "int16_t"
{types = Specifier.INT16_T;}
		|
        "uint16_t"
{types = Specifier.UINT16_T;}
		|
        "int32_t"
{types = Specifier.INT32_T;}
		|
        "uint32_t"
{types = Specifier.UINT32_T;}
		|
        "int64_t"
{types = Specifier.INT64_T;}
		|
        "uint64_t"
{types = Specifier.UINT64_T;}
		|
        "size_t"
{types = Specifier.SIZE_T;}
		|
        "_Float128"
{types = Specifier.FLOAT128;}
		|
        "__float128"
{types = Specifier.__FLOAT128;}
		|
        "__float80"
{types = Specifier.__FLOAT80;}
		|
        "__ibm128"
{types = Specifier.__IBM128;}
		|
        "_Float16"
{types = Specifier.FLOAT16;}
		|
        "__global__"
{types = CUDASpecifier.CUDA_GLOBAL;}
		|
        "__shared__"
{types = CUDASpecifier.CUDA_SHARED;}
		|
        "__host__"
{types = CUDASpecifier.CUDA_HOST;}
		|
        "__device__"
{types = CUDASpecifier.CUDA_DEVICE;}
		|
        "__constant__"
{types = CUDASpecifier.CUDA_CONSTANT;}
		|
        "__noinline__"
{types = CUDASpecifier.CUDA_NOINLINE;}
		|
        "__kernel"
{types = OpenCLSpecifier.OPENCL_KERNEL;}
		|
        "__global"
{types = OpenCLSpecifier.OPENCL_GLOBAL;}
		|
        "__local"
{types = OpenCLSpecifier.OPENCL_LOCAL;}
		|
        "__constant"
{types = OpenCLSpecifier.OPENCL_CONSTANT;}
        |
        types = structOrUnionSpecifier
        ( options{warnWhenFollowAmbig=false;}: attributeDecl )*
        |
        types = enumSpecifier
        |
        types = typedefName
        |
        /* Maybe unused */
        /*
        "typeof"^ LPAREN
        ( ( typeName2 ) => tyname=typeName2 | expr1=expr )
        RPAREN
        |
        */
        "__complex"
{types = Specifier.DOUBLE;}
        )
{hastypedef = typedefold;}
        ;
        
 //[DEBUG: Added by Seyong Lee] typeSpecifier excluding typedefname
typeSpecifier_noTypeDefName returns [Specifier types]
{
types = null;
String tname = null;
Expression expr1 = null;
List tyname = null;
boolean typedefold = false;
}
        :
{typedefold = hastypedef; hastypedef = false;}
        (
        "void"
{types = Specifier.VOID;}
        |
        "char"
{types = Specifier.CHAR;}
        |
        "short"
{types = Specifier.SHORT;}
        |
        "int"
{types = Specifier.INT;}
        |
        "long"
{types = Specifier.LONG;}
        |
        "float"
{types = Specifier.FLOAT;}
        |
        "double"
{types = Specifier.DOUBLE;}
        |
        "signed"
{types = Specifier.SIGNED;}
        |
        "unsigned"
{types = Specifier.UNSIGNED;}
        /* C99 built-in type support */
        |
        "_Bool"
{types = Specifier.CBOOL;}
        |
        "_Complex"
{types = Specifier.CCOMPLEX;}
        |
        "_Imaginary"
{types = Specifier.CIMAGINARY;}
		|
        "__thread"
{types = Specifier.THREAD;}
		|
        "__nvl__"
{types = Specifier.NVL;}
		|
        "__nvl_wp__"
{types = Specifier.NVL_WP;}
		|
        "int8_t"
{types = Specifier.INT8_T;}
		|
        "uint8_t"
{types = Specifier.UINT8_T;}
		|
        "int16_t"
{types = Specifier.INT16_T;}
		|
        "uint16_t"
{types = Specifier.UINT16_T;}
		|
        "int32_t"
{types = Specifier.INT32_T;}
		|
        "uint32_t"
{types = Specifier.UINT32_T;}
		|
        "int64_t"
{types = Specifier.INT64_T;}
		|
        "uint64_t"
{types = Specifier.UINT64_T;}
		|
        "size_t"
{types = Specifier.SIZE_T;}
		|
        "_Float128"
{types = Specifier.FLOAT128;}
		|
        "__float128"
{types = Specifier.__FLOAT128;}
		|
        "__float80"
{types = Specifier.__FLOAT80;}
		|
        "__ibm128"
{types = Specifier.__IBM128;}
		|
        "_Float16"
{types = Specifier.FLOAT16;}
		|
        "__global__"
{types = CUDASpecifier.CUDA_GLOBAL;}
		|
        "__shared__"
{types = CUDASpecifier.CUDA_SHARED;}
		|
        "__host__"
{types = CUDASpecifier.CUDA_HOST;}
		|
        "__device__"
{types = CUDASpecifier.CUDA_DEVICE;}
		|
        "__constant__"
{types = CUDASpecifier.CUDA_CONSTANT;}
		|
        "__noinline__"
{types = CUDASpecifier.CUDA_NOINLINE;}
		|
        "__kernel"
{types = OpenCLSpecifier.OPENCL_KERNEL;}
		|
        "__global"
{types = OpenCLSpecifier.OPENCL_GLOBAL;}
		|
        "__local"
{types = OpenCLSpecifier.OPENCL_LOCAL;}
		|
        "__constant"
{types = OpenCLSpecifier.OPENCL_CONSTANT;}
        |
        types = structOrUnionSpecifier
        ( options{warnWhenFollowAmbig=false;}: attributeDecl )*
        |
        types = enumSpecifier
        |
        /* Maybe unused */
        /*
        "typeof"^ LPAREN
        ( ( typeName2 ) => tyname=typeName2 | expr1=expr )
        RPAREN
        |
        */
        "__complex"
{types = Specifier.DOUBLE;}
        )
{hastypedef = typedefold;}
        ;


typedefName returns[Specifier name]
{name = null;}
        :
{isTypedefName ( LT(1).getText() )}?
        i:ID
        //{ ## = #(#[NTypedefName], #i); }
{name = new UserSpecifier(new NameID(i.getText()));}
        ;


structOrUnion returns [int type]
{type=0;}
        :
        "struct"
{type = 1;}
        |
        "union"
{type = 2;}
        ;


/* A User Type */
structOrUnionSpecifier returns [Specifier spec]
{
ClassDeclaration decls = null;
String name=null;
int type=0;
spec = null;
int linenum = 0;
}
        :
        type=structOrUnion!
        // [Attributes added by Joel E. Denny]
        (attributeDecl)?
        (
        //Named stucture with body
        ( ID LCURLY ) => i:ID l:LCURLY
{
name = i.getText();linenum = i.getLine(); putPragma(i,symtab);
String sname = null;
if (type == 1) {
  decls = new ClassDeclaration(ClassDeclaration.STRUCT, new NameID(name));
  spec = new UserSpecifier(new NameID("struct "+name));
} else {
  decls = new ClassDeclaration(ClassDeclaration.UNION, new NameID(name));
  spec = new UserSpecifier(new NameID("union "+name));
}
}
        ( structDeclarationList[decls] )?
{
if (symtab instanceof ClassDeclaration) {
  int si = symtabstack.size()-1;
  for (;si>=0;si--) {
    if (!(symtabstack.get(si) instanceof ClassDeclaration)) {
      ((SymbolTable)symtabstack.get(si)).addDeclaration(decls);
      break;
    }
  }
} else
  symtab.addDeclaration(decls);
}
        RCURLY
        |
        // unnamed structure with body
        // This is for one time use
        // Added "named_" to prevent illegal identifiers.
        l1:LCURLY
{
name = "named_"+getLexer().originalSource +"_"+ ((CToken)l1).getTokenNumber();
name = name.replaceAll("[.]","_");
name = name.replaceAll("-","_");
linenum = l1.getLine(); putPragma(l1,symtab);
if (type == 1) {
  decls = new ClassDeclaration(ClassDeclaration.STRUCT, new NameID(name));
  spec = new UserSpecifier(new NameID("struct "+name));
} else {
  decls = new ClassDeclaration(ClassDeclaration.UNION, new NameID(name));
  spec = new UserSpecifier(new NameID("union "+name));
}
}
        ( structDeclarationList[decls] )?
{
if (symtab instanceof ClassDeclaration) {
  int si = symtabstack.size()-1;
  for (;si>=0;si--) {
    if (!(symtabstack.get(si) instanceof ClassDeclaration)) {
      ((SymbolTable)symtabstack.get(si)).addDeclaration(decls);
      break;
    }
  }
} else
  symtab.addDeclaration(decls);
}
        RCURLY
        | // named structure without body
        sou3:ID
{
name = sou3.getText();linenum = sou3.getLine(); putPragma(sou3,symtab);
if(type == 1) {
  spec = new UserSpecifier(new NameID("struct "+name));
  decls = new ClassDeclaration(ClassDeclaration.STRUCT,new NameID(name),true);
} else {
  spec = new UserSpecifier(new NameID("union "+name));
  decls = new ClassDeclaration(ClassDeclaration.UNION,new NameID(name),true);
}
prev_decl = decls;
}
        )
        ;


/* Declarations are added to ClassDeclaration */
structDeclarationList [ClassDeclaration cdecl]
{Declaration sdecl= null;/*SymbolTable prev_symtab = symtab;*/}
        :
{enterSymtab(cdecl);}
        (
        sdecl=structDeclaration
{if(sdecl != null ) cdecl.addDeclaration(sdecl);}
        )+
{exitSymtab(); /*symtab = prev_symtab;*/}
        ;


/* A declaration */
structDeclaration returns [Declaration sdecl]
{
List bsqlist=null;
List bsdlist=null;
sdecl=null;
}
        :
        bsqlist = specifierQualifierList
        // passes specifier to put in symtab
        bsdlist = structDeclaratorList
        ( COMMA! )? ( SEMI! )+
{sdecl = new VariableDeclaration(bsqlist,bsdlist); hastypedef = false;}
        ;


/* List of Specifiers */
// [DEBUG by Seyong Lee] old definition is deprecated.
specifierQualifierList_old returns [List sqlist]
{
sqlist=new ArrayList();
Specifier tspec=null;
Specifier tqual=null;
}
        :
        (
        // this loop properly aborts when it finds a non-typedefName ID MBZ
        options {warnWhenFollowAmbig = false;}
        :
        /* A type : BaseType */
        ( "struct" | "union" | "enum" | typeSpecifier ) =>
        tspec = typeSpecifier
{sqlist.add(tspec);}
        |
        /* A Modifier : int value */
        tqual=typeQualifier
{sqlist.add(tqual);}
        )+
        ;
        
/* List of Specifiers */
// [DEBUG by Seyong Lee] new definition of specifierQualifierList
specifierQualifierList returns [List sqlist]
{
sqlist=new ArrayList();
Specifier tspec=null;
Specifier tqual=null;
}
        :
        (
        // this loop properly aborts when it finds a non-typedefName ID MBZ
        options {warnWhenFollowAmbig = false;}
        :
        (
        /* A Modifier : int value */
        tqual=typeQualifier
{sqlist.add(tqual);}
		)*
		
		// There should exist at least one type specifier.
        /* A type : BaseType */
        ( "struct" | "union" | "enum" | typeSpecifier ) =>
        tspec = typeSpecifier
{sqlist.add(tspec);}
		
		(
		// List of non-typeDefName specifier
        /* A type : BaseType */
        ( "struct" | "union" | "enum" | typeSpecifier_noTypeDefName ) =>
        tspec = typeSpecifier_noTypeDefName
{sqlist.add(tspec);}
        |
        /* A Modifier : int value */
        tqual=typeQualifier
{sqlist.add(tqual);}
		)*
        )
        ;


/* List of Declarators */
structDeclaratorList returns [List sdlist]
{
sdlist = new ArrayList();
Declarator sdecl=null;
}
        :
        sdecl = structDeclarator
{
// why am I getting a null value here ?
if (sdecl != null)
  sdlist.add(sdecl);
}
        (
        options{warnWhenFollowAmbig=false;}
        :
        COMMA! sdecl=structDeclarator
{
// [DEBUG by Seyong Lee] may be null if used in a function parameter declaration
if( sdecl != null )
  sdlist.add(sdecl);
}
        )*
        ;


/* Declarator */
structDeclarator returns [Declarator sdecl]
{
// [Modified by Joel E. Denny to handle unnamed bit-fields.]
sdecl=new VariableDeclarator(new NameID(""));
Expression expr1=null;
}
        :
        ( sdecl = declarator )?
        //( COLON expr1=constExpr )?
        /* bit-field recognition */
        ( COLON expr1=assignExpr )?
{
if (sdecl != null && expr1 != null) {
  // [Modified by Joel E. Denny not to throw away expressions that Cetus
  // cannot simplify to an IntegerLiteral but which are constant expressions.
  // For example, "sizeof(int)".]
  expr1 = Symbolic.simplify(expr1);
  //if (expr1 instanceof IntegerLiteral)
    sdecl.addTrailingSpecifier(new BitfieldSpecifier(expr1));
  //else
  //  ; // need to throw parse error
}
}
/* This needs to be fixed */
//{sdecl.addExpr(expr1);}
// Ignore this GCC dialect
/*
{if(sdecl == null && expr1 == null){
System.err.println("Errorororororo");
}
}*/
        ( attributeDecl )*
        ;


/* UserSpecifier (Enumuration) */
enumSpecifier returns[Specifier spec]
{
cetus.hir.Enumeration espec = null;
String enumN = null;
List elist=null;
spec = null;
}
        :
        "enum"^
        // [Attributes added by Joel E. Denny]
        (attributeDecl)?
        (
        ( ID LCURLY ) => i:ID
        {putPragma(i,symtab);}
        LCURLY elist=enumList RCURLY
{enumN =i.getText();}
        |
        el1:LCURLY {putPragma(el1,symtab);} elist=enumList RCURLY
{
enumN = getLexer().originalSource +"_"+ ((CToken)el1).getTokenNumber();
enumN =enumN.replaceAll("[.]","_");
enumN =enumN.replaceAll("-","_");
}
        |
        espec2:ID
{enumN =espec2.getText();putPragma(espec2,symtab);}
        )
        // has name and list of members
{
if (elist != null) {
  espec = new cetus.hir.Enumeration(new NameID(enumN),elist);
  if (symtab instanceof ClassDeclaration) {
    int si = symtabstack.size()-1;
    for (;si>=0;si--) {
      if (!(symtabstack.get(si) instanceof ClassDeclaration)) {
        ((SymbolTable)symtabstack.get(si)).addDeclaration(espec);
        break;
      }
    }
  } else
    symtab.addDeclaration(espec);
}
spec = new UserSpecifier(new NameID("enum "+enumN));
}
        ;


/* List of Declarator */
enumList returns [List elist]
{
Declarator enum1=null;
elist = new ArrayList();
}
        :
        enum1=enumerator
{elist.add(enum1);}
        (
        options{warnWhenFollowAmbig=false;}
        :
        COMMA! enum1=enumerator {elist.add(enum1);}
        )*
        ( COMMA! )?
        ;


/* Declarator */

// Complicated due to setting values for each enum value
enumerator returns[Declarator decl]
{decl=null;Expression expr2=null; String val = null;}
        :
        /* Variable Declarator */
        i:ID
{
val = i.getText();
decl = new VariableDeclarator(new NameID(val));
}
        // [Attributes added by Joel E. Denny]
        (attributeDecl)?
        /* Initializer */
        (
        ASSIGN expr2=constExpr
{decl.setInitializer(new Initializer(expr2));}
        )?
        ;

// Non standard VC++ specific attributes are matched and discarded
vcDeclSpec:
    "__declspec" LPAREN (extendedDeclModifier)* RPAREN
    ;

extendedDeclModifier:
    ID(
      LPAREN(
        Number
        |(StringLiteral)+
        |ID ASSIGN ID
      )RPAREN
    )?
    ;

// Non standard GCC specific attributes are matched and discarded
// [Extended by Joel E. Denny to return simple attributes]
attributeDecl returns [List<Attribute> list]
{
list = new ArrayList<>();
List<Attribute> subList;
}
        :
        "__attribute"^
        LPAREN LPAREN subList = attributeList RPAREN RPAREN
{list.addAll(subList);}
        (
          subList = attributeDecl
{list.addAll(subList);}
        )*
        |
        "__asm"^
        LPAREN stringConst RPAREN
        ;


// [Extended by Joel E. Denny to return simple attributes]
attributeList returns [List<Attribute> list]
{list = new ArrayList<>(); Attribute attr;}
        :
        attr = attribute
{if (attr != null) list.add(attr);}
        (
        options{warnWhenFollowAmbig=false;}
        :
        COMMA attr = attribute
{if (attr != null) list.add(attr);}
        )*
        //( COMMA )?
        ;


// [Extended by Joel E. Denny to return simple attributes]
attribute returns [Attribute spec]
{spec = null;}
        :
        (
        // Word
        (
            id:ID
{spec = new Attribute(id.getText());}
            //| declSpecifiers
            |
            storageClassSpecifier
            |
            typeQualifier
        )
        (
            LPAREN
            (
            ID
            //|
            //assignExpr
            |
            //epsilon
            )
            (
            //(COMMA assignExpr)*
            expr
            |
            //epsilon
            )
            RPAREN
        )?
        //~(LPAREN | RPAREN | COMMA)
        //|  LPAREN attributeList RPAREN
        )?
        ;


/* List of Declarator */
initDeclList returns [List dlist]
{
Declarator decl=null;
dlist = new ArrayList();
}
        :
        decl = initDecl
{dlist.add(decl);}
        (
        options{warnWhenFollowAmbig=false;}
        :
        COMMA!
        decl = initDecl
{dlist.add(decl);}
        )*
        ( COMMA! )?
        ;


/* Declarator */
initDecl returns [Declarator decl]
{
decl = null;
//Initializer binit=null;
Object binit = null;
Expression expr1=null;
}
        :
        // casting could cause a problem
        decl = declarator
        ( attributeDecl )* // Not Handled
        (
        ASSIGN binit=initializer
        |
        COLON expr1=expr // What is this guy ?
        )?
{
if (binit instanceof Expression)
  binit = new Initializer((Expression)binit);
if (binit != null) {
  decl.setInitializer((Initializer)binit);
/*
System.out.println("Initializer " + decl.getClass());
decl.print(System.out);
System.out.print(" ");
((Initializer)binit).print(System.out);
System.out.println("");
*/
}
}
        ;


// add a pointer to the type list
// [NVL support added by Joel E. Denny]
pointerGroup returns [List bp]
{
bp = new ArrayList();
Specifier temp = null;
boolean b_const = false;
boolean b_volatile = false;
boolean b_restrict = false;
boolean b_nvl = false;
boolean b_nvl_wp = false;
}
        :
        (
        STAR
        // add the modifer
        (
        temp = typeQualifier
{
if (temp == Specifier.CONST)
  b_const = true;
else if (temp == Specifier.VOLATILE)
  b_volatile = true;
else if (temp == Specifier.RESTRICT)
  b_restrict = true;
else if (temp == Specifier.NVL)
  b_nvl = true;
else if (temp == Specifier.NVL_WP)
  b_nvl_wp = true;
}
        )*
{
// 5 quals
if (b_const && b_restrict && b_volatile && b_nvl && b_nvl_wp)
  bp.add(PointerSpecifier.CONST_RESTRICT_VOLATILE_NVL_NVL_WP);

// 4 quals
else if (b_const && b_restrict && b_volatile && b_nvl)
  bp.add(PointerSpecifier.CONST_RESTRICT_VOLATILE_NVL);
else if (b_const && b_restrict && b_volatile && b_nvl_wp)
  bp.add(PointerSpecifier.CONST_RESTRICT_VOLATILE_NVL_WP);
else if (b_const && b_restrict && b_nvl && b_nvl_wp)
  bp.add(PointerSpecifier.CONST_RESTRICT_NVL_NVL_WP);
else if (b_const && b_volatile && b_nvl && b_nvl_wp)
  bp.add(PointerSpecifier.CONST_VOLATILE_NVL_NVL_WP);
else if (b_restrict && b_volatile && b_nvl && b_nvl_wp)
  bp.add(PointerSpecifier.RESTRICT_VOLATILE_NVL_NVL_WP);

// 3 quals
else if (b_const && b_restrict && b_volatile)
  bp.add(PointerSpecifier.CONST_RESTRICT_VOLATILE);
else if (b_const && b_restrict && b_nvl)
  bp.add(PointerSpecifier.CONST_RESTRICT_NVL);
else if (b_const && b_volatile && b_nvl)
  bp.add(PointerSpecifier.CONST_VOLATILE_NVL);
else if (b_restrict && b_volatile && b_nvl)
  bp.add(PointerSpecifier.RESTRICT_VOLATILE_NVL);
else if (b_const && b_restrict && b_nvl_wp)
  bp.add(PointerSpecifier.CONST_RESTRICT_NVL_WP);
else if (b_const && b_volatile && b_nvl_wp)
  bp.add(PointerSpecifier.CONST_VOLATILE_NVL_WP);
else if (b_restrict && b_volatile && b_nvl_wp)
  bp.add(PointerSpecifier.RESTRICT_VOLATILE_NVL_WP);
else if (b_const && b_nvl && b_nvl_wp)
  bp.add(PointerSpecifier.CONST_NVL_NVL_WP);
else if (b_restrict && b_nvl && b_nvl_wp)
  bp.add(PointerSpecifier.RESTRICT_NVL_NVL_WP);
else if (b_volatile && b_nvl && b_nvl_wp)
  bp.add(PointerSpecifier.VOLATILE_NVL_NVL_WP);

// 2 quals
else if (b_const && b_restrict)
  bp.add(PointerSpecifier.CONST_RESTRICT);
else if (b_const && b_volatile)
  bp.add(PointerSpecifier.CONST_VOLATILE);
else if (b_restrict && b_volatile)
  bp.add(PointerSpecifier.RESTRICT_VOLATILE);
else if (b_const && b_nvl)
  bp.add(PointerSpecifier.CONST_NVL);
else if (b_restrict && b_nvl)
  bp.add(PointerSpecifier.RESTRICT_NVL);
else if (b_volatile && b_nvl)
  bp.add(PointerSpecifier.VOLATILE_NVL);
else if (b_const && b_nvl_wp)
  bp.add(PointerSpecifier.CONST_NVL_WP);
else if (b_restrict && b_nvl_wp)
  bp.add(PointerSpecifier.RESTRICT_NVL_WP);
else if (b_volatile && b_nvl_wp)
  bp.add(PointerSpecifier.VOLATILE_NVL_WP);
else if (b_nvl && b_nvl_wp)
  bp.add(PointerSpecifier.NVL_NVL_WP);

// 1 qual
else if (b_const)
  bp.add(PointerSpecifier.CONST);
else if (b_restrict)
  bp.add(PointerSpecifier.RESTRICT);
else if (b_volatile)
  bp.add(PointerSpecifier.VOLATILE);
else if (b_nvl)
  bp.add(PointerSpecifier.NVL);
else if (b_nvl_wp)
  bp.add(PointerSpecifier.NVL_WP);

// 0 quals
else
  bp.add(PointerSpecifier.UNQUALIFIED);

b_const = false;
b_volatile = false;
b_restrict = false;
b_nvl = false;
b_nvl_wp = false;
}
        )+
        ;


// need to decide what to add
idList returns [List ilist]
{
int i = 1;
String name;
Specifier temp = null;
ilist = new ArrayList();
}
        :
        idl1:ID
{
name = idl1.getText();
ilist.add(
    new VariableDeclaration(new VariableDeclarator(new NameID(name))));
}
        (
        options{warnWhenFollowAmbig=false;}
        :
        COMMA
        idl2:ID
{
name = idl2.getText();
ilist.add(
    new VariableDeclaration(new VariableDeclarator(new NameID(name))));
}
        )*
        ;


initializer returns [Object binit]
{
binit = null;
Expression expr1 = null;
List ilist = null;
Initializer init = null;
}
        :
        (
            binit=assignExpr
{((Expression)binit).setParens(false);}
            |
            binit=initializerElementLabel
            |
            ilist=lcurlyInitializer
{binit = new Initializer(ilist);}
        )
        |
        ilist=lcurlyInitializer
{binit = new Initializer(ilist);}
        ;


// Designated initializers are handled here.
initializerElementLabel returns [Initializer ret]
{
Expression expr1 = null, expr2=null, expr3=null;
ret=null;
List list = null;
}
        :
        LBRACKET
        (
            (expr1=constExpr VARARGS) => expr1=rangeExpr
            |
            expr2=constExpr
        )
        RBRACKET ASSIGN
        (
            expr3=assignExpr
            |
            list=lcurlyInitializer
        )
{
Expression sub = (expr1 != null)? expr1: expr2;
if (sub != null) {
    Expression designator = new ArrayAccess(new NameID(""), sub);
    if (expr3 != null) {
        ret = new Initializer(designator, expr3);
        expr3.setParens(false);
    } else if (list != null) {
        ret = new Initializer(designator, list);
    }
}
}
        |
        ID COLON /* not a part of C standard */
        |
        DOT id:ID ASSIGN
        (
            expr3=assignExpr
            |
            list=lcurlyInitializer
        )
{
Expression designator = new AccessExpression(new NameID(""),
        AccessOperator.MEMBER_ACCESS, new NameID(id.getText()));
if (expr3 != null) {
    ret = new Initializer(designator, expr3);
    expr3.setParens(false);
} else if (list != null) {
    ret = new Initializer(designator, list);
}
}
        ;


// GCC allows empty initializer lists
lcurlyInitializer returns [List ilist]
{ilist = new ArrayList();}
        :
        LCURLY^
        (ilist=initializerList ( COMMA! )? )?
        RCURLY
        ;


initializerList returns [List ilist]
{Object init = null; ilist = new ArrayList();}
        :
        (
        init = initializer
{ilist.add(init);}
        )
        (
        options{warnWhenFollowAmbig=false;}
        :
        COMMA! init = initializer
{ilist.add(init);}
        )*
        ;


/* Declarator */
declarator returns [Declarator decl]
{
Expression expr1=null;
String declName = null;
decl = null;
Declarator tdecl = null;
IDExpression idex = null;
List plist = null;
List bp = null;
Specifier aspec = null;
boolean isArraySpec = false;
boolean isNested = false;
List llist = new ArrayList();
List tlist = null;
}
        :
        // Pass "Type" to add pointer Type
        ( bp=pointerGroup )?
{/* if(bp == null) bp = new LinkedList(); */}
        (attributeDecl)? // For cygwin support
        (
        id:ID
        // Add the name of the Var
{
putPragma(id,symtab);
declName = id.getText();
idex = new NameID(declName);
if(hastypedef) {
  typetable.put(declName,"typedef");
  //System.err.println("Add typedef declname: " + declName); 
}
}
        |
        /* Nested Declarator */
        LPAREN
        tdecl = declarator
        RPAREN
        )
        // Attribute Specifier List Possible
        (attributeDecl)?
        // I give up this part !!!
        (
        /* Parameter List */
        plist = declaratorParamaterList
        |
        /* ArraySpecifier */
        LBRACKET ( expr1=expr )? RBRACKET
{
isArraySpec = true;
llist.add(expr1);
}
        )*
{
/* Possible combinations []+, () */
if (plist != null) {
  /* () */
  ;
} else {
  /* []+ */
  if (isArraySpec) {
    aspec = new ArraySpecifier(llist);
    tlist = new ArrayList();
    tlist.add(aspec);
  }
}
if (bp == null)
  bp = new ArrayList();
if (tdecl != null) { // assume tlist == null
  //assert tlist == null : "Assertion (tlist == null) failed 2";
  if (tlist == null)
    tlist = new ArrayList();
    decl = new NestedDeclarator(bp,tdecl,plist,tlist);
} else {
  if (plist != null) // assume tlist == null
    decl = new ProcedureDeclarator(bp,idex,plist);
  else {
    if (tlist != null)
      decl = new VariableDeclarator(bp,idex,tlist);
    else
      decl = new VariableDeclarator(bp,idex);
  }
}
}
        ;


/* List */
declaratorParamaterList returns [List plist]
{plist = new ArrayList();}
        :
        LPAREN^
        (
        (declSpecifiers) => plist=parameterTypeList
        |
        (plist=idList)?
        )
        ( COMMA! )?
        RPAREN
        ;


/* List of (?) */
parameterTypeList returns [List ptlist]
{ptlist = new ArrayList(); Declaration pdecl = null;}
        :
        pdecl=parameterDeclaration
{ptlist.add(pdecl);}
        (
        options {warnWhenFollowAmbig = false;}
        :
        ( COMMA | SEMI )
        pdecl = parameterDeclaration
{ptlist.add(pdecl);}
        )*
        /* What about "..." ? */
        (
        ( COMMA | SEMI )
        VARARGS
{
ptlist.add(
    new VariableDeclaration(new VariableDeclarator(new NameID("..."))));
}
        )?
        ;


/* Declaration (?) */
parameterDeclaration returns [Declaration pdecl]
{
pdecl =null;
List dspec = null;
Declarator decl = null;
boolean prevhastypedef = hastypedef;
hastypedef = false;
}
        :
        dspec=declSpecifiers
        (
        ( declarator )=> decl = declarator
        |
        decl = nonemptyAbstractDeclarator
        )?
{
if (decl != null) {
  pdecl = new VariableDeclaration(dspec,decl);
if (isFuncDef) {
  currproc.add(pdecl);
}
} else
  pdecl = new VariableDeclaration(
      dspec,new VariableDeclarator(new NameID("")));
hastypedef = prevhastypedef;
}
        ;


/* JTC:
* This handles both new and old style functions.
* see declarator rule to see differences in parameters
* and here (declaration SEMI)* is the param type decls for the
* old style.  may want to do some checking to check for illegal
* combinations (but I assume all parsed code will be legal?)
*/

functionDef returns [Procedure curFunc]
{
CompoundStatement stmt=null;
Declaration decl=null;
Declarator bdecl=null;
List dspec=null;
curFunc = null;
String declName=null;
int dcount = 0;
SymbolTable prev_symtab =null;
SymbolTable temp_symtab = new CompoundStatement();
}
        :
        (
{isFuncDef = true;}
        (functionDeclSpecifiers) => dspec=functionDeclSpecifiers
        |
        //epsilon
        )
{if (dspec == null) dspec = new ArrayList();}
        bdecl = declarator
        /* This type of declaration is a problem */
{enterSymtab(temp_symtab); old_style_func = true; func_decl_list.clear();}
        ( declaration {dcount++;})* (VARARGS)? ( SEMI! )*
{
old_style_func = false;
exitSymtab();
isFuncDef = false;
if (dcount > 0) {
  HashMap hm = null;
  NameID name = null;
  Declaration tdecl = null;
/**
 *  This implementation is not so good since it relies on
 * the fact that function parameter starts from the second
 *  children and getChildren returns a reference to the
 * actual internal list
 */
List<Traversable> bdecl_children = bdecl.getChildren();
int bdecl_size = bdecl_children.size();
for (int i = 1; i < bdecl_size; i++) {
  VariableDeclaration vdec = (VariableDeclaration)bdecl_children.get(i);
  List decl_ids = vdec.getDeclaredIDs();
  int decl_ids_size = decl_ids.size();
  for (int j = 0; j < decl_ids_size; j++) {
    // declarator name
    name = (NameID)decl_ids.get(j);
    // find matching Declaration
    tdecl = (Declaration)(func_decl_list.get(name.toString()));
    if (tdecl == null) {
      PrintTools.printlnStatus("cannot find symbol " + name
          + "in old style function declaration, now assuming an int",1);
      tdecl = new VariableDeclaration(
          Specifier.INT, new VariableDeclarator(name.clone()));
    }
    bdecl_children.set(i, tdecl);
    tdecl.setParent(bdecl);
  }
}
Iterator diter = temp_symtab.getDeclarations().iterator();
Object tobject = null;
while (diter.hasNext()) {
  tobject = diter.next();
  if (tobject instanceof PreAnnotation)
    symtab.addDeclaration((Declaration)tobject);
}
}

}
        stmt=compoundStatement
{
// support for K&R style declaration: "dcount" is counting the number of
// declaration in old style.
// [Extended by Joel E. Denny to collect simple attributes]
List<Attribute> attrs = new ArrayList<>();
List dspec_new = new ArrayList();
for (Object obj : dspec) {
    if (obj instanceof Attribute)
        attrs.add((Attribute)obj);
    else
        dspec_new.add(obj);
}
curFunc = new Procedure(dspec_new, bdecl, stmt, dcount>0,
                        attrs.isEmpty() ? null : new AttributeSpecifier(attrs));
PrintTools.printStatus("Creating Procedure: ",1);
PrintTools.printlnStatus(bdecl,1);
// already handled in constructor
currproc.clear();
}
        ;


// [Extended by Joel E. Denny to collect simple attributes]
functionDeclSpecifiers returns [List dspec]
{
dspec = new ArrayList();
Specifier type=null;
Specifier tqual=null;
Specifier tspec=null;
List<Attribute> attrs;
}
        :
        (
        // this loop properly aborts when it finds a non-typedefName ID MBZ
        options {warnWhenFollowAmbig = false;}
        :
        type=functionStorageClassSpecifier
{dspec.add(type);}
        |
        tqual=typeQualifier
{dspec.add(tqual);}
        |
        ( "struct" | "union" | "enum" | tspec=typeSpecifier)=>
        tspec=typeSpecifier
{dspec.add(tspec);}
        |
        attrs=attributeDecl
{dspec.addAll(attrs);}
        |
        vcDeclSpec
        )+
        ;


// [Modified by Joel E. Denny to set line number on declaration.]
declarationList
{Declaration decl=null;List tlist = new ArrayList();}
        :
        (
        // this loop properly aborts when it finds a non-typedefName ID MBZ
        options {warnWhenFollowAmbig = false;}
        :
        localLabelDeclaration
        |
        ( declarationPredictor )=>
        decl=declaration
{if(decl != null ) curr_cstmt.addDeclaration(decl, LT(-1).getLine());}
        )+
        ;


declarationPredictor
{Declaration decl=null;}
        :
        (
        //only want to look at declaration if I don't see typedef
        options {warnWhenFollowAmbig = false;}
        :
        "typedef"
        |
        decl=declaration
        )
        ;


localLabelDeclaration
        :
        (
        // GNU note:  any __label__ declarations must come before regular
        // declarations.
        "__label__"^ ID
        (
        options{warnWhenFollowAmbig=false;}
        : COMMA! ID
        )*
        ( COMMA! )? ( SEMI! )+
        )
        ;


compoundStatement returns [CompoundStatement stmt]
{
stmt = null;
int linenum = 0;
SymbolTable prev_symtab = null;
CompoundStatement prev_cstmt = null;
}
        :
        lcur:LCURLY^
{
linenum = lcur.getLine();
prev_symtab = symtab;
prev_cstmt = curr_cstmt;
stmt = new CompoundStatement();
enterSymtab(stmt);
stmt.setLineNumber(linenum);
putPragma(lcur,prev_symtab);
curr_cstmt = stmt;
}
        (
        // this ambiguity is ok, declarationList and nestedFunctionDef end
        // properly
        options {warnWhenFollowAmbig = false;}
        :
        ( "typedef" | "__label__" | declaration ) => declarationList
        |
        (nestedFunctionDef) => nestedFunctionDef // not going to handle this
        )*
        ( statementList )?
        rcur:RCURLY
{
linenum = rcur.getLine();
putPragma(rcur,symtab);
curr_cstmt = prev_cstmt;
exitSymtab();
}

        ;
        exception
          catch [RecognitionException ex] {
            System.err.println("Cetus does not support C99 features yet");
            System.err.println("Please check if Declarations and Statements are interleaved");
            reportError(ex);
        }


// Not handled now
nestedFunctionDef
{Declarator decl=null;}
        :
        ( "auto" )? //only for nested functions
        ( (functionDeclSpecifiers)=> functionDeclSpecifiers )?
        // "null" could cause a problem
        decl = declarator
        ( declaration )*
        compoundStatement
        ;


// [Modified by Joel E. Denny to set line number on DeclarationStatement.]
statementList
{Statement statb = null; Declaration decl = null;}
        :
        
            (
            
            // User type must be examined first to avoid pointer variable
            // declarations' being parsed into multiplication expression.
            {isTypedefName (LT(1).getText())}?
            decl = declaration
//{curr_cstmt.addDeclaration(decl);}
{
DeclarationStatement declStat = new DeclarationStatement(decl);
declStat.setLineNumber(LT(-1).getLine());
curr_cstmt.addStatement(declStat);
}
            |
            statb = statement
{curr_cstmt.addStatement(statb);}
            
            |
            decl = declaration
{
DeclarationStatement declStat = new DeclarationStatement(decl);
declStat.setLineNumber(LT(-1).getLine());
curr_cstmt.addStatement(declStat);
}
//{curr_cstmt.addDeclaration(decl);}
// C99 extension that allows mixed statements and declarations; the parsed
// declarations are inserted into the current scope and printed as ANSI C
// declarations.
// Alternatively, the declaration can be placed at the original position, but
// it can be very difficult to make it consistent over a sequence of
// transformations.
            )+
/*
        exception
          catch [RecognitionException ex] {
            System.err.println("Cetus does not support C99 features yet");
            System.err.println("Please check if Declarations and Statements are interleaved");
            reportError(ex);
        }
*/
        ;


statement returns [Statement statb]
{
Expression stmtb_expr;
statb = null;
Expression expr1=null, expr2=null, expr3=null;
Statement stmt1=null,stmt2=null;
Declaration decl = null;
int a=0;
int sline = 0;
}
        :
        /* NullStatement */
        tsemi:SEMI
{
sline = tsemi.getLine();
statb = new NullStatement();
putPragma(tsemi,symtab);
}
        |
        /* CompoundStatement */
        statb=compoundStatement
        |
        // [Modified by Joel E. Denny to set line number.]
        /* ExpressionStatement */
        stmtb_expr=expr exprsemi:SEMI!
{
sline = exprsemi.getLine();
putPragma(exprsemi,symtab);
/* I really shouldn't do this test */
statb = new ExpressionStatement(stmtb_expr);
statb.setLineNumber(sline);
}
        /* Iteration statements */
        |
        /* WhileLoop */
        twhile:"while"^ LPAREN!
{
sline = twhile.getLine();
putPragma(twhile,symtab);
}
        expr1=expr RPAREN! stmt1=statement
{
statb = new WhileLoop(expr1, stmt1);
statb.setLineNumber(sline);
}
        |
        /* DoLoop */
        tdo:"do"^
{
sline = tdo.getLine();
putPragma(tdo,symtab);
}
        stmt1=statement "while"! LPAREN!
        expr1=expr RPAREN! SEMI!
{
statb = new DoLoop(stmt1, expr1);
statb.setLineNumber(sline);
}
        |
        /* ForLoop */
        !tfor:"for"
{
sline = tfor.getLine();
putPragma(tfor,symtab);
}
        LPAREN
        (
            // [Modified by Joel E. Denny: Copied semantic predicate from
            // statementList to handle for loop declaration starting with
            // typedef name.]
            {isTypedefName (LT(1).getText())}?
            decl=declaration // support of C99 block scope
            |
            (expr1=expr)? SEMI
            |
            // [Modified by Joel E. Denny: Made declaration non-optional
            // so that missing semicolon isn't permitted.]
            decl=declaration // support of C99 block scope
        )
        (expr2=expr)? SEMI
        (expr3=expr)?
        RPAREN
        stmt1=statement
{
if (expr1 != null) {
  statb = new ForLoop(new ExpressionStatement(expr1), expr2, expr3, stmt1);
} else {
  statb = new ForLoop(new NullStatement(), expr2, expr3, stmt1);
  if (decl != null) {
    // Constructs a legal ANSI C scope to handle C99's block scope
    CompoundStatement cstmt = new CompoundStatement();
    cstmt.addDeclaration(decl);
    cstmt.addStatement(statb);
    statb = cstmt;
  }
}
statb.setLineNumber(sline);
}
        /* Jump statements */
        |
        /* GotoStatement */
        tgoto:"goto"^
{
sline = tgoto.getLine();
putPragma(tgoto,symtab);
}
        gotoTarget:ID SEMI!
{
statb = new GotoStatement(new NameID(gotoTarget.getText()));
statb.setLineNumber(sline);
}
        |
        /* ContinueStatement */
        tcontinue:"continue" SEMI!
{
sline = tcontinue.getLine();
statb = new ContinueStatement();
statb.setLineNumber(sline);
putPragma(tcontinue,symtab);
}
        |
        /* BreakStatement */
        tbreak:"break" SEMI!
{
sline = tbreak.getLine();
statb = new BreakStatement();
statb.setLineNumber(sline);
putPragma(tbreak,symtab);
}
        |
        /* ReturnStatement */
        treturn:"return"^
{
sline = treturn.getLine();
}
        ( expr1=expr )? SEMI!
{
if (expr1 != null)
  statb=new ReturnStatement(expr1);
else
  statb=new ReturnStatement();
statb.setLineNumber(sline);
putPragma(treturn,symtab);
}
        |
        /* Label */
        lid:ID COLON!
{
sline = lid.getLine();
Object o = null;
Declaration target = null;
statb = new Label(new NameID(lid.getText()));
statb.setLineNumber(sline);
putPragma(lid,symtab);
}
        // Attribute Specifier List Possible
        (attributeDecl)?
        (
        options {warnWhenFollowAmbig=false;}
        :
        stmt1=statement
{
CompoundStatement cstmt = new CompoundStatement();
cstmt.addStatement(statb);
statb = cstmt;
cstmt.addStatement(stmt1);
}
        )?
        // GNU allows range expressions in case statements
        |
        /* Case */
        tcase:"case"^
{
sline = tcase.getLine();
}
        (
        (constExpr VARARGS)=> expr1=rangeExpr
        |
        expr1=constExpr
        )
{
statb = new Case(expr1);
statb.setLineNumber(sline);
putPragma(tcase,symtab);
}
        COLON!
        (
        options{warnWhenFollowAmbig=false;}
        :
        stmt1=statement
{
CompoundStatement cstmt = new CompoundStatement();
cstmt.addStatement(statb);
statb = cstmt;
cstmt.addStatement(stmt1);
}
        )?
        |
        /* Default */
        tdefault:"default"^
{
sline = tdefault.getLine();
statb = new Default();
statb.setLineNumber(sline);
putPragma(tdefault,symtab);
}
        COLON!
        (
        options{warnWhenFollowAmbig=false;}
        :
        stmt1=statement
{
CompoundStatement cstmt = new CompoundStatement();
cstmt.addStatement(statb);
statb = cstmt;
cstmt.addStatement(stmt1);
}
        )?
        /* Selection statements */
        |
        /* IfStatement  */
        tif:"if"^
{
sline = tif.getLine();
putPragma(tif,symtab);
}
        LPAREN! expr1=expr RPAREN! stmt1=statement
        //standard if-else ambiguity
        (
        options {warnWhenFollowAmbig = false;}
        :
        "else" stmt2=statement
        )?
{
if (stmt2 != null)
  statb = new IfStatement(expr1,stmt1,stmt2);
else
  statb = new IfStatement(expr1,stmt1);
statb.setLineNumber(sline);
}
        |
        /* SwitchStatement */
        tswitch:"switch"^ LPAREN!
{
sline = tswitch.getLine();
}
        expr1=expr RPAREN!
{
statb = new SwitchStatement(expr1);
statb.setLineNumber(sline);
putPragma(tswitch,symtab);
}
        stmt1=statement
{
((SwitchStatement)statb).setBody((CompoundStatement)stmt1);
unwrapSwitch((SwitchStatement)statb);
}
        ;


/* Expression */
expr returns [Expression ret_expr]
{
ret_expr = null;
Expression expr1=null,expr2=null;
List elist = new ArrayList();
}
        :
        ret_expr=assignExpr
{elist.add(ret_expr);}
        (
        options {warnWhenFollowAmbig = false;}
        :
        /* MBZ:
        COMMA is ambiguous between comma expressions and
        argument lists.  argExprList should get priority,
        and it does by being deeper in the expr rule tree
        and using (COMMA assignExpr)*
        */
        /* CommaExpression is not handled now */
        c:COMMA^
        expr1=assignExpr
{elist.add(expr1);}
        )*
{
if (elist.size() > 1) {
  ret_expr = new CommaExpression(elist);
}
}
        ;


assignExpr returns [Expression ret_expr]
{ret_expr = null; Expression expr1=null; AssignmentOperator code=null;}
        :
        ret_expr=conditionalExpr
        (
        code = assignOperator!
        expr1=assignExpr
{ret_expr = new AssignmentExpression(ret_expr,code,expr1); }
        )?
        ;


assignOperator returns [AssignmentOperator code]
{code = null;}
        :
        ASSIGN
{code = AssignmentOperator.NORMAL;}
        |
        DIV_ASSIGN
{code = AssignmentOperator.DIVIDE;}
        |
        PLUS_ASSIGN
{code = AssignmentOperator.ADD;}
        |
        MINUS_ASSIGN
{code = AssignmentOperator.SUBTRACT;}
        |
        STAR_ASSIGN
{code = AssignmentOperator.MULTIPLY;}
        |
        MOD_ASSIGN
{code = AssignmentOperator.MODULUS;}
        |
        RSHIFT_ASSIGN
{code = AssignmentOperator.SHIFT_RIGHT;}
        |
        LSHIFT_ASSIGN
{code = AssignmentOperator.SHIFT_LEFT;}
        |
        BAND_ASSIGN
{code = AssignmentOperator.BITWISE_AND;}
        |
        BOR_ASSIGN
{code = AssignmentOperator.BITWISE_INCLUSIVE_OR;}
        |
        BXOR_ASSIGN
{code = AssignmentOperator.BITWISE_EXCLUSIVE_OR;}
        ;


constExpr returns [Expression ret_expr]
{ret_expr = null;}
        :
        ret_expr=conditionalExpr
        ;


logicalOrExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=logicalAndExpr
        (
        LOR^ expr1=logicalAndExpr
{ret_expr = new BinaryExpression(ret_expr,BinaryOperator.LOGICAL_OR,expr1);}
        )*
        ;


logicalAndExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=inclusiveOrExpr
        (
        LAND^ expr1=inclusiveOrExpr
{ret_expr = new BinaryExpression(ret_expr,BinaryOperator.LOGICAL_AND,expr1);}
        )*
        ;


inclusiveOrExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=exclusiveOrExpr
        (
        BOR^ expr1=exclusiveOrExpr
{
ret_expr = new BinaryExpression
    (ret_expr,BinaryOperator.BITWISE_INCLUSIVE_OR,expr1);
}
        )*
        ;


exclusiveOrExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=bitAndExpr
        (
        BXOR^ expr1=bitAndExpr
{
ret_expr = new BinaryExpression
    (ret_expr,BinaryOperator.BITWISE_EXCLUSIVE_OR,expr1);
}
        )*
        ;


bitAndExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=equalityExpr
        (
        BAND^ expr1=equalityExpr
{ret_expr = new BinaryExpression(ret_expr,BinaryOperator.BITWISE_AND,expr1);}
        )*
        ;


equalityExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=relationalExpr
        (
        (
        EQUAL^
{code = BinaryOperator.COMPARE_EQ;}
        |
        NOT_EQUAL^
{code = BinaryOperator.COMPARE_NE;}
        )
        expr1=relationalExpr
{ret_expr = new BinaryExpression(ret_expr,code,expr1);}
        )*
        ;


relationalExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=shiftExpr
        (
        (
        LT^
{code = BinaryOperator.COMPARE_LT;}
        |
        LTE^
{code = BinaryOperator.COMPARE_LE;}
        |
        GT^
{code = BinaryOperator.COMPARE_GT;}
        |
        GTE^
{code = BinaryOperator.COMPARE_GE;}
        )
        expr1=shiftExpr
{ret_expr = new BinaryExpression(ret_expr,code,expr1);}
        )*
        ;


shiftExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=additiveExpr
        (
        (
        LSHIFT^
{code = BinaryOperator.SHIFT_LEFT;}
        |
        RSHIFT^
{code = BinaryOperator.SHIFT_RIGHT;}
        )
        expr1=additiveExpr
{ret_expr = new BinaryExpression(ret_expr,code,expr1);}
        )*
        ;


additiveExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=multExpr
        (
        (
        PLUS^
{code = BinaryOperator.ADD;}
        |
        MINUS^
{code=BinaryOperator.SUBTRACT;}
        )
        expr1=multExpr
{ret_expr = new BinaryExpression(ret_expr,code,expr1);}
        )*
        ;


multExpr returns [Expression ret_expr]
{
Expression expr1, expr2; ret_expr=null;
BinaryOperator code = null;
}
        :
        ret_expr=castExpr
        (
        (
        STAR^
{code = BinaryOperator.MULTIPLY;}
        |
        DIV^
{code=BinaryOperator.DIVIDE;}
        |
        MOD^
{code=BinaryOperator.MODULUS;}
        )
        expr1=castExpr
{ret_expr = new BinaryExpression(ret_expr,code,expr1);}
        )*
        ;


typeName returns [List tname]
{
tname=null;
Declarator decl = null;
}
        :
        tname = specifierQualifierList
        /* Need to add this part */
        (decl = nonemptyAbstractDeclarator {tname.add(decl);})?
        ;
        
//[DEBUG] Added by Seyong Lee; below is used for typecast and sizeof, which 
//do not have to have declarator without name.
typeName2 returns [List tname]
{
tname=null;
Declarator decl = null;
}
        :
        tname = specifierQualifierList
        /* Need to add this part */
        (decl = nonemptyAbstractDeclarator2[tname, true])?
        ;


postfixExpr returns [Expression ret_expr]
{
ret_expr=null;
Expression expr1=null;
Expression expr2=null;
List tname = null;
}
        :
        expr1=primaryExpr
{ret_expr = expr1;}
        ( ret_expr=postfixSuffix[expr1] )?
        // [NVL support added by Joel E. Denny]
        | "__builtin_nvl_get_root"
        (
        (
        LPAREN
        ( expr1 = assignExpr )
        COMMA
        ( tname = typeName2 )
        RPAREN
        )
{ret_expr = new NVLGetRootExpression(expr1, tname);}
        )
        ( ret_expr=postfixSuffix[ret_expr] )?
        | "__builtin_nvl_alloc_nv"
        (
        (
        LPAREN
        ( expr1 = assignExpr )
        COMMA
        ( expr2 = assignExpr )
        COMMA
        ( tname = typeName2 )
        RPAREN
        )
{ret_expr = new NVLAllocNVExpression(expr1, expr2, tname);}
        )
        ( ret_expr=postfixSuffix[ret_expr] )?
        ;


postfixSuffix [Expression expr1] returns [Expression ret_expr]
{
Expression expr2=null;
SymbolTable saveSymtab = null;
String s;
ret_expr = expr1;
List args = null;
}
        :
        (
        /* POINTER_ACCESS */
        PTR ptr_id:ID
{
ret_expr = new AccessExpression(
    ret_expr, AccessOperator.POINTER_ACCESS, SymbolTools.getOrphanID(ptr_id.getText()));
}
        |
        /* MEMBER_ACCESS */
        DOT dot_id:ID
{
ret_expr = new AccessExpression(
    ret_expr, AccessOperator.MEMBER_ACCESS, SymbolTools.getOrphanID(dot_id.getText()));
}
        /* FunctionCall */
        |
        args=functionCall
{
if (args == null)
  ret_expr = new FunctionCall(ret_expr);
else
  ret_expr = new FunctionCall(ret_expr,args);
}
        /* ArrayAcess - Need a fix for multi-dimension access */
        |
        LBRACKET expr2=expr RBRACKET
{
if (ret_expr instanceof ArrayAccess) {
  ArrayAccess aacc = (ArrayAccess)ret_expr;
  int dim = aacc.getNumIndices();
  int n = 0;
  List alist = new ArrayList();
  for (n = 0;n < dim; n++) {
    alist.add(aacc.getIndex(n).clone());
  }
  alist.add(expr2);
  aacc.setIndices(alist);
} else
  ret_expr = new ArrayAccess(ret_expr,expr2);
}
        |
        INC
{ret_expr = new UnaryExpression(UnaryOperator.POST_INCREMENT,ret_expr);}
        |
        DEC
{ret_expr = new UnaryExpression(UnaryOperator.POST_DECREMENT,ret_expr);}
        )+
        ;


functionCall returns [List args]
{args=null;}
        :
        LPAREN^  (args=argExprList)? RPAREN
        ;


conditionalExpr returns [Expression ret_expr]
{ret_expr=null; Expression expr1=null,expr2=null,expr3=null;}
        :
        expr1=logicalOrExpr
{ret_expr = expr1;}
        (
        QUESTION^ (expr2=expr)? COLON expr3=conditionalExpr
{ret_expr = new ConditionalExpression(expr1,expr2,expr3);}
        )?
        ;


//used in initializers only
rangeExpr returns [Expression ret_expr]
{ret_expr = null;}
        :
        constExpr VARARGS constExpr
        ;


//[DEBUG] Modified by Seyong Lee; typeName is changed to typeName2.
castExpr returns [Expression ret_expr]
{
ret_expr = null;
Expression expr1 = null;
List tname = null;
List init = null;
}
        :
        ( LPAREN typeName2 RPAREN ) => LPAREN^ tname=typeName2 RPAREN
        (
            expr1=castExpr
{ret_expr = new Typecast(tname,expr1);}
            |
            init=lcurlyInitializer
{ret_expr = new CompoundLiteral(tname, init);}
        )
        |
        ret_expr=unaryExpr
        ;


/* This causing problems with type casting */
nonemptyAbstractDeclarator returns [Declarator adecl]
{
Expression expr1=null;
List plist=null;
List bp = null;
Declarator tdecl = null;
Specifier aspec = null;
boolean isArraySpec = false;
boolean isNested = false;
List llist = new ArrayList();
List tlist = null;
boolean empty = true;
adecl = null;
}
        :
        (
        bp = pointerGroup
        (
        (
        LPAREN
        (
        (
        (
        tdecl = nonemptyAbstractDeclarator
        )
        |
        // function proto
        plist=parameterTypeList
        )
{empty = false;}
        )?
        ( COMMA! )?
        RPAREN
        )
{
if(empty)
plist = new ArrayList();
empty = true;
}
        |
        (LBRACKET (expr1=expr)? RBRACKET)
{
isArraySpec = true;
llist.add(expr1);
}
        )*
        |
        (
        (
        LPAREN
        (
        (
        (
        tdecl = nonemptyAbstractDeclarator
        )
        |
        // function proto
        plist=parameterTypeList
        )
{empty = false;}
        )?
        ( COMMA! )?
        RPAREN
        )
{
if (empty)
  plist = new ArrayList();
empty = true;
}
        |
        (LBRACKET (expr1=expr)? RBRACKET)
{
isArraySpec = true;
llist.add(expr1);
}
        )+
        )
{
if (isArraySpec) {
  /* []+ */
  aspec = new ArraySpecifier(llist);
  tlist = new ArrayList();
  tlist.add(aspec);
}
NameID idex = null;
// nested declarator (tlist == null ?)
if (bp == null)
  bp = new ArrayList();
// assume tlist == null
if (tdecl != null) {
  //assert tlist == null : "Assertion (tlist == null) failed 2";
  if (tlist == null)
    tlist = new ArrayList();
  adecl = new NestedDeclarator(bp,tdecl,plist,tlist);
} else {
  idex = new NameID("");
  if (plist != null) // assume tlist == null
    adecl = new ProcedureDeclarator(bp,idex,plist);
  else {
    if (tlist != null)
      adecl = new VariableDeclarator(bp,idex,tlist);
    else
      adecl = new VariableDeclarator(bp,idex);
  }
}
}
        ;

//[DEBUG] Added by Seyong Lee; used for typecast and sizeof expression.
nonemptyAbstractDeclarator2 [List tname, boolean isRoot] returns [Declarator adecl]
{
Expression expr1=null;
List plist=null;
List bp = null;
Declarator tdecl = null;
Specifier aspec = null;
boolean isArraySpec = false;
boolean isNested = false;
List llist = new ArrayList();
List tlist = null;
boolean empty = true;
boolean parenExist = false;
adecl = null;
}
        :
        (
        bp = pointerGroup
        (
        (
        LPAREN
        (
        (
        (
        tdecl = nonemptyAbstractDeclarator2[tname, false]
        )
        |
        // function proto
        plist=parameterTypeList
        )
{empty = false;}
        )?
        ( COMMA! )?
        RPAREN
        )
{
if(empty)
plist = new ArrayList();
empty = true;
parenExist = true;
}
        |
        (LBRACKET (expr1=expr)? RBRACKET)
{
isArraySpec = true;
llist.add(expr1);
}
        )*
        |
        (
        (
        LPAREN
        (
        (
        (
        tdecl = nonemptyAbstractDeclarator2[tname, false]
        )
        |
        // function proto
        plist=parameterTypeList
        )
{empty = false;}
        )?
        ( COMMA! )?
        RPAREN
        )
{
if (empty)
  plist = new ArrayList();
empty = true;
parenExist = true;
}
        |
        (LBRACKET (expr1=expr)? RBRACKET)
{
isArraySpec = true;
llist.add(expr1);
}
        )+
        )
{
if (isArraySpec) {
  /* []+ */
  aspec = new ArraySpecifier(llist);
  tlist = new ArrayList();
  tlist.add(aspec);
}
NameID idex = null;
// nested declarator (tlist == null ?)
if (bp == null)
  bp = new ArrayList();
// assume tlist == null
if (tdecl != null) {
  //assert tlist == null : "Assertion (tlist == null) failed 2";
  if (tlist == null)
    tlist = new ArrayList();
  adecl = new NestedDeclarator(bp,tdecl,plist,tlist);
} else if( !isRoot || (plist != null) || (tlist != null) ) {
	//Create empty declarator only for expressions inside a parenthesis.
  idex = new NameID("");
  if (plist != null) // assume tlist == null
    adecl = new ProcedureDeclarator(bp,idex,plist);
  else {
    if (tlist != null)
      adecl = new VariableDeclarator(bp,idex,tlist);
    else
      adecl = new VariableDeclarator(bp,idex);
  }
}
if( isRoot ) {
	if (adecl != null) {
		tname.add(adecl);
	} else if( !bp.isEmpty() ) {
		tname.addAll(bp);
	}
}
}
        ;

//[DEBUG] Modified by Seyong Lee; typeName is changed to typeName2.
unaryExpr returns [Expression ret_expr]
{
Expression expr1=null;
UnaryOperator code;
ret_expr = null;
List tname = null;
}
        :
        ret_expr=postfixExpr
        |
        INC^ expr1=castExpr
{ret_expr = new UnaryExpression(UnaryOperator.PRE_INCREMENT, expr1);}
        |
        DEC^ expr1=castExpr
{ret_expr = new UnaryExpression(UnaryOperator.PRE_DECREMENT, expr1);}
        |
        code=unaryOperator expr1=castExpr
{ret_expr = new UnaryExpression(code, expr1);}
        /* sizeof is not handled */
        |
        "sizeof"^
        (
        (LPAREN typeName2 ) => LPAREN tname=typeName2 RPAREN
{ret_expr = new SizeofExpression(tname);}
        |
        //[DEBUG] added by Seyong Lee: sizeof expr can have optional parentheses.
        //Later, commented out this rule because it breaks the following program:
        //
        //  #include <stddef.h>
        //  void fn() {
        //    size_t s;
        //    s = sizeof (3.2 *4);
        //  }
        //
        //Cetus reports:
        //
        //  Cetus Parsing Error: Exception Type: antlr.MismatchedTokenException
        //  Source: test.c Line:4 Column: 19 token name:STAR
        //  line 4:19: expecting RPAREN, found '*'
        //
        //As far as we can tell, this rule isn't needed anymore. The rule after
        //it allows parentheses around an expression via the derivation:
        //unaryExpr -> postfixExpr -> primaryExpr -> ( expr )
//        (LPAREN unaryExpr ) => LPAREN expr1=unaryExpr RPAREN
//{ret_expr = new SizeofExpression(expr1);}
//		|
        expr1=unaryExpr
{ret_expr = new SizeofExpression(expr1);}
        )
        |
        // Handles __alignof__ operator
        "__alignof__"^
        (
        ( LPAREN typeName2 ) => LPAREN tname=typeName2 RPAREN
{ret_expr = new AlignofExpression(tname);}
        |
        expr1=unaryExpr
{ret_expr = new AlignofExpression(expr1);}
        )
        |
        // Handles the builtin GCC function __builtin_va_arg
        // as an intrinsic function (operator)
        "__builtin_va_arg"
        (
        (
        LPAREN
        ( expr1 = unaryExpr )
        COMMA
        ( tname = typeName2 )
        RPAREN
        )
{ret_expr = new VaArgExpression(expr1, tname);}
        )
        |
        // Handles the builtin GCC function __builtin_offsetof
        // as an intrinsic function (operator)
        "__builtin_offsetof"
        (
        (
        LPAREN
        ( tname = typeName2 )
        COMMA
        ( expr1 = unaryExpr )
        RPAREN
        )
{ret_expr = new OffsetofExpression(tname, expr1);}
        )
        |
        (ret_expr=gnuAsmExpr)
        ;


unaryOperator returns [UnaryOperator code]
{code = null;}
        :
        BAND
{code = UnaryOperator.ADDRESS_OF;}
        |
        STAR
{code = UnaryOperator.DEREFERENCE;}
        |
        PLUS
{code = UnaryOperator.PLUS;}
        |
        MINUS
{code = UnaryOperator.MINUS;}
        |
        BNOT
{code = UnaryOperator.BITWISE_COMPLEMENT;}
        |
        LNOT
{code = UnaryOperator.LOGICAL_NEGATION;}
        |
        "__real"
{code = null;}
        |
        "__imag"
{code = null;}
        ;


gnuAsmExpr returns [Expression ret]
{
ret = null;
String str = "";
List<Traversable> expr_list = new ArrayList<Traversable>();
int count = 0;
}
        :
{count = mark();} // mark the previous token of __asm__
        "__asm"^ ("volatile")?
        LPAREN stringConst
        (
        options { warnWhenFollowAmbig = false; }
        :
        COLON (strOptExprPair[expr_list] ( COMMA strOptExprPair[expr_list])* )?
        (
        options { warnWhenFollowAmbig = false; }
        :
        COLON (strOptExprPair[expr_list] ( COMMA strOptExprPair[expr_list])* )?
        )?
        )?
        ( COLON stringConst ( COMMA stringConst)* )?
        RPAREN
{
// Recover the original stream and stores it in "SomeExpression" augmented with
// list of evaluated expressions.
for (int i=count-mark()+1; i <= 0; i++)
  str += " " + LT(i).getText();
ret = new SomeExpression(str, expr_list);
}
        ;


// GCC requires the PARENs
strOptExprPair [List<Traversable> expr_list]
{Expression e = null;}
        :
        stringConst
        (LPAREN (e=expr) RPAREN
{expr_list.add(e);}
        )?
        ;


primaryExpr returns [Expression p]
{
Expression expr1=null;
CompoundStatement cstmt = null;
p=null;
String name = null;
}
        :
        /* Identifier */
        prim_id:ID
{
name = prim_id.getText();
p=SymbolTools.getOrphanID(name);
}
        |
        /* Need to handle these correctly */
        prim_num:Number
{
// [Modified by Joel E. Denny to pass base and suffix to IntegerLiteral.]
name = prim_num.getText();
boolean handled = false;
name = name.toUpperCase();
String suffix = name.replaceAll("[X0-9A-E\\-+.]","");
name = name.replaceAll("L","");
name = name.replaceAll("U","");
IntegerLiteral.Base integerBase = IntegerLiteral.Base.DECIMAL;
if (name.startsWith("0X")) {
  integerBase = IntegerLiteral.Base.HEX;
  suffix = suffix.replaceAll("F", "");
}
else {
  if (name.startsWith("0") && !name.equals("0"))
    integerBase = IntegerLiteral.Base.OCTAL;
  name = name.replaceAll("F","");
  name = name.replaceAll("I","");
  // 1.0IF can be generated from _Complex_I
}
try {
  Integer i2 = Integer.decode(name);
  p=new IntegerLiteral(integerBase, i2.intValue(), suffix);
  handled = true;
} catch(NumberFormatException e) {
  ;
}
if (handled == false) {
  try {
    Long in = Long.decode(name);
    p=new IntegerLiteral(integerBase, in.longValue(), suffix);
    handled = true;
  } catch(NumberFormatException e) {
    ;
  }
}
if (handled == false) {
  try {
    double d = Double.parseDouble(name);
    if (suffix.matches("F|L|IF"))
      p = new FloatLiteral(d, suffix);
    else
      p = new FloatLiteral(d);
    handled = true;
  } catch(NumberFormatException e) {
    p=new NameID(name);
    PrintTools.printlnStatus("Strange number "+name,0);
  }
}
}
        |
        name=charConst
{
// [Modified by Joel E. Denny to handle wide characters.]
if(name.charAt(0) == 'L' && name.charAt(2) != '\\'
   || name.charAt(0) == '\'' && name.charAt(1) != '\\')
  p = new CharLiteral(name);
// escape sequence is not handled at this point
else {
  p = new EscapeLiteral(name);
}
}
        |
        /* StringLiteral */
        name=stringConst
{
p=new StringLiteral(name);
((StringLiteral)p).stripQuotes();
}
        // JTC:
        // ID should catch the enumerator
        // leaving it in gives ambiguous err
        //      | enumerator
        |
        /* Compound statement Expression */
        (LPAREN LCURLY) =>
        LPAREN^
        cstmt = compoundStatement
        RPAREN
{
PrintTools.printlnStatus("[DEBUG] Warning: CompoundStatement Expression !",1);
p = new StatementExpression(cstmt);
}
        |
        /* Paren */
        LPAREN^ expr1=expr RPAREN
{
p=expr1;
}
        ;


/* Type of list is unclear */
argExprList returns [List eList]
{
Expression expr1 = null;
eList=new ArrayList();
Declaration pdecl = null;
}
        :
        expr1=assignExpr
{eList.add(expr1);}
        (
        COMMA!
        (
        expr1=assignExpr
{eList.add(expr1);}
        |
        pdecl=parameterDeclaration
{eList.add(pdecl);}
        )
        )*
        ;


protected charConst returns [String name]
{name = null;}
        :
        cl:CharLiteral
{name=cl.getText();}
        ;


protected stringConst returns [String name]
{name = "";}
        :
        (
        sl:StringLiteral
{name += sl.getText();}
        )+
        ;


protected
intConst
        :       IntOctalConst
        |       LongOctalConst
        |       UnsignedOctalConst
        |       IntIntConst
        |       LongIntConst
        |       UnsignedIntConst
        |       IntHexConst
        |       LongHexConst
        |       UnsignedHexConst
        ;


protected
floatConst
        :       FloatDoubleConst
        |       DoubleDoubleConst
        |       LongDoubleConst
        ;


dummy
        :       NTypedefName
        |       NInitDecl
        |       NDeclarator
        |       NStructDeclarator
        |       NDeclaration
        |       NCast
        |       NPointerGroup
        |       NExpressionGroup
        |       NFunctionCallArgs
        |       NNonemptyAbstractDeclarator
        |       NInitializer
        |       NStatementExpr
        |       NEmptyExpression
        |       NParameterTypeList
        |       NFunctionDef
        |       NCompoundStatement
        |       NParameterDeclaration
        |       NCommaExpr
        |       NUnaryExpr
        |       NLabel
        |       NPostfixExpr
        |       NRangeExpr
        |       NStringSeq
        |       NInitializerElementLabel
        |       NLcurlyInitializer
        |       NAsmAttribute
        |       NGnuAsmExpr
        |       NTypeMissing
        ;


{
import java.io.*;
import antlr.*;
@SuppressWarnings({"unchecked", "cast"})
}

class NewCLexer extends Lexer;

options
{
k = 3;
exportVocab = NEWC;
testLiterals = false;
defaultErrorHandler=false;
}

tokens
{
LITERAL___extension__ = "__extension__";
}

{
public void initialize(String src)
{
  setOriginalSource(src);
  initialize();
}

public void initialize()
{
  literals.put(new ANTLRHashString("__alignof__", this),
      new Integer(LITERAL___alignof__));
  literals.put(new ANTLRHashString("__ALIGNOF__", this),
      new Integer(LITERAL___alignof__));
  literals.put(new ANTLRHashString("__asm", this),
      new Integer(LITERAL___asm));
  literals.put(new ANTLRHashString("__asm__", this),
      new Integer(LITERAL___asm));
  literals.put(new ANTLRHashString("__attribute__", this),
      new Integer(LITERAL___attribute));
  literals.put(new ANTLRHashString("__complex__", this),
      new Integer(LITERAL___complex));
  literals.put(new ANTLRHashString("__const", this),
      new Integer(LITERAL_const));
  literals.put(new ANTLRHashString("__const__", this),
      new Integer(LITERAL_const));
  literals.put(new ANTLRHashString("__imag__", this),
      new Integer(LITERAL___imag));
  literals.put(new ANTLRHashString("__inline", this),
      //new Integer(LITERAL___extension__));
      new Integer(LITERAL_inline));
  literals.put(new ANTLRHashString("__inline__", this),
      //new Integer(LITERAL___extension__));
      new Integer(LITERAL_inline));
  literals.put(new ANTLRHashString("__real__", this),
      new Integer(LITERAL___real));
  literals.put(new ANTLRHashString("__restrict", this),
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("__restrict__", this),
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("__extension", this),
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("__signed", this),
      new Integer(LITERAL_signed));
  literals.put(new ANTLRHashString("__signed__", this),
      new Integer(LITERAL_signed));
  /*
  literals.put(new ANTLRHashString("__typeof", this),
      new Integer(LITERAL_typeof));
  literals.put(new ANTLRHashString("__typeof__", this),
      new Integer(LITERAL_typeof));
  */
  literals.put(new ANTLRHashString("__volatile", this),
      new Integer(LITERAL_volatile));
  literals.put(new ANTLRHashString("__volatile__", this),
      new Integer(LITERAL_volatile));
  // GCC Builtin function
  literals.put(new ANTLRHashString("__builtin_va_arg", this),
      new Integer(LITERAL___builtin_va_arg));
  literals.put(new ANTLRHashString("__builtin_offsetof", this),
      new Integer(LITERAL___builtin_offsetof));
  // MinGW specific
  literals.put(new ANTLRHashString("__MINGW_IMPORT", this),
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("_CRTIMP", this),
      new Integer(LITERAL___extension__));
  // Microsoft specific
  literals.put(new ANTLRHashString("__cdecl", this),
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("__w64", this),
      new Integer(LITERAL___extension__));
  literals.put(new ANTLRHashString("__int64", this),
      new Integer(LITERAL_int));
  literals.put(new ANTLRHashString("__int32", this),
      new Integer(LITERAL_int));
  literals.put(new ANTLRHashString("__int16", this),
      new Integer(LITERAL_int));
  literals.put(new ANTLRHashString("__int8", this),
      new Integer(LITERAL_int));
  // [NVL support added by Joel E. Denny]
  literals.put(new ANTLRHashString("__builtin_nvl_get_root", this),
      new Integer(LITERAL___builtin_nvl_get_root));
  literals.put(new ANTLRHashString("__builtin_nvl_alloc_nv", this),
      new Integer(LITERAL___builtin_nvl_alloc_nv));
}

LineObject lineObject = new LineObject();
String originalSource = "";
PreprocessorInfoChannel preprocessorInfoChannel = new PreprocessorInfoChannel();
int tokenNumber = 0;
boolean countingTokens = true;
int deferredLineCount = 0;
NewCParser parser = null;

public void setCountingTokens(boolean ct)
{
  countingTokens = ct;
  if ( countingTokens ) {
    tokenNumber = 0;
  } else {
    tokenNumber = 1;
  }
}

public void setParser(NewCParser p)
{
  parser = p;
}

public void setOriginalSource(String src)
{
  originalSource = src;
  lineObject.setSource(src);
}

public void setSource(String src)
{
  lineObject.setSource(src);
}

public PreprocessorInfoChannel getPreprocessorInfoChannel()
{
  return preprocessorInfoChannel;
}

public void setPreprocessingDirective(String pre,int t)
{
  preprocessorInfoChannel.addLineForTokenNumber(
      new Pragma(pre,t), new Integer(tokenNumber));
}

protected Token makeToken(int t)
{
  if ( t != Token.SKIP && countingTokens) {
    tokenNumber++;
  }
  CToken tok = (CToken) super.makeToken(t);
  tok.setLine(lineObject.line);
  tok.setSource(lineObject.source);
  tok.setTokenNumber(tokenNumber);

  lineObject.line += deferredLineCount;
  deferredLineCount = 0;
  return tok;
}

public void deferredNewline()
{
  deferredLineCount++;
}

public void newline()
{
  lineObject.newline();
  setColumn(1);
}

}


protected
Vocabulary
        :       '\3'..'\377'
        ;

/* Operators: */
ASSIGN          : '=' ;
COLON           : ':' ;
COMMA           : ',' ;
QUESTION        : '?' ;
SEMI            : ';' ;
PTR             : "->" ;


// DOT & VARARGS are commented out since they are generated as part of
// the Number rule below due to some bizarre lexical ambiguity shme.

// DOT  :       '.' ;
protected
DOT:;

// VARARGS      : "..." ;
protected
VARARGS:;


LPAREN          : '(' ;
RPAREN          : ')' ;
LBRACKET        : '[' ;
RBRACKET        : ']' ;
LCURLY          : '{' ;
RCURLY          : '}' ;

EQUAL           : "==" ;
NOT_EQUAL       : "!=" ;
LTE             : "<=" ;
LT              : "<" ;
GTE             : ">=" ;
GT              : ">" ;

DIV             : '/' ;
DIV_ASSIGN      : "/=" ;
PLUS            : '+' ;
PLUS_ASSIGN     : "+=" ;
INC             : "++" ;
MINUS           : '-' ;
MINUS_ASSIGN    : "-=" ;
DEC             : "--" ;
STAR            : '*' ;
STAR_ASSIGN     : "*=" ;
MOD             : '%' ;
MOD_ASSIGN      : "%=" ;
RSHIFT          : ">>" ;
RSHIFT_ASSIGN   : ">>=" ;
LSHIFT          : "<<" ;
LSHIFT_ASSIGN   : "<<=" ;

LAND            : "&&" ;
LNOT            : '!' ;
LOR             : "||" ;

BAND            : '&' ;
BAND_ASSIGN     : "&=" ;
BNOT            : '~' ;
BOR             : '|' ;
BOR_ASSIGN      : "|=" ;
BXOR            : '^' ;
BXOR_ASSIGN     : "^=" ;


Whitespace
        :
        (
        ( ' ' | '\t' | '\014')
        | "\r\n" {newline();}
        | ( '\n' | '\r' ) {newline();}
        ) { _ttype = Token.SKIP;  }
        ;


Comment
        :
        (
        "/*"
        (
        { LA(2) != '/' }? '*'
        | "\r\n" { deferredNewline();}
        | ( '\r' | '\n' ) { deferredNewline();}
        | ~( '*'| '\r' | '\n' )
        )*
        "*/"
{setPreprocessingDirective(getText(),Pragma.comment);}
        )
{_ttype = Token.SKIP;}
        ;


CPPComment
        :
        (
        "//" ( ~('\n') )*
{setPreprocessingDirective(getText(),Pragma.comment);}
        )
{_ttype = Token.SKIP;}
        ;


PREPROC_DIRECTIVE
        options {paraphrase = "a line directive";}
        :
        '#'
        (
                // Line Directive - Skip token
                ( "line" || ((Space)+ Digit)) => LineDirective {_ttype = Token.SKIP; }
                |
                (
                // Pragma - Do not skip token
                        "pragma"
                        (
                                ( ~('\n'))*
                                {setPreprocessingDirective(getText(),Pragma.pragma);_ttype = Token.SKIP;}
                                /*
                                |
                                (START_INCLUDE)=>(Space)+ "startinclude" ( ~('\n'))*
                                // {startHeader(getText());}
                                |
                                (END_INCLUDE)=>(Space)+ "endinclude" ( ~('\n'))*
                                */
                                // {endHeader();}
                        )
                )
                |
                // Other control Sequence - Skip token
                ( ~('\n'))* {_ttype = Token.SKIP; }
        )
        ;
/*
protected START_INCLUDE
        :
        (Space)+ "startinclude"
        ;
protected END_INCLUDE
        :
        (Space)+ "endinclude"
        ;
*/

protected  Space
        :
        ( ' ' | '\t' | '\014')
        ;


protected LineDirective
{
boolean oldCountingTokens = countingTokens;
countingTokens = false;
}
        :
{
lineObject = new LineObject();
deferredLineCount = 0;
}
        ("line")?
        //this would be for if the directive started "#line",
        //but not there for GNU directives
        (Space)+
        n:Number
{lineObject.setLine(Integer.parseInt(n.getText())-1);}
        (
        (Space)+
        (
        fn:StringLiteral
{
try {
  lineObject.setSource(fn.getText().substring(1,fn.getText().length()-1));
} catch (StringIndexOutOfBoundsException e) { /*not possible*/
}
}
        |
        fi:ID
{lineObject.setSource(fi.getText());}
        )?
        (Space)*
        ("1" {lineObject.setEnteringFile(true);})?
        (Space)*
        ("2" {lineObject.setReturningToFile(true);})?
        (Space)*
        ("3" {lineObject.setSystemHeader(true);})?
        (Space)*
        ("4" {lineObject.setTreatAsC(true);})?
        (~('\r' | '\n'))*
        //("\r\n" | "\r" | "\n")
        )?
{
/*
preprocessorInfoChannel.addLineForTokenNumber(
    new LineObject(lineObject), new Integer(tokenNumber));
*/
countingTokens = oldCountingTokens;
}
        ;


/* Literals: */

/* Note that we do NOT handle tri-graphs nor multi-byte sequences. */

/*
 * Note that we can't have empty character constants (even though we
 * can have empty strings :-).
 */
CharLiteral
        :
        '\'' ( Escape | ~( '\'' ) ) '\''
        ;


protected BadStringLiteral
        :       // Imaginary token.
        ;


protected Escape
        :
        '\\'
        (
        options{warnWhenFollowAmbig=false;}
        :
        ~('0'..'7' | 'x')
        | ('0'..'3') ( options{warnWhenFollowAmbig=false;}: Digit )*
        | ('4'..'7') ( options{warnWhenFollowAmbig=false;}: Digit )*
        | 'x'
        (
        options{warnWhenFollowAmbig=false;}
        :
        Digit | 'a'..'f' | 'A'..'F'
        )+
        )
        ;


/* Numeric Constants: */
protected IntSuffix
        : 'L'
        | 'l'
        | 'U'
        | 'u'
        | 'I'
        | 'i'
        | 'J'
        | 'j'
        ;


protected NumberSuffix
        :
        IntSuffix
        | 'F'
        | 'f'
        ;


protected Digit
        :
        '0'..'9'
        ;


protected HexDigit
        :
        'a'..'f' | 'A'..'F' | '0'..'9'
        ;


protected HexFloatTail
        :
        ( 'P' | 'p' ) ( '+' | '-' )? ( Digit )+ ( 'f' | 'l' | 'F' | 'L' )?
        ;


protected Exponent
        :
        ( 'e' | 'E' ) ( '+' | '-' )? ( Digit )+
        ;


Number
        :
        ( ( Digit )+ ( '.' | 'e' | 'E' ) ) => ( Digit )+
        (
            '.' ( Digit )* ( Exponent )?
            |
            Exponent
        )
        ( NumberSuffix )*
        |
        ( "..." ) => "..."
{_ttype = VARARGS;}
        |
        '.'
{_ttype = DOT;}
        (
            ( Digit )+ ( Exponent )?
{_ttype = Number;}
            ( NumberSuffix )*
        )?
        |
        '0' ( '0'..'7' )* ( NumberSuffix )*
        |
        '1'..'9' ( Digit )* ( NumberSuffix )*
        |
        /* hexadecimal integer and floating point */
        '0' ( 'x' | 'X' )
        (
            ( HexDigit )+
            (
                ( IntSuffix )*
                |
                ( '.' ( HexDigit )* )? HexFloatTail
            )
            |
            '.' ( HexDigit )+ HexFloatTail
        )
        ;


IDMEAT
        :
        i:ID
{
if ( i.getType() == LITERAL___extension__ ) {
  $setType(Token.SKIP);
} else {
  $setType(i.getType());
}
}
        ;


protected ID
        options {testLiterals = true;}
        :
        ( 'a'..'z' | 'A'..'Z' | '_' | '$')
        ( 'a'..'z' | 'A'..'Z' | '_' | '$' | '0'..'9' )*
        ;


WideCharLiteral
        :
        'L' CharLiteral
{$setType(CharLiteral);}
        ;


WideStringLiteral
        :
        'L' StringLiteral
{$setType(StringLiteral);}
        ;


StringLiteral
        :
        '"'
        (
        ('\\' ~('\n')) => Escape
        |
        (
        '\r'
{newline();}
        |
        '\n'
{newline();}
        |
        '\\' '\n'
{newline();}
        )
        |
        ~( '"' | '\r' | '\n' | '\\' )
        )*
        '"'
        ;
