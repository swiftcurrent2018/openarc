package openacc.hir;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;
import cetus.hir.*;

/**
 * <b>TextureSpecifier</b> represents an Texture specifier in CUDA, for example 
 * texture<Type, cudaTextureType, ReadMode> texRef;
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future TechnologiesGroup, Oak Ridge National Laboratory
 *
 */
public class TextureSpecifier extends CUDASpecifier
{
	private List<Specifier> specs;
	private NameID cudaTextureType;
	private NameID readMode;

	public TextureSpecifier()
	{
		specs = new LinkedList<Specifier>();
	}

	public TextureSpecifier(Specifier type)
	{
		specs = new LinkedList<Specifier>();
		specs.add(type);
		cudaTextureType = new NameID("cudaTextureType1D");
		readMode = new NameID("cudaReadModeElementType");
	}
	
	public TextureSpecifier(List<Specifier> ispecs)
	{
		specs = new LinkedList<Specifier>();
		specs.addAll(ispecs);
		cudaTextureType = new NameID("cudaTextureType1D");
		readMode = new NameID("cudaReadModeElementType");
	}

	public TextureSpecifier(List<Specifier> ispecs, NameID textureType)
	{
		specs = new LinkedList<Specifier>();
		specs.addAll(ispecs);
		cudaTextureType = textureType;
		readMode = new NameID("cudaReadModeElementType");
	}

	public TextureSpecifier(List<Specifier> ispecs, NameID textureType, String ireadMode)
	{
		specs = new LinkedList<Specifier>();
		specs.addAll(ispecs);
		cudaTextureType = textureType;
		readMode = new NameID(ireadMode);
	}
	
	/** 
	 * Constructor for old CUDA driver API with version 3.2 or older.
	 * 
	 * @param ispecs
	 * @param idim
	 * @param ireadMode
	 */
	public TextureSpecifier(List<Specifier> ispecs, int idim, String ireadMode)
	{
		specs = new LinkedList<Specifier>();
		specs.addAll(ispecs);
		cudaTextureType = new NameID((new Integer(idim)).toString());
		readMode = new NameID(ireadMode);
	}

	/** Prints the specifier to the print writer. */
	public void print(PrintWriter o)
	{
		o.print("texture<");
		o.print(PrintTools.listToString(specs, " "));
		o.print(", " + cudaTextureType + ", " + readMode);
		o.print(">");
	}

	/** Returns a string representation of the specifier. */
	@Override
	public String toString()
	{
		StringWriter sw = new StringWriter(16);
		print(new PrintWriter(sw));
		return sw.toString();
	}

}
