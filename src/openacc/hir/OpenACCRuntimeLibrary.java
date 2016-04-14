package openacc.hir;

import java.util.*;
import cetus.hir.*;

/**
 * Repository for OpenACC runtime library functions. This class provides a basic
 * information about the OpenACC runtime library calls. Knowing that if a function call
 * may or must not have side effects can greatly improve the precision of a
 * program analysis in general. 
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future TechnologiesGroup, Oak Ridge National Laboratory
 *
 */
public class OpenACCRuntimeLibrary
{
  /** Only a single object is constructed. */
  private static final OpenACCRuntimeLibrary std = new OpenACCRuntimeLibrary();

  /** Predefined properties for each library functions */
  private Map<String, Set<Property>> catalog;

  /** Predefined set of properties */
  private enum Property
  {
    SIDE_EFFECT_GLOBAL,    // contains side effects on global variables.
    SIDE_EFFECT_PARAMETER, // contains side effects through parameters.
    OPENACC_VERSION1,      // OpenACC version 1.0 library
    OPENACC_VERSION2,      // OpenACC version 2.0 library
    OPENARC_EXTENSION,     // OpenARC library
    MEMORY_API,             // memory-related API
    SYNC_API,    // synchronization-related API
    CONFIG_API,      // configuration-related API
    KernelAPI,              // kernel-related API
    MEM_COPYIN,            // memory transfer from the host to the device
    MEM_COPYOUT,           // memory transfer from the device to the host
    DEVICE_MALLOC,         // allocate device memory            
    HOST_MALLOC,           // allocate host memory
    DEVICE_FREE,           // free device memory
    HOST_FREE,             // free host memory
    CHECK_PRESENT          // check present table.
  }

  /**
  * Checks if the given function call is a standard library call
  *  supported by OpenCL runtime system.
  * @param fcall the function call to be examined.
  * @return true if the function call exists in the entries.
  */
  public static boolean contains(FunctionCall fcall)
  {
    return (std.catalog.get(fcall.getName().toString()) != null);
  }
  
  public static boolean contains(String fcallName)
  {
    return (std.catalog.get(fcallName) != null);
  }

  /**
  * Checks if the given function call may have side effects.
  * @param fcall the function call to be examined.
  * @return true if the function call has a side effect.
  */
  public static boolean isSideEffectFree(FunctionCall fcall)
  {
    if ( !contains(fcall) )
      return false;
    Set<Property> properties = std.catalog.get(fcall.getName().toString());
    return (
      !properties.contains(Property.SIDE_EFFECT_GLOBAL) &&
      !properties.contains(Property.SIDE_EFFECT_PARAMETER)
    );
  }

  /**
  * Checks if the given function call is OpenACC version 1.0 API
  * @param fcall the function call to be examined.
  * @return true if the function call has a side effect.
  */
  public static boolean isOpenACCV1API(FunctionCall fcall)
  {
    if ( !contains(fcall) )
      return false;
    Set<Property> properties = std.catalog.get(fcall.getName().toString());
    return (
      properties.contains(Property.OPENACC_VERSION1) 
    );
  }

  /**
  * Checks if the given function call is OpenACC version 2.0 API
  * @param fcall the function call to be examined.
  * @return true if the function call has a side effect.
  */
  public static boolean isOpenACCV2API(FunctionCall fcall)
  {
    if ( !contains(fcall) )
      return false;
    Set<Property> properties = std.catalog.get(fcall.getName().toString());
    return (
      properties.contains(Property.OPENACC_VERSION2) 
    );
  }

  /**
  * Checks if the given function call is OpenARC API
  * @param fcall the function call to be examined.
  * @return true if the function call has a side effect.
  */
  public static boolean isOpenARCAPI(FunctionCall fcall)
  {
    if ( !contains(fcall) )
      return false;
    Set<Property> properties = std.catalog.get(fcall.getName().toString());
    return (
      properties.contains(Property.OPENARC_EXTENSION) 
    );
  }

  /**
  * Checks if the given function call is memory-related API.
  * @param fcall the function call to be examined.
  * @return true if the function call has a side effect.
  */
  public static boolean isMemoryAPI(FunctionCall fcall)
  {
    if ( !contains(fcall) )
      return false;
    Set<Property> properties = std.catalog.get(fcall.getName().toString());
    return (
      properties.contains(Property.MEMORY_API) 
    );
  }

  /**
  * Checks if the given function call allocates host memory 
  * @param fcall the function call to be examined.
  * @return true if the function call has a side effect.
  */
  public static boolean isHostMallocAPI(FunctionCall fcall)
  {
    if ( !contains(fcall) )
      return false;
    Set<Property> properties = std.catalog.get(fcall.getName().toString());
    return (
      properties.contains(Property.MEMORY_API) &&
      properties.contains(Property.HOST_MALLOC) 
    );
  }

  /**
  * Checks if the given function call allocates device memory 
  * @param fcall the function call to be examined.
  * @return true if the function call has a side effect.
  */
  public static boolean isDeviceMallocAPI(FunctionCall fcall)
  {
    if ( !contains(fcall) )
      return false;
    Set<Property> properties = std.catalog.get(fcall.getName().toString());
    return (
      properties.contains(Property.MEMORY_API) &&
      properties.contains(Property.DEVICE_MALLOC) 
    );
  }

  /**
  * Checks if the given function call frees host memory 
  * @param fcall the function call to be examined.
  * @return true if the function call has a side effect.
  */
  public static boolean isHostFreeAPI(FunctionCall fcall)
  {
    if ( !contains(fcall) )
      return false;
    Set<Property> properties = std.catalog.get(fcall.getName().toString());
    return (
      properties.contains(Property.MEMORY_API) &&
      properties.contains(Property.HOST_FREE) 
    );
  }

  /**
  * Checks if the given function call allocates device memory 
  * @param fcall the function call to be examined.
  * @return true if the function call has a side effect.
  */
  public static boolean isDeviceFreeAPI(FunctionCall fcall)
  {
    if ( !contains(fcall) )
      return false;
    Set<Property> properties = std.catalog.get(fcall.getName().toString());
    return (
      properties.contains(Property.MEMORY_API) &&
      properties.contains(Property.DEVICE_FREE) 
    );
  }

  /**
  * Checks if the given function call transfers from host to device.
  * @param fcall the function call to be examined.
  * @return true if the function call has a side effect.
  */
  public static boolean isCopyInAPI(FunctionCall fcall)
  {
    if ( !contains(fcall) )
      return false;
    Set<Property> properties = std.catalog.get(fcall.getName().toString());
    return (
      properties.contains(Property.MEMORY_API) &&
      properties.contains(Property.MEM_COPYIN) 
    );
  }

  /**
  * Checks if the given function call transfers from device to host.
  * @param fcall the function call to be examined.
  * @return true if the function call has a side effect.
  */
  public static boolean isCopyOutAPI(FunctionCall fcall)
  {
    if ( !contains(fcall) )
      return false;
    Set<Property> properties = std.catalog.get(fcall.getName().toString());
    return (
      properties.contains(Property.MEMORY_API) &&
      properties.contains(Property.MEM_COPYOUT) 
    );
  }

  /**
  * Checks if the given function call checks present table.
  * @param fcall the function call to be examined.
  * @return true if the function call has a side effect.
  */
  public static boolean isPresentCheckAPI(FunctionCall fcall)
  {
    if ( !contains(fcall) )
      return false;
    Set<Property> properties = std.catalog.get(fcall.getName().toString());
    return (
      properties.contains(Property.MEMORY_API) &&
      properties.contains(Property.CHECK_PRESENT) 
    );
  }

  /**
  * Checks if the given function call is synchronization-related API.
  * @param fcall the function call to be examined.
  * @return true if the function call has a side effect.
  */
  public static boolean isSyncAPI(FunctionCall fcall)
  {
    if ( !contains(fcall) )
      return false;
    Set<Property> properties = std.catalog.get(fcall.getName().toString());
    return (
      properties.contains(Property.SYNC_API) 
    );
  }

  /** Constructs a new repository */
  private OpenACCRuntimeLibrary()
  {
    catalog = new HashMap<String, Set<Property>>();
    addEntries();
  }

  /**
  * Adds each entry to the repository. 
  */
  private void addEntries()
  {
    // OpenACC version 1.0 library
    add("acc_get_num_devices",     Property.OPENACC_VERSION1, Property.CONFIG_API);
    add("acc_set_device_type",     Property.OPENACC_VERSION1, Property.SIDE_EFFECT_GLOBAL, Property.CONFIG_API);
    add("acc_get_device_type",     Property.OPENACC_VERSION1, Property.CONFIG_API);
    add("acc_set_device_num",     Property.OPENACC_VERSION1, Property.SIDE_EFFECT_GLOBAL, Property.CONFIG_API);
    add("acc_get_device_num",     Property.OPENACC_VERSION1, Property.CONFIG_API);
    add("acc_init",     Property.OPENACC_VERSION1, Property.SIDE_EFFECT_GLOBAL, Property.CONFIG_API);
    add("acc_shutdown",     Property.OPENACC_VERSION1, Property.SIDE_EFFECT_GLOBAL, Property.CONFIG_API);
    add("acc_on_device",     Property.OPENACC_VERSION1, Property.CONFIG_API);
    add("acc_async_test",     Property.OPENACC_VERSION1, Property.SYNC_API);
    add("acc_async_test_all",     Property.OPENACC_VERSION1, Property.SYNC_API);
    add("acc_async_wait",     Property.OPENACC_VERSION1, Property.SYNC_API);
    add("acc_async_wait_all",     Property.OPENACC_VERSION1, Property.SYNC_API);
    add("acc_async_test",     Property.OPENACC_VERSION1, Property.SYNC_API);
    add("acc_malloc",     Property.OPENACC_VERSION1, Property.MEMORY_API, Property.DEVICE_MALLOC);
    add("acc_free",     Property.OPENACC_VERSION1, Property.MEMORY_API, Property.DEVICE_FREE, Property.SIDE_EFFECT_PARAMETER);
    // OpenACC version 2.0 library
    add("acc_wait",     Property.OPENACC_VERSION2, Property.SYNC_API);
    add("acc_wait_all",     Property.OPENACC_VERSION2, Property.SYNC_API);
    add("acc_wait_async",     Property.OPENACC_VERSION2, Property.SYNC_API);
    add("acc_wait_all_async",     Property.OPENACC_VERSION2, Property.SYNC_API);
    add("acc_copyin",     Property.OPENACC_VERSION2, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.MEM_COPYIN);
    add("acc_pcopyin",     Property.OPENACC_VERSION2, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.MEM_COPYIN, 
    		Property.CHECK_PRESENT);
    add("acc_present_or_copyin",     Property.OPENACC_VERSION2, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.MEM_COPYIN, 
    		Property.CHECK_PRESENT);
    add("acc_create",     Property.OPENACC_VERSION2, Property.MEMORY_API, Property.DEVICE_MALLOC);
    add("acc_pcreate",     Property.OPENACC_VERSION2, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.CHECK_PRESENT);
    add("acc_present_or_create",     Property.OPENACC_VERSION2, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.CHECK_PRESENT);
    add("acc_copyout",     Property.OPENACC_VERSION2, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.MEM_COPYOUT);
    add("acc_delete",     Property.OPENACC_VERSION2, Property.MEMORY_API, Property.DEVICE_FREE);
    add("acc_update_device",     Property.OPENACC_VERSION2, Property.MEMORY_API, Property.MEM_COPYIN);
    add("acc_update_self",     Property.OPENACC_VERSION2, Property.MEMORY_API, Property.MEM_COPYOUT, Property.SIDE_EFFECT_PARAMETER);
    add("acc_map_data",     Property.OPENACC_VERSION2, Property.CONFIG_API);
    add("acc_unmap_data",     Property.OPENACC_VERSION2, Property.CONFIG_API);
    add("acc_deviceptr",     Property.OPENACC_VERSION2, Property.CONFIG_API);
    add("acc_hostptr",     Property.OPENACC_VERSION2, Property.CONFIG_API);
    add("acc_is_present",     Property.OPENACC_VERSION2, Property.CONFIG_API);
    add("acc_memcpy_to_device",     Property.OPENACC_VERSION2, Property.MEMORY_API, Property.MEM_COPYIN, Property.SIDE_EFFECT_PARAMETER);
    add("acc_memcpy_from_device",     Property.OPENACC_VERSION2, Property.MEMORY_API, Property.MEM_COPYIN, Property.SIDE_EFFECT_PARAMETER);
    // OpenARC library
    add("acc_copyin_unified",     Property.OPENARC_EXTENSION, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.MEM_COPYIN);
    add("acc_pcopyin_unified",     Property.OPENARC_EXTENSION, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.MEM_COPYIN, 
    		Property.CHECK_PRESENT);
    add("acc_present_or_copyin_unified",     Property.OPENARC_EXTENSION, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.MEM_COPYIN, 
    		Property.CHECK_PRESENT);
    add("acc_create_unified",     Property.OPENARC_EXTENSION, Property.MEMORY_API, Property.DEVICE_MALLOC);
    add("acc_pcreate_unified",     Property.OPENARC_EXTENSION, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.CHECK_PRESENT);
    add("acc_present_or_create_unified",     Property.OPENARC_EXTENSION, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.CHECK_PRESENT);
    add("acc_copyout_unified",     Property.OPENARC_EXTENSION, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.MEM_COPYOUT);
    add("acc_delete_unified",     Property.OPENARC_EXTENSION, Property.MEMORY_API, Property.DEVICE_FREE);
    add("acc_copyin_const",     Property.OPENARC_EXTENSION, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.MEM_COPYIN);
    add("acc_pcopyin_const",     Property.OPENARC_EXTENSION, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.MEM_COPYIN, 
    		Property.CHECK_PRESENT);
    add("acc_present_or_copyin_const",     Property.OPENARC_EXTENSION, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.MEM_COPYIN, 
    		Property.CHECK_PRESENT);
    add("acc_create_const",     Property.OPENARC_EXTENSION, Property.MEMORY_API, Property.DEVICE_MALLOC);
    add("acc_pcreate_const",     Property.OPENARC_EXTENSION, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.CHECK_PRESENT);
    add("acc_present_or_create_const",     Property.OPENARC_EXTENSION, Property.MEMORY_API, Property.DEVICE_MALLOC, Property.CHECK_PRESENT);
  }

  /** Adds the specified properties to the call */
  private void add(String name, Property... properties)
  {
    catalog.put(name, EnumSet.noneOf(Property.class));
    Set<Property> props = catalog.get(name);
    for ( Property property : properties )
      props.add(property);
  }
}
