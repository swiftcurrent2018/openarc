package openacc.exec;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Provides access to target-specific configuration properties collected by
 * {@code configure.mk} from OpenARC's {@code makefile.header} into the file
 * {@code build.cfg}.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class BuildConfig {
  private static BuildConfig buildConfig = null;
  private final Properties properties = new Properties();
  private BuildConfig() {
    final InputStream cfg
      = ACC2GPUDriver.class.getResourceAsStream("build.cfg");
    if (cfg == null)
      throw new IllegalStateException("could not find config file");
    try {
      properties.load(cfg);
    } catch (IOException e) {
      throw new IllegalStateException("failure reading config file: " + e);
    }
  }
  public static BuildConfig getBuildConfig() {
    if (buildConfig != null)
      return buildConfig;
    return buildConfig = new BuildConfig();
  }
  public String getProperty(String name) {
    final String value = properties.getProperty(name);
    if (value == null)
      throw new IllegalStateException("config property is missing: " + name);
    return value;
  }
}
