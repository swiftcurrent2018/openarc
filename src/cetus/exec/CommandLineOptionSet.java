package cetus.exec;

import java.util.*;
import cetus.hir.PrintTools;

/**
* Class for registering and managing information of options.
*/
public class CommandLineOptionSet {

    public final int UTILITY = 1;
    public final int ANALYSIS = 2;
    public final int TRANSFORM = 3;
    public final int CODEGEN = 4;
    private final String[] type_name =
            {"UTILITY", "ANALYSIS", "TRANSFORM", "CODEGEN"};

    private class OptionRecord {

        public int option_type;
        public String value;
        // arg seems unnecessary => used for specifying suboption availability
        public String arg;
        public String usage;
        // default values when option is turned on without a value.
        private String value_on;
        // extra options for selecting specific hir for inclusion/exclusion.
        // these two members are not used for now
        private int select_type;        // 0 for exclusion 1 for inclusion
        private Map<String, List<String>> select_map;

        public OptionRecord(int type, String usage) {
            this.option_type = type;
            this.value = null;
            this.arg = null;
            this.usage = usage;
            this.value_on = null;
            select_type = 0;
            select_map = new HashMap<String, List<String>>();
        }
        
        public OptionRecord(int type, String arg, String usage) {
            this.option_type = type;
            this.value = null;
            this.arg = arg;
            this.usage = usage;
            this.value_on = null;
            select_type = 0;
            select_map = new HashMap<String, List<String>>();
        }
        
        public OptionRecord(int type, String value, String arg, String usage) {
            this.option_type = type;
            this.value = value;
            this.arg = arg;
            this.usage = usage;
            this.value_on = null;
            select_type = 0;
            select_map = new HashMap<String, List<String>>();
        }
    }

    /** Storage for the entire set of options. */
    private TreeMap<String, OptionRecord> name_to_record;

    /** Default constructor */
    public CommandLineOptionSet() {
        name_to_record = new TreeMap<String, OptionRecord>();
    }

    /**
    * Registers a new UTILITY option with the given usage information.
    * @param name the name of the option.
    * @param usage the usage of the option.
    */
    public void add(String name, String usage) {
        name_to_record.put(name, new OptionRecord(UTILITY, usage));
    }

    /**
    * Registers a new UTILITY option with the given argument and usage
    * information.
    * @param name the name of the option.
    * @param arg the argument allowed for the option.
    * @param usage the usage of the option.
    */
    public void add(String name, String arg, String usage) {
        name_to_record.put(name, new OptionRecord(UTILITY, arg, usage));
    }

    /**
    * Registers a new option with the given type and usage information.
    * @param type the category to which the option belongs.
    * @param name the name of the option.
    * @param usage the usage of the option.
    */
    public void add(int type, String name, String usage) {
        name_to_record.put(name, new OptionRecord(type, usage));
    }

    /**
    * Registers a new option with the given type, argument, and usage.
    * @param type the category to which the option belongs.
    * @param name the name of the option.
    * @param arg the argument allowed for the option.
    * @param usage the usage of the option.
    */
    public void add(int type, String name, String arg, String usage) {
        name_to_record.put(name, new OptionRecord(type, arg, usage));
    }

    /**
    * Registers a new option with the given type, value, argument, and usage.
    * @param type the category to which the option belongs.
    * @param name the name of the option.
    * @param value the default value of the option.
    * @param arg the argument allowed for the option.
    * @param usage the usage of the option.
    */
    public void add(int type,
                    String name, String value, String arg, String usage) {
        name_to_record.put(name, new OptionRecord(type, value, arg, usage));
    }

    /**
    * Registers a new option with the given type, value, turn-on value,
    * argument, and usage.
    * @param type the category to which the option belongs.
    * @param name the name of the option.
    * @param value the default value of the option.
    * @param value_on the default value when the option is turned on without a
    *                 value
    * @param arg the argument allowed for the option.
    * @param usage the usage of the option.
    */
    public void add(int type, String name,
                    String value, String value_on, String arg, String usage) {
        OptionRecord record = new OptionRecord(type, value, arg, usage);
        record.value_on = value_on;
        name_to_record.put(name, record);
    }

    /**
    * Checks if the option set contains an option with the specified name.
    * @param name the name to be searched for.
    * @return true if such an option exists.
    */
    public boolean contains(String name) {
        return name_to_record.containsKey(name);
    }

    /**
    * Returns a string dump of the options.
    */
    public String dumpOptions() {
        String retval = "";
        for (Map.Entry<String, OptionRecord>stringOptionRecordEntry :
                name_to_record.entrySet()) {
            OptionRecord record = stringOptionRecordEntry.getValue();
            // Print usage
            retval += "#Option: " + stringOptionRecordEntry.getKey();
            retval += "\n#";
            // Print option name and example value
            retval += stringOptionRecordEntry.getKey();
            if (record.arg != null) {
                retval += "=";
                retval += record.arg;
            }
            retval += "\n#";
            retval += record.usage.replaceAll("\n", "\n#");
            retval += "\n";
            // Print option name and default value
            retval += stringOptionRecordEntry.getKey();
            if (record.value != null) {
                retval += "=";
                retval += record.value;
            }
            retval += "\n";
        }
        return retval;
    }

    /**
    * Returns the usage information for the entire option set.
    */
    public String getUsage() {
        StringBuilder sb = new StringBuilder(8000);
        String sep = PrintTools.line_sep;
        for (int j = UTILITY; j <= CODEGEN; j++) {
            for (int i = 0; i < 80; i++) sb.append("-");
            sb.append(sep).append(type_name[j-1]).append(sep);
            for (int i = 0; i < 80; i++) sb.append("-");
            sb.append(sep).append(getUsage(j));
        }
        return sb.toString();
    }

    /**
    * Returns the usage information for the specified option type.
    */
    public String getUsage(int type) {
        String usage = "";
        for (Map.Entry<String, OptionRecord> stringOptionRecordEntry :
                name_to_record.entrySet()) {
            OptionRecord record = stringOptionRecordEntry.getValue();
            if (record.option_type == type) {
                usage += "-";
                usage += stringOptionRecordEntry.getKey();
                if (record.arg != null) {
                    usage += "=";
                    usage += record.arg;
                }
                usage += "\n    ";
                usage += record.usage;
                usage += "\n\n";
            }
        }
        return usage;
    }

    /**
    * Returns the value of the specified option name.
    */
    public String getValue(String name) {
        OptionRecord record = name_to_record.get(name);
        if (record == null) {
            return null;
        } else {
            return record.value;
        }
    }

    /**
    * Sets a new value for the specified option name.
    */
    public void setValue(String name, String value) {
        OptionRecord record = name_to_record.get(name);
        if (record != null) {
            record.value = value;
        }
    }

    /**
    * Copies the default value from predefined one. This is necessary for a
    * case where user does not specify a value but just turns on the option.
    */
    public void setValue(String name) {
        OptionRecord record = name_to_record.get(name);
        if (record != null) {
            if (record.value_on == null) {
                record.value = "1";
            } else {
                record.value = record.value_on;
            }
        }
    }

    /**
    * Returns the type of the specified option name.
    */
    public int getType(String name) {
        OptionRecord record = name_to_record.get(name);
        if (record == null) {
            return 0;
        } else {
            return record.option_type;
        }
    }

    /**
    * Includes the specified IR type and name in the inclusion set for the
    * specified option name.
    * @param name the affected option name.
    * @param hir_type the IR type.
    * @param hir_name the IR name.
    */
    public void include(String name, String hir_type, String hir_name) {
        OptionRecord record = name_to_record.get(name);
        if (record.select_map.get(hir_type) == null)
            record.select_map.put(hir_type, new LinkedList<String>());
        record.select_map.get(hir_type).add(hir_name);
        record.select_type = 1;
    }

    /**
    * Excludes the specified IR type and name in the inclusion set for the
    * specified option name.
    * @param name the affected option name.
    * @param hir_type the IR type.
    * @param hir_name the IR name.
    */
    public void exclude(String name, String hir_type, String hir_name) {
        OptionRecord record = name_to_record.get(name);
        if (record.select_map.get(hir_type) == null)
            record.select_map.put(hir_type, new LinkedList<String>());
        record.select_map.get(hir_type).add(hir_name);
        record.select_type = 0;
    }

    public boolean isIncluded(String name, String hir_type, String hir_name) {
        OptionRecord record = name_to_record.get(name);
        return ((record.select_type == 1 &&
                 record.select_map.get(hir_type) != null &&
                 record.select_map.get(hir_type).contains(hir_name))
                ||
                (record.select_type == 0 &&
                 (record.select_map.get(hir_type) == null ||
                  !record.select_map.get(hir_type).contains(hir_name)))
            );
    }

    public boolean isExcluded(String name, String hir_type, String hir_name) {
        return !isIncluded(name, hir_type, hir_name);
    }
}
