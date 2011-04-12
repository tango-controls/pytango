.. _revision:

Revision
--------

:Contributers: T\. Coutinho

:Last Update: |today|

.. _history-modifications:

History of modifications:

+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
|   Date   | Revision                                                                         |                          Description                | Author                |
+==========+==================================================================================+=====================================================+=======================+
| 18/07/03 | 1.0                                                                              | Initial Version                                     | M\. Ounsy             |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 06/10/03 | 2.0                                                                              | Extension of the "Getting Started" paragraph        | A\. Buteau/M\. Ounsy  |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 14/10/03 | 3.0                                                                              | Added Exception Handling paragraph                  | M\. Ounsy             |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 13/06/05 | 4.0                                                                              | Ported to Latex, added events, AttributeProxy       | V\. Forchì            |
|          |                                                                                  | and ApiUtil                                         |                       |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
|          |                                                                                  | fixed bug with python 2.5 and and state events      |                       |
| 13/06/05 | 4.1                                                                              | new Database constructor                            | V\. Forchì            |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 15/01/06 | 5.0                                                                              | Added Device Server classes                         | E\.Taurel             |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 15/03/07 | 6.0                                                                              | Added AttrInfoEx, AttributeConfig events, 64bits,   | T\. Coutinho          |
|          |                                                                                  | write_attribute                                     |                       |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 21/03/07 | 6.1                                                                              | Added groups                                        | T\. Coutinho          |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 15/06/07 | `6.2 <http://www.tango-controls.org/Documents/bindings/PyTango-3.0.3.pdf>`_      | Added dynamic attributes doc                        | E\. Taurel            |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 06/05/08 | `7.0 <http://www.tango-controls.org/Documents/bindings/PyTango-3.0.4.pdf>`_      | Update to Tango 6.1. Added DB methods, version info | T\. Coutinho          |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 10/07/09 | `8.0 <http://www.tango-controls.org/static/PyTango/v7/doc/html/index.html>`_     | Update to Tango 7. Major refactoring. Migrated doc  | T\. Coutinho/R\. Suñe |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 24/07/09 | `8.1 <http://www.tango-controls.org/static/PyTango/v7/doc/html/index.html>`_     | Added migration info, added missing API doc         | T\. Coutinho/R\. Suñe |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 21/09/09 | `8.2 <http://www.tango-controls.org/static/PyTango/v7/doc/html/index.html>`_     | Added migration info, release of 7.0.0beta2         | T\. Coutinho/R\. Suñe |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 12/11/09 | `8.3 <http://www.tango-controls.org/static/PyTango/v71/doc/html/index.html>`_    | Update to Tango 7.1.                                | T\. Coutinho/R\. Suñe |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| ??/12/09 | `8.4 <http://www.tango-controls.org/static/PyTango/v71rc1/doc/html/index.html>`_ | Update to PyTango 7.1.0 rc1                         | T\. Coutinho/R\. Suñe |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 19/02/10 | `8.5 <http://www.tango-controls.org/static/PyTango/v711/doc/html/index.html>`_   | Update to PyTango 7.1.1                             | T\. Coutinho/R\. Suñe |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 06/08/10 | `8.6 <http://www.tango-controls.org/static/PyTango/v712/doc/html/index.html>`_   | Update to PyTango 7.1.2                             | T\. Coutinho          |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 05/11/10 | `8.7 <http://www.tango-controls.org/static/PyTango/v713/doc/html/index.html>`_   | Update to PyTango 7.1.3                             | T\. Coutinho          |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 08/04/11 | `8.8 <http://www.tango-controls.org/static/PyTango/v714/doc/html/index.html>`_   | Update to PyTango 7.1.4                             | T\. Coutinho          |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+
| 12/04/11 | `8.9 <http://www.tango-controls.org/static/PyTango/v715/doc/html/index.html>`_   | Update to PyTango 7.1.5                             | T\. Coutinho          |
+----------+----------------------------------------------------------------------------------+-----------------------------------------------------+-----------------------+

.. _version-history:

Version history
---------------

+------------+-------------------------------------------------------------------------------------+
| version    | Changes                                                                             |
+============+=====================================================================================+
| 7.1.5      |                                                                                     |
|            | Bug fixes:                                                                          |
|            |     - from sourceforge:                                                             |
|            |         - 3284174: 7.1.4 does not build with gcc 4.5 and tango 7.2.6                |
|            |         - 3284265: [pytango][7.1.4] a few files without licence and copyright       |
|            |         - 3284318: copyleft vs copyright                                            |
|            |         - 3284434: [pytango][doc] few ERROR during the doc generation               |
|            |         - 3284435: [pytango][doc] few warning during the doc generation             |
+------------+-------------------------------------------------------------------------------------+
| 7.1.4      | Features:                                                                           |
|            |     - from sourceforge:                                                             |
|            |         - 3274309: Generic Callback for events                                      |
|            |                                                                                     |
|            | Bug fixes:                                                                          |
|            |     - from sourceforge:                                                             |
|            |         - 3011775: Seg Faults due to removed dynamic attributes                     |
|            |         - 3105169: PyTango 7.1.3 does not compile with Tango 7.2.X                  |
|            |         - 3107243: spock profile does not work with python 2.5                      |
|            |         - 3124427: PyTango.WAttribute.set_max_value() changes min value             |
|            |         - 3170399: Missing documentation about is_<attr>_allowed method             |
|            |         - 3189082: Missing get_properties() for Attribute class                     |
|            |         - 3196068: delete_device() not called after server_admin.Kill()             |
|            |         - 3257286: Binding crashes when reading a WRITE string attribute            |
|            |         - 3267628: DP.read_attribute(, extract=List/tuple) write value is wrong     |
|            |         - 3274262: Database.is_multi_tango_host missing                             |
|            |         - 3274319: EncodedAttribute is missing in PyTango (<= 7.1.3)                |
|            |         - 3277269: read_attribute(DevEncoded) is not numpy as expected              |
|            |         - 3278946: DeviceAttribute copy constructor is not working                  |
|            |                                                                                     |
|            | Documentation:                                                                      |
|            |     - Added :ref:`utilities` chapter                                                |
|            |     - Added :ref:`encoded` chapter                                                  |
|            |     - Improved :ref:`server` chapter                                                |
+------------+-------------------------------------------------------------------------------------+
| 7.1.3      | Features:                                                                           |
|            |     - tango logging with print statement                                            |
|            |     - tango logging with decorators                                                 |
|            |     - from sourceforge:                                                             |
|            |         - 3060380: ApiUtil should be exported to PyTango                            |
|            |                                                                                     |
|            | Bug fixes:                                                                          |
|            |     - added licence header to all source code files                                 |
|            |     - spock didn't work without TANGO_HOST env. variable (it didn't recognize       |
|            |       tangorc)                                                                      |
|            |     - spock should give a proper message if it tries to be initialized outside      |
|            |       ipython                                                                       |
|            |     - from sourceforge:                                                             |
|            |         - 3048798: licence issue GPL != LGPL                                        |
|            |         - 3073378: DeviceImpl.signal_handler raising exception crashes DS           |
|            |         - 3088031: Python DS unable to read DevVarBooleanArray property             |
|            |         - 3102776: PyTango 7.1.2 does not work with python 2.4 & boost 1.33.0       |
|            |         - 3102778: Fix compilation warnings in linux                                |
+------------+-------------------------------------------------------------------------------------+
| 7.1.2      | Features:                                                                           |
|            |     - from sourceforge:                                                             |
|            |         - 2995964: Dynamic device creation                                          |
|            |         - 3010399: The DeviceClass.get_device_list that exists in C++ is missing    |
|            |         - 3023686: Missing DeviceProxy.<attribute name>                             |
|            |         - 3025396: DeviceImpl is missing some CORBA methods                         |
|            |         - 3032005: IPython extension for PyTango                                    |
|            |         - 3033476: Make client objects pickable                                     |
|            |         - 3039902: PyTango.Util.add_class would be useful                           |
|            |                                                                                     |
|            | Bug fixes:                                                                          |
|            |     - from sourceforge:                                                             |
|            |         - 2975940: DS command with DevVarCharArray return type fails                |
|            |         - 3000467: DeviceProxy.unlock is LOCKING instead of unlocking!              |
|            |         - 3010395: Util.get_device_* methods don't work                             |
|            |         - 3010425: Database.dev_name does not work                                  |
|            |         - 3016949: command_inout_asynch callback does not work                      |
|            |         - 3020300: PyTango does not compile with gcc 4.1.x                          |
|            |         - 3030399: Database put(delete)_attribute_alias generates segfault          |
+------------+-------------------------------------------------------------------------------------+
| 7.1.1      | Features:                                                                           |
|            |     - Improved setup script                                                         |
|            |     - Interfaced with PyPI                                                          |
|            |     - Cleaned build script warnings due to unclean python C++ macro definitions     |
|            |     - from sourceforge: 2985993, 2971217                                            |
|            |                                                                                     |
|            | Bug fixes:                                                                          |
|            |     - from sourceforge: 2983299, 2953689, 2953030                                   |
+------------+-------------------------------------------------------------------------------------+
| 7.1.0      | Features:                                                                           |
|            |     - from sourceforge:                                                             |
|            |       - 2908176: read_*, write_* and is_*_allowed() methods can now be defined      |
|            |       - 2941036: TimeVal conversion to time and datetime                            |
|            |     - added str representation on Attr, Attribute, DeviceImpl and DeviceClass       |
|            |                                                                                     |
|            | Bug fixes:                                                                          |
|            |     - from sourceforge: 2903755, 2908176, 2914194, 2909927, 2936173, 2949099        |
+------------+-------------------------------------------------------------------------------------+
| 7.1.0rc1   | Features:                                                                           |
|            |     - v = image_attribute.get_write_value() returns square sequences (arrays of     |
|            |       arrays, or numpy objects) now instead of flat lists. Also for spectrum        |
|            |       attributes a numpy is returned by default now instead.                        |
|            |     - image_attribute.set_value(v) accepts numpy arrays now or square sequences     |
|            |       instead of just flat lists. So, dim_x and dim_y are useless now. Also the     |
|            |       numpy path is faster.                                                         |
|            |     - new enum AttrSerialModel                                                      |
|            |     - Attribute new methods: set(get)_attr_serial_model, set_change_event,          |
|            |       set_archive_event, is_change_event, is_check_change_event,                    |
|            |       is_archive_criteria, is_check_archive_criteria, remove_configuration          |
|            |     - added support for numpy scalars in tango operations like write_attribute      |
|            |       (ex: now a DEV_LONG attribute can receive a numpy.int32 argument in a         |
|            |       write_attribute method call)                                                  |
|            |                                                                                     |
|            | Bug fixes:                                                                          |
|            |     - DeviceImpl.set_value for scalar attributes                                    |
|            |     - DeviceImpl.push_***_event                                                     |
|            |     - server commands with DevVar***StringArray as parameter or as return type      |
|            |     - in windows,a bug in PyTango.Util prevented servers from starting up           |
|            |     - DeviceImpl.get_device_properties for string properties assigns only first     |
|            |       character of string to object member instead of entire string                 |
|            |     - added missing methods to Util                                                 |
|            |     - exported SubDevDiag class                                                     |
|            |     - error in read/events of attributes of type DevBoolean READ_WRITE              |
|            |     - error in automatic unsubscribe events of DeviceProxy when the object          |
|            |       disapears (happens only on some compilers with some optimization flags)       |
|            |     - fix possible bug when comparing attribute names in DeviceProxy                |
|            |     - pretty print of DevFailed -> fix deprecation warning in python 2.6            |
|            |     - device class properties where not properly fetched when there is no           |
|            |       property value defined                                                        |
|            |     - memory leak when converting DevFailed exceptions from C++ to python           |
|            |     - python device server file without extension does not start                    |
|            |                                                                                     |
|            | Documentation:                                                                      |
|            |     - Improved FAQ                                                                  |
|            |     - Improved compilation chapter                                                  |
|            |     - Improved migration information                                                |
+------------+-------------------------------------------------------------------------------------+
