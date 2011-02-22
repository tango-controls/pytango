
.. currentmodule:: PyTango

.. _spock-highlights:

Highlights
==========

Tab completion
--------------

Spock exports many tango specific objects to the console namespace.
These include:

    - the PyTango module itself
      
      .. sourcecode:: spock

            Spock <homer:10000> [1]: PyTango
                         Result [1]: <module 'PyTango' from ...>
                         
    - The :class:`DeviceProxy` (=Device), :class:`AttributeProxy` (=Attribute),
      :class:`Database` and :class:`Group` classes
      
      .. sourcecode:: spock

            Spock <homer:10000> [1]: De<tab>
            DeprecationWarning            Device       DeviceProxy

            Spock <homer:10000> [2]: Device
                         Result [2]: <class 'PyTango._PyTango.DeviceProxy'>
            
            Spock <homer:10000> [3]: Device("sys/tg_test/1")
                         Result [3]: DeviceProxy(sys/tg_test/1)
                         
            Spock <homer:10000> [4]: Datab<tab>
            
            Spock <homer:10000> [4]: Database
            
            Spock <homer:10000> [4]: Att<tab>
            Attribute       AttributeError  AttributeProxy
            
    - The tango :class:`PyTango.Database` object to which the spock session is 
      currently connected
      
      .. sourcecode:: spock

            Spock <homer:10000> [1]: db
                         Result [1]: Database(homer, 10000)
    
Device name completion
----------------------

Spock knows the complete list of device names (including alias) for the current
tango database. This means that when you try to create a new Device, by pressing
<tab> you can see a context sensitive list of devices.

.. sourcecode:: spock

    Spock <homer:10000> [1]: test = Device("<tab>
    Display all 3654 possibilities? (y or n) n
    
    Spock <homer:10000> [1]: test = Device("sys<tab>
    sys/access_control/1  sys/database/2        sys/tautest/1         sys/tg_test/1
    
    Spock <homer:10000> [2]: test = Device("sys/tg_test/1")

Attribute name completion
-------------------------

Spock can inspect the list of attributes in case the device server for the device
where the attribute resides is running.

.. sourcecode:: spock

    Spock <homer:10000> [1]: short_scalar = Attribute("sys<tab>
    sys/access_control/1/  sys/database/2/        sys/tautest/1/         sys/tg_test/1/
    
    Spock <homer:10000> [1]: short_scalar = Attribute("sys/tg_test/1/<tab>
    sys/tg_test/1/State                sys/tg_test/1/no_value
    sys/tg_test/1/Status               sys/tg_test/1/short_image
    sys/tg_test/1/ampli                sys/tg_test/1/short_image_ro
    sys/tg_test/1/boolean_image        sys/tg_test/1/short_scalar
    sys/tg_test/1/boolean_image_ro     sys/tg_test/1/short_scalar_ro
    sys/tg_test/1/boolean_scalar       sys/tg_test/1/short_scalar_rww
    sys/tg_test/1/boolean_spectrum     sys/tg_test/1/short_scalar_w
    sys/tg_test/1/boolean_spectrum_ro  sys/tg_test/1/short_spectrum
    sys/tg_test/1/double_image         sys/tg_test/1/short_spectrum_ro
    sys/tg_test/1/double_image_ro      sys/tg_test/1/string_image
    sys/tg_test/1/double_scalar        sys/tg_test/1/string_image_ro
    ...

    Spock <homer:10000> [1]: short_scalar = Attribute("sys/tg_test/1/shost_scalar")
    
    Spock <homer:10000> [29]: print test.read()
    DeviceAttribute[
    data_format = PyTango._PyTango.AttrDataFormat.SCALAR
      dim_x = 1
      dim_y = 0
    has_failed = False
    is_empty = False
       name = 'short_scalar'
    nb_read = 1
    nb_written = 1
    quality = PyTango._PyTango.AttrQuality.ATTR_VALID
    r_dimension = AttributeDimension(dim_x = 1, dim_y = 0)
       time = TimeVal(tv_nsec = 0, tv_sec = 1279723723, tv_usec = 905598)
       type = PyTango._PyTango.CmdArgType.DevShort
      value = 47
    w_dim_x = 1
    w_dim_y = 0
    w_dimension = AttributeDimension(dim_x = 1, dim_y = 0)
    w_value = 0]

Automatic tango object member completion
----------------------------------------

When you create a new tango object, (ex.: a device), spock is able to find out
dynamically which are the members of this device (including tango commands 
and attributes if the device is currently running)

.. sourcecode:: spock

    Spock <homer:10000> [1]: test = Device("sys/tg_test/1")
    
    Spock <homer:10000> [2]: test.<tab>
    Display all 240 possibilities? (y or n)
    ...
    test.DevVoid                            test.get_access_control
    test.Init                               test.get_asynch_replies
    test.State                              test.get_attribute_config
    test.Status                             test.get_attribute_config_ex
    test.SwitchStates                       test.get_attribute_list
    ...
    
    Spock <homer:10000> [2]: test.short_<tab>
    test.short_image        test.short_scalar       test.short_scalar_rww   test.short_spectrum
    test.short_image_ro     test.short_scalar_ro    test.short_scalar_w     test.short_spectrum_ro

    Spock <homer:10000> [2]: test.short_scalar        # old style: test.read_attribute("short_scalar").value
                 Result [2]: 252

    Spock <homer:10000> [3]: test.Dev<tab>
    test.DevBoolean               test.DevUShort                test.DevVarShortArray
    test.DevDouble                test.DevVarCharArray          test.DevVarStringArray
    test.DevFloat                 test.DevVarDoubleArray        test.DevVarULongArray
    test.DevLong                  test.DevVarDoubleStringArray  test.DevVarUShortArray
    test.DevShort                 test.DevVarFloatArray         test.DevVoid
    test.DevString                test.DevVarLongArray          
    test.DevULong                 test.DevVarLongStringArray
    
    Spock <homer:10000> [3]: test.DevDouble(56.433)  # old style: test.command_inout("DevDouble").
                 Result [3]: 56.433

Tango classes as :class:`DeviceProxy`
---------------------------------------------

Spock exports all known tango classes as python alias to :class:`DeviceProxy`. 
This way, if you want to create a device of class which you already know 
(say, Libera, for example) you can do:

.. sourcecode:: spock

    Spock <homer:10000> [1]: lib01 = Libera("BO01/DI/BPM-01")

One great advantage is that the tango device name completion is sensitive to the
type of device you want to create. This means that if you are in the middle of
writting a device name and you press the <tab> key, only devices of the tango
class 'Libera' will show up as possible completions.

.. sourcecode:: spock

    Spock <homer:10000> [1]: bpm1 = Libera("<tab>
    BO01/DI/BPM-01  BO01/DI/BPM-09  BO02/DI/BPM-06  BO03/DI/BPM-03  BO03/DI/BPM-11  BO04/DI/BPM-08
    BO01/DI/BPM-02  BO01/DI/BPM-10  BO02/DI/BPM-07  BO03/DI/BPM-04  BO04/DI/BPM-01  BO04/DI/BPM-09
    BO01/DI/BPM-03  BO01/DI/BPM-11  BO02/DI/BPM-08  BO03/DI/BPM-05  BO04/DI/BPM-02  BO04/DI/BPM-10
    BO01/DI/BPM-04  BO02/DI/BPM-01  BO02/DI/BPM-09  BO03/DI/BPM-06  BO04/DI/BPM-03  BO04/DI/BPM-11
    BO01/DI/BPM-05  BO02/DI/BPM-02  BO02/DI/BPM-10  BO03/DI/BPM-07  BO04/DI/BPM-04  
    BO01/DI/BPM-06  BO02/DI/BPM-03  BO02/DI/BPM-11  BO03/DI/BPM-08  BO04/DI/BPM-05  
    BO01/DI/BPM-07  BO02/DI/BPM-04  BO03/DI/BPM-01  BO03/DI/BPM-09  BO04/DI/BPM-06  
    BO01/DI/BPM-08  BO02/DI/BPM-05  BO03/DI/BPM-02  BO03/DI/BPM-10  BO04/DI/BPM-07

    Spock <homer:10000> [1]: bpm1 = Libera("BO01<tab>
    BO01/DI/BPM-01  BO01/DI/BPM-03  BO01/DI/BPM-05  BO01/DI/BPM-07  BO01/DI/BPM-09  BO01/DI/BPM-11
    BO01/DI/BPM-02  BO01/DI/BPM-04  BO01/DI/BPM-06  BO01/DI/BPM-08  BO01/DI/BPM-10
    
    Spock <homer:10000> [1]: bpm1 = Libera("BO01/DI/BPM-
    
    
List tango devices, classes, servers
------------------------------------

Spock provides a set of magic functions (ipython lingo) that allow you to check
for the list tango devices, classes and servers which are registered in the 
current database.

.. sourcecode:: spock

    Spock <homer:10000> [1]: lsdev
                                      Device                     Alias                    Server                Class
    ---------------------------------------- ------------------------- ------------------------- --------------------
                  expchan/BL99_Dummy0DCtrl/1                  BL99_0D1                 Pool/BL99      ZeroDExpChannel
                      simulator/bl98/motor08                                      Simulator/BL98            SimuMotor
                  expchan/BL99_Dummy0DCtrl/3                  BL99_0D3                 Pool/BL99      ZeroDExpChannel
                  expchan/BL99_Dummy0DCtrl/2                  BL99_0D2                 Pool/BL99      ZeroDExpChannel
                  expchan/BL99_Dummy0DCtrl/5                  BL99_0D5                 Pool/BL99      ZeroDExpChannel
                  expchan/BL99_Dummy0DCtrl/4                  BL99_0D4                 Pool/BL99      ZeroDExpChannel
                  expchan/BL99_Dummy0DCtrl/7                  BL99_0D7                 Pool/BL99      ZeroDExpChannel
                  expchan/BL99_Dummy0DCtrl/6                  BL99_0D6                 Pool/BL99      ZeroDExpChannel
                      simulator/bl98/motor01                                      Simulator/BL98            SimuMotor
                      simulator/bl98/motor02                                      Simulator/BL98            SimuMotor
                      simulator/bl98/motor03                                      Simulator/BL98            SimuMotor
       mg/BL99/_mg_macserv_26065_-1320158352                                           Pool/BL99           MotorGroup
                      simulator/bl98/motor05                                      Simulator/BL98            SimuMotor
                      simulator/bl98/motor06                                      Simulator/BL98            SimuMotor
                      simulator/bl98/motor07                                      Simulator/BL98            SimuMotor
                    simulator/BL98/motctrl01                                      Simulator/BL98        SimuMotorCtrl
                  expchan/BL99_Simu0DCtrl1/1                  BL99_0D8                 Pool/BL99      ZeroDExpChannel
                 expchan/BL99_UxTimerCtrl1/1                BL99_Timer                 Pool/BL99         CTExpChannel
    ...
    
    Spock <homer:10000> [1]: lsdevclass
    SimuCoTiCtrl                   TangoAccessControl             ZeroDExpChannel
    Door                           Motor                          DataBase
    MotorGroup                     IORegister                     SimuMotorCtrl
    TangoTest                      MacroServer                    TauTest
    SimuMotor                      SimuCounterEx                  MeasurementGroup
    Pool                           CTExpChannel

    Spock <homer:10000> [1]: lsserv
    MacroServer/BL99               MacroServer/BL98               Pool/V2
    Pool/BL99                      Pool/BL98                      TangoTest/test
    Pool/tcoutinho                 Simulator/BL98
    TangoAccessControl/1           TauTest/tautest                DataBaseds/2
    MacroServer/tcoutinho          Simulator/BL99

Customized tango error message and introspection
------------------------------------------------

Spock intercepts tango exceptions that occur when you do tango operations 
(ex.: write an attribute with a value outside the allowed limits) and tries to
display it in a summarized, user friendly way.
If you need more detailed information about the last tango error, you can use
the magic command 'tango_error'.

.. sourcecode:: spock

    Spock <homer:10000> [1]: test = Device("sys/tg_test/1")

    Spock <homer:10000> [2]: test.no_value
    API_AttrValueNotSet : Read value for attribute no_value has not been updated
    For more detailed information type: tango_error

    Spock <homer:10000> [3]: tango_error
    Last tango error:
    DevFailed[
    DevError[
        desc = 'Read value for attribute no_value has not been updated'
      origin = 'Device_3Impl::read_attributes_no_except'
      reason = 'API_AttrValueNotSet'
    severity = PyTango._PyTango.ErrSeverity.ERR]
    DevError[
        desc = 'Failed to read_attribute on device sys/tg_test/1, attribute no_value'
      origin = 'DeviceProxy::read_attribute()'
      reason = 'API_AttributeFailed'
    severity = PyTango._PyTango.ErrSeverity.ERR]]

Switching database
------------------

You can switch database simply by executing the 'switchdb <host> [<port>]' magic
command.

.. sourcecode:: spock

    Spock <homer:10000> [1]: switchdb

    Must give new database name in format <host>[:<port>].
    <port> is optional. If not given it defaults to 10000.

    Examples:
    switchdb homer:10005
    switchdb homer 10005
    switchdb homer
    
    Spock <homer:10000> [2]: switchdb bart      # by default port is 10000
    
    Spock <bart:10000> [3]: switchdb lisa 10005  # you can use spaces between host and port
    
    Spock <lisa:10005> [4]: switchdb marge:10005 # or the traditional ':'

Refreshing the database
-----------------------

When spock starts up or when the database is switched, a query is made to the
tango Database device server which provides all necessary data. This
data is stored locally in a spock cache which is used to provide all the nice 
features.
If the Database server is changed in some way (ex: a new device server is registered),
the local database cache is not consistent anymore with the tango database.
Therefore, spock provides a magic command 'refreshdb' that allows you to reread
all tango information from the database.

.. sourcecode:: spock

    Spock <homer:10000> [1]: refreshdb
    
Storing your favorite tango objects for later usage
---------------------------------------------------

Since version 7.1.2, :class:`DeviceProxy`, :class:`AttributeProxy` and 
:class:`Database` became pickable.
This means that they can be used by the IPython_ 'store' magic command (type
'store?' on the spock console to get information on how to use this command).
You can, for example, assign your favorite devices in local python variables and
then store these for the next time you startup IPython_ with spock profile.

.. sourcecode:: spock

    Spock <homer:10000> [1]: theta = Motor("BL99_M1")  # notice how we used tango alias
    
    Spock <homer:10000> [2]: store theta
    Stored 'theta' (DeviceProxy)
    
    Spock <homer:10000> [3]: Ctrl+D
    
    (IPython session is closed and started again...)
    
    Spock <homer:10000> [1]: print theta
    DeviceProxy(motor/bl99/1)
    
Tango color modes
-----------------

IPython_ (0.10) provides three color modes: 'Linux', 'NoColor' and 'LightBG'.
Spock provides in addition 'Tango' (default), 'PurpleTango' (='Tango'),
'BlueTango' and 'GreenTango'.

You can switch between color modes online using the magic command 'colors'.

.. sourcecode:: spock

    Spock <homer:10000> [1]: colors BlueTango
    
    Spock <homer:10000> [2]: colors NoColor

Adding spock to your own ipython profile
----------------------------------------

Adding spock to the ipython default profile
###########################################

Let's assume that you find spock so useful that each time you start ipython, you want
spock features to be loaded by default.
The way to do this is by editing your default ipython configuration file: 
$HOME/.ipython/ipy_user_conf.py and add the lines 1 and 7.

.. note::
    The code shown below is a small part of your $HOME/.ipython/ipy_user_conf.py.
    It is shown here only the relevant part for this example.

.. sourcecode:: python

    import PyTango.ipython

    def main():

        # uncomment if you want to get ipython -p sh behaviour
        # without having to use command line switches  
        # import ipy_profile_sh
        PyTango.ipython.init_ipython(ip, console=False)

And now, every time you start ipython::

    ipython

spock features will also be loaded.

.. sourcecode:: ipython

    In [1]: db
    Out[1]: Database(homer, 10000)


Adding spock to an existing customized profile
##############################################

If you have been working with IPython_ before and have already have defined a
customized personal profile, you can extend your profile with spock features 
without breaking your existing options. The trick is to initialize spock extension
with a parameter that tells spock to maintain the existing options (like colors,
command line and initial banner).

So, for example, let's say you have created a profile called nuclear, and therefore
you have a file called $HOME/.ipython/ipy_profile_nuclear.py with the following
contents:

.. sourcecode:: python

    import os
    import IPython.ipapi

    def main():
        ip = IPython.ipapi.get()
        
        o = ip.options
        o.banner = "Springfield nuclear powerplant CLI\n\nWelcome Homer Simpson"
        o.colors = "Linux"
        o.prompt_in1 = "Mr. Burns owns you [\\#]: "
        
    main()

In order to have spock features available to this profile you simply need to
add two lines of code (lines 3 and 7):

.. sourcecode:: python

    import os
    import IPython.ipapi
    import PyTango.ipython

    def main():
        ip = IPython.ipapi.get()
        PyTango.ipython.init_ipython(ip, console=False)
        
        o = ip.options
        o.banner = "Springfield nuclear powerplant CLI\n\nMr. Burns owns you!"
        o.colors = "Linux"
        o.prompt_in1 = "The Simpsons [\\#]: "
        
    main()

This will load the spock features into your profile while preserving your
profile's console options (like colors, command line and initial banner).

Creating a profile that extends spock profile
#############################################

It is also possible to create a profile that includes all spock features and at
the same time adds new ones. Let's suppose that you want to create a customized
profile called 'orbit' that automaticaly exports devices of class 
'Libera' for the booster accelerator (assuming you are working on a synchrotron
like institute ;-).
Here is the code for the $HOME/.ipython/ipy_profile_orbit.py:

.. sourcecode:: python

    import os
    import IPython.ipapi
    import IPython.genutils
    import IPython.ColorANSI
    import PyTango.ipython
    import StringIO

    def magic_liberas(ip, p=''):
        """Lists all known Libera devices."""
        data = PyTango.ipython.get_device_map()
        s = StringIO.StringIO()
        cols = 30, 15, 20
        l = "%{0}s %{1}s %{2}s".format(*cols)
        print >>s, l % ("Device", "Alias", "Server")
        print >>s, l % (cols[0]*"-", cols[1]*"-", cols[2]*"-")
        for d, v in data.items():
            if v[2] != 'Libera': continue
            print >>s, l % (d, v[0], v[1])
        s.seek(0)
        IPython.genutils.page(s.read())

    def main():
        ip = IPython.ipapi.get()

        PyTango.ipython.init_ipython(ip)

        o = ip.options
        
        Colors = IPython.ColorANSI.TermColors
        c = dict(Colors.__dict__)

        o.banner += "\n{Brown}Welcome to Orbit analysis{Normal}\n".format(**c)

        o.prompt_in1 = "Orbit [\\#]: "
        o.colors = "BlueTango"
        
        ip.expose_magic("liberas", magic_liberas)

        db = ip.user_ns.get('db')
        dev_class_dict = PyTango.ipython.get_class_map()

        if not dev_class_dict.has_key("Libera"):
            return
        
        for libera in dev_class_dict['Libera']:
            domain, family, member = libera.split("/")
            var_name = domain + "_" + member
            var_name = var_name.replace("-","_")
            ip.to_user_ns( { var_name : PyTango.DeviceProxy(libera) } )

    main()

Then start your CLI with::

    ipython -p orbit

and you will have something like this

.. image:: spock02.png

Advanced event monitoring
-------------------------

With spock it is possible to monitor change events triggered by any tango
attribute which has events enabled.

To start monitoring the change events of an attribute:

.. sourcecode:: spock

    Spock <homer:10000> [1]: mon -a BL99_M1/Position
    'BL99_M1/Position' is now being monitored. Type 'mon' to see all events
    
To list all events that have been intercepted:

.. sourcecode:: spock

    Spock <homer:10000> [2]: mon
      ID           Device    Attribute            Value       Quality             Time
    ---- ---------------- ------------ ---------------- ------------- ----------------
       0     motor/bl99/1        state               ON    ATTR_VALID  17:11:08.026472
       1     motor/bl99/1     position            190.0    ATTR_VALID  17:11:20.691112
       2     motor/bl99/1        state           MOVING    ATTR_VALID  17:12:11.858985
       3     motor/bl99/1     position    188.954072857 ATTR_CHANGING  17:12:11.987817
       4     motor/bl99/1     position    186.045533882 ATTR_CHANGING  17:12:12.124448
       5     motor/bl99/1     position    181.295838155 ATTR_CHANGING  17:12:12.260884
       6     motor/bl99/1     position     174.55354729 ATTR_CHANGING  17:12:12.400036
       7     motor/bl99/1     position     166.08870515 ATTR_CHANGING  17:12:12.536387
       8     motor/bl99/1     position     155.77528943 ATTR_CHANGING  17:12:12.672846
       9     motor/bl99/1     position    143.358230136 ATTR_CHANGING  17:12:12.811878
      10     motor/bl99/1     position    131.476140017 ATTR_CHANGING  17:12:12.950391
      11     motor/bl99/1     position    121.555421781 ATTR_CHANGING  17:12:13.087970
      12     motor/bl99/1     position    113.457930987 ATTR_CHANGING  17:12:13.226531
      13     motor/bl99/1     position    107.319423091 ATTR_CHANGING  17:12:13.363559
      14     motor/bl99/1     position    102.928229946 ATTR_CHANGING  17:12:13.505102
      15     motor/bl99/1     position    100.584726495 ATTR_CHANGING  17:12:13.640794
      16     motor/bl99/1     position            100.0    ATTR_ALARM  17:12:13.738136
      17     motor/bl99/1        state            ALARM    ATTR_VALID  17:12:13.743481

    Spock <homer:10000> [3]: mon -l mot.* state
      ID           Device    Attribute            Value       Quality             Time
    ---- ---------------- ------------ ---------------- ------------- ----------------
       0     motor/bl99/1        state               ON    ATTR_VALID  17:11:08.026472
       2     motor/bl99/1        state           MOVING    ATTR_VALID  17:12:11.858985
      17     motor/bl99/1        state            ALARM    ATTR_VALID  17:12:13.743481

To stop monitoring the attribute:

.. sourcecode:: spock

    Spock <homer:10000> [1]: mon -d BL99_M1/Position
    Stopped monitoring 'BL99_M1/Position'

.. note::
    Type 'mon?' to see detailed information about this magic command

.. warning::
     
    Experimental: The monitor table can be shown in a GUI. For this feature to 
    be available you need two things:

        #. Have PyQt4 (>= 4.1) installed
        #. Start spock with '-q4thread' support: 'ipython -q4thread -p spock'

    When you type '%mon' the following table should appear:

        .. image:: spock04.png

.. _IPython: http://ipython.scipy.org/
.. _Tango: http://www.tango-controls.org/
