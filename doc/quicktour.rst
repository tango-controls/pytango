.. _quick-tour:

A quick tour
------------

This quick tour will guide you through the first steps on using PyTango.
This is the new quick tour guide based on the :ref:`spock` console.
You can still find the old version of this tour based on a simple python
console :ref:`here <quick-tour-old>`.

Check PyTango version
~~~~~~~~~~~~~~~~~~~~~

Start an ipython spock console with::

    ipython -p spock

and type:

    .. sourcecode:: spock

        Spock <homer:10000> [1]: PyTango.__version__
                     Result [1]: '7.1.2'

        Spock <homer:10000> [2]: PyTango.__version_long__
                     Result [2]: '7.1.2dev0'

        Spock <homer:10000> [3]: PyTango.__version_number__
                     Result [3]: 712

        Spock <homer:10000> [4]: PyTango.__version_description__
                     Result [4]: 'This version implements the C++ Tango 7.1 API.'

or alternatively:

    .. sourcecode:: spock

        Spock <homer:10000> [1]: PyTango.Release.version
                     Result [1]: '7.1.2'

        Spock <homer:10000> [2]: PyTango.Release.version_long
                     Result [2]: '7.1.2dev0'

        Spock <homer:10000> [3]: PyTango.Release.version_number
                     Result [3]: 712

        Spock <homer:10000> [4]: PyTango.Release.version_description
                     Result [4]: 'This version implements the C++ Tango 7.1 API.'

.. tip::

    When typing, try pressing <tab>. Since Spock has autocomplete embedded you
    should get a list of possible completions. Example::
    
        PyTango.Release.<tab>
        
    Should get a list of all members of :class:`PyTango.Release` class.

Check Tango C++ version
~~~~~~~~~~~~~~~~~~~~~~~

From a client (This is only possible since PyTango 7.0.0)

    .. sourcecode:: spock

        Spock <homer:10000> [1]: import PyTango.constants

        Spock <homer:10000> [2]: PyTango.constants.TgLibVers
                     Result [2]: '7.1.1'

From a server you can alternatively do::
    
    u = PyTango.Util.instance()
    tg_cpp_lib_ver = u.get_tango_lib_release()
    

Test the connection to the Device and get it's current state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the most basic examples is to get a reference to a device and
determine if it is running or not.

    .. sourcecode:: spock
        
        Spock <homer:10000> [1]: # What is a DeviceProxy, really?
        Spock <homer:10000> [1]: DeviceProxy?
        DeviceProxy is the high level Tango object which provides the client with
        an easy-to-use interface to TANGO devices. DeviceProxy provides interfaces
        to all TANGO Device interfaces.The DeviceProxy manages timeouts, stateless
        connections and reconnection if the device server is restarted. To create
        a DeviceProxy, a Tango Device name must be set in the object constructor.

        Example :
           dev = PyTango.DeviceProxy("sys/tg_test/1")
           
        Spock <homer:10000> [2]: tangotest = DeviceProxy("sys/tg_test/1")

        Spock <homer:10000> [3]: # ping it
        Spock <homer:10000> [4]: tangotest.ping()
                     Result [4]: 110

        Spock <homer:10000> [3]: # Lets test the state
        Spock <homer:10000> [5]: tangotest.state()
                     Result [5]: PyTango._PyTango.DevState.RUNNING

        Spock <homer:10000> [3]: # And now the status
        Spock <homer:10000> [5]: tangotest.status()
                     Result [5]: 'The device is in RUNNING state.'

.. note::
    Did you notice that you didn't write PyTango.DeviceProxy but instead just DeviceProxy?
    This is because :ref:`spock` automatically exports the :class:`PyTango.DeviceProxy`,
    :class:`PyTango.AttributeProxy`, :class:`PyTango.Database` and :class:`PyTango.Group`
    classes to the namespace. If you are writting code outside :ref:`spock` you **MUST**
    use the `PyTango` module prefix.

.. tip::

    When typing the device name in the :class:`PyTango.DeviceProxy` creation
    line, try pressing the <tab> key. You should get a list of devices::
    
        tangotest = DeviceProxy("sys<tab>
        
    Better yet (and since the Tango Class of 'sys/tg_test/1' is 'TangoTest'),
    try doing::
    
        tangotest = TangoTest("<tab>

    Now the list of devices should be reduced to the ones that belong to the 
    'TangoTest' class. Note that TangoTest only works in Spock. If you are 
    writting code outside :ref:`spock` you **MUST** use 
    :class:`PyTango.DeviceProxy` instead.
    
Execute commands with scalar arguments on a Device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As you can see in the following example, when scalar types are used, PyTango
automatically manages the data types, and writing scripts is quite easy.

    .. sourcecode:: spock
    
        Spock <homer:10000> [1]: tangotest = TangoTest("sys/tg_test/1")

        Spock <homer:10000> [2]: # classical way
        Spock <homer:10000> [2]: r = tangotest.command_inout("DevString", "Hello, world!")

        Spock <homer:10000> [3]: print "Result of execution of DevString command =", r
        Result of execution of DevString command = Hello, world!

        Spock <homer:10000> [4]: # 'pythonic' way
        Spock <homer:10000> [5]: tangotest.DevString("Hello, world!")
                     Result [5]: 'Hello, world!'
        
        Spock <homer:10000> [6]: # type is automatically managed by PyTango
        Spock <homer:10000> [7]: tangotest.DevULong(12456)
                     Result [7]: 12456

Execute commands with more complex types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case you have to use put your arguments data in the correct python
structures.

    .. sourcecode:: spock
    
        Spock <homer:10000> [1]: tangotest = TangoTest("sys/tg_test/1")

        Spock <homer:10000> [2]: argin = [1, 2, 3], ["Hello", "World"]

        Spock <homer:10000> [3]: tango_test.DevVarLongArray(argin)
                     Result [3]: [array([1, 2, 3]), ['Hello', 'World']]
        
.. note::
    notice that the command returns a list of two elements. The first element is
    a :class:`numpy.ndarray` (assuming PyTango is compiled with numpy support).
    This is because PyTango does a best effort to convert all numeric array types
    to numpy arrays.
    
Reading and writing attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Basic read/write attribute operations.

    .. sourcecode:: spock
    
        Spock <homer:10000> [1]: # Read a scalar attribute
        Spock <homer:10000> [2]: print tangotest.read_attribute("long_scalar")
        DeviceAttribute[
        data_format = PyTango._PyTango.AttrDataFormat.SCALAR
              dim_x = 1
              dim_y = 0
         has_failed = False
           is_empty = False
               name = 'long_scalar'
            nb_read = 1
         nb_written = 1
            quality = PyTango._PyTango.AttrQuality.ATTR_VALID
        r_dimension = AttributeDimension(dim_x = 1, dim_y = 0)
               time = TimeVal(tv_nsec = 0, tv_sec = 1281084943, tv_usec = 461730)
               type = PyTango._PyTango.CmdArgType.DevLong
              value = 239
            w_dim_x = 1
            w_dim_y = 0
        w_dimension = AttributeDimension(dim_x = 1, dim_y = 0)
            w_value = 0]
            
        Spock <homer:10000> [3]: # Read a spectrum attribute
        Spock <pc151:10000> [4]: print tangotest.read_attribute("double_spectrum")
        DeviceAttribute[
        data_format = PyTango._PyTango.AttrDataFormat.SPECTRUM
              dim_x = 20
              dim_y = 0
         has_failed = False
           is_empty = False
               name = 'double_spectrum'
            nb_read = 20
         nb_written = 20
            quality = PyTango._PyTango.AttrQuality.ATTR_VALID
        r_dimension = AttributeDimension(dim_x = 20, dim_y = 0)
               time = TimeVal(tv_nsec = 0, tv_sec = 1281085195, tv_usec = 244760)
               type = PyTango._PyTango.CmdArgType.DevDouble
              value = array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
                11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.])
            w_dim_x = 20
            w_dim_y = 0
        w_dimension = AttributeDimension(dim_x = 20, dim_y = 0)
            w_value = array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
                11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.])]

        Spock <homer:10000> [5]: # Write a scalar attribute
        Spock <homer:10000> [6]: scalar_value = 18
        Spock <homer:10000> [7]: tangotest.write_attribute("long_scalar", scalar_value)

        Spock <homer:10000> [8]: # Write a spectrum attribute
        Spock <homer:10000> [9]: spectrum_value = numpy.random.rand(100)*10
        Spock <homer:10000> [10]: tangotest.write_attribute("double_spectrum", spectrum_value)
        
        
        Spock <homer:10000> [11]: # Write an image attribute
        Spock <homer:10000> [12]: image_value = numpy.random.randint(0,10,size=(10,10))
        Spock <homer:10000> [13]: tangotest.write_attribute("long_image", image_value)

.. tip::
    
    If you are only interested in the attribute's read value you can do insted:
    
    .. sourcecode:: spock
        
            Spock <homer:10000> [1]: tangotest.long_scalar
                         Result [1]: 239
    
    The same is valid for writting a new value to an attribute:
    
    .. sourcecode:: spock
        
            Spock <homer:10000> [1]: tangotest.long_scalar = 18
    
.. note::

    If PyTango is compiled with numpy support the values got when reading
    a spectrum or an image will be numpy arrays. This results in a faster and
    more memory efficient PyTango. You can also use numpy to specify the values when
    writing attributes, especially if you know the exact attribute type.::

        # Creating an unitialized double spectrum of 1000 elements
        spectrum_value = PyTango.numpy_spectrum(PyTango.DevDouble, 1000)

        # Creating an spectrum with a range
        # Note that I do NOT use PyTango.DevLong here, BUT PyTango.NumpyType.DevLong
        # numpy functions do not understand normal python types, so there's a
        # translation available in PyTango.NumpyType
        spectrum_value = numpy.arange(5, 1000, 2, PyTango.NumpyType.DevLong)

        # Creating a 2x2 long image from an existing one
        image_value = PyTango.numpy_image(PyTango.DevLong, [[1,2],[3,4]])

Registering devices
~~~~~~~~~~~~~~~~~~~

Defining devices in the Tango DataBase:

    .. sourcecode:: spock
    
        Spock <homer:10000> [1]: # The 3 devices name we want to create
        Spock <homer:10000> [2]: # Note: these 3 devices will be served by the same DServer
        Spock <homer:10000> [3]: new_device_name1="px1/tdl/mouse1"
        Spock <homer:10000> [4]: new_device_name2="px1/tdl/mouse2"
        Spock <homer:10000> [5]: new_device_name3="px1/tdl/mouse3"

        Spock <homer:10000> [6]: # Define the Tango Class served by this DServer
        Spock <homer:10000> [7]: new_device_info_mouse = PyTango.DbDevInfo()
        Spock <homer:10000> [8]: new_device_info_mouse._class = "Mouse"
        Spock <homer:10000> [9]: new_device_info_mouse.server = "ds_Mouse/server_mouse"

        Spock <homer:10000> [10]: # add the first device
        Spock <homer:10000> [11]: new_device_info_mouse.name = new_device_name1
        Spock <homer:10000> [12]: db.add_device(new_device_info_mouse)

        Spock <homer:10000> [13]: # add the next device
        Spock <homer:10000> [14]: new_device_info_mouse.name = new_device_name2
        Spock <homer:10000> [15]: db.add_device(new_device_info_mouse)

        Spock <homer:10000> [16]: # add the third device
        Spock <homer:10000> [17]: new_device_info_mouse.name = new_device_name3
        Spock <homer:10000> [18]: db.add_device(new_device_info_mouse)

Setting up Device properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A more complex example using python subtilities.
The following python script example (containing some functions and instructions
manipulating a Galil motor axis device server) gives an idea of how the Tango
API should be accessed from Python.

    .. sourcecode:: spock
    
        Spock <homer:10000> [1]: # connecting to the motor axis device
        Spock <homer:10000> [2]: axis1 = DeviceProxy ("microxas/motorisation/galilbox")

        Spock <homer:10000> [3]: # Getting Device Properties
        Spock <homer:10000> [4]: property_names = ["AxisBoxAttachement",
                           ....:                   "AxisEncoderType",
                           ....:                   "AxisNumber",
                           ....:                   "CurrentAcceleration",
                           ....:                   "CurrentAccuracy",
                           ....:                   "CurrentBacklash",
                           ....:                   "CurrentDeceleration",
                           ....:                   "CurrentDirection",
                           ....:                   "CurrentMotionAccuracy",
                           ....:                   "CurrentOvershoot",
                           ....:                   "CurrentRetry",
                           ....:                   "CurrentScale",
                           ....:                   "CurrentSpeed",
                           ....:                   "CurrentVelocity",
                           ....:                   "EncoderMotorRatio",
                           ....:                   "logging_level",
                           ....:                   "logging_target",
                           ....:                   "UserEncoderRatio",
                           ....:                   "UserOffset"]
        
        Spock <homer:10000> [5]: axis_properties = axis1.get_property(property_names)
        Spock <homer:10000> [6]: for prop in axis_properties.keys():
                           ....:     print "%s: %s" % (prop, axis_properties[prop][0])

        Spock <homer:10000> [7]: # Changing Properties
        Spock <homer:10000> [8]: axis_properties["AxisBoxAttachement"] = ["microxas/motorisation/galilbox"]
        Spock <homer:10000> [9]: axis_properties["AxisEncoderType"] = ["1"]
        Spock <homer:10000> [10]: axis_properties["AxisNumber"] = ["6"]
        Spock <homer:10000> [11]: axis1.put_property(axis_properties)

        Spock <homer:10000> [12]: # Reading attributes
        Spock <homer:10000> [13]: att_list = axis.get_attribute_list()
        Spock <homer:10000> [14]: for att in att_list:
                            ....:     att_val = axis.read_attribute(att)
                            ....:     print "%s: %s" % (att.name, att_val.value)

        Spock <homer:10000> [15]: # Changing some attribute values
        Spock <homer:10000> [16]: axis1.write_attribute("AxisBackslash", 0.5)
        Spock <homer:10000> [17]: axis1.write_attribute("AxisDirection", 1.0)
        Spock <homer:10000> [18]: axis1.write_attribute("AxisVelocity", 1000.0)
        Spock <homer:10000> [19]: axis1.write_attribute("AxisOvershoot", 500.0)

        Spock <homer:10000> [20]: # Testing some device commands
        Spock <homer:10000> [21]: pos1=axis1.read_attribute("AxisCurrentPosition")
        Spock <homer:10000> [22]: axis1.command_inout("AxisBackward")
        Spock <homer:10000> [23]: while pos1.value > 1000.0:
                            ....:     pos1 = axis1.read_attribute("AxisCurrentPosition")
                            ....:     print "position axis 1 = ", pos1.value
                            
        Spock <homer:10000> [24]: axis1.command_inout("AxisStop")

A quick tour of Tango device server binding through an example
--------------------------------------------------------------

To write a tango device server in python, you need to import two modules in your script which are:

1. The PyTango module

2. The python sys module provided in the classical python distribution

The following in the python script for a Tango device server with two commands and two attributes. The commands are:

1. IOLOng which receives a Tango Long and return it multiply by 2. This command is allowed only if the device is in the ON state.

2. IOStringArray which receives an array of Tango strings and which returns it but in the reverse order. This command is only allowed if the device is in the ON state.

The attributes are:

1. Long_attr wich is a Tango long attribute, Scalar and Read only with a minimum alarm set to 1000 and a maximum alarm set to 1500

2. Short_attr_rw which is a Tango short attribute, Scalar and Read/Write

The following code is the complete device server code::

    import PyTango
    import sys

    class PyDsExp(PyTango.Device_3Impl):

        def __init__(self,cl,name):
            PyTango.Device_3Impl.__init__(self,cl,name)
            self.debug_stream('In PyDsExp __init__')
            PyDsExp.init_device(self)

        def init_device(self):
            self.debug_stream('In Python init_device method')
            self.set_state(PyTango.DevState.ON)
            self.attr_short_rw = 66
            self.attr_long = 1246

    #------------------------------------------------------------------

        def delete_device(self):
            self.debug_stream('[delete_device] for device %s ' % self.get_name())

    #------------------------------------------------------------------
    # COMMANDS
    #------------------------------------------------------------------

        def is_IOLong_allowed(self):
            return self.get_state() == PyTango.DevState.ON

        def IOLong(self, in_data):
            self.debug_stream('[IOLong::execute] received number %s' % str(in_data))
            in_data = in_data * 2;
            self.debug_stream('[IOLong::execute] return number %s' % str(in_data))
            return in_data;

    #------------------------------------------------------------------

        def is_IOStringArray_allowed(self):
            return self.get_state() == PyTango.DevState.ON

        def IOStringArray(self, in_data):
            l = range(len(in_data)-1, -1, -1);
            out_index=0
            out_data=[]
            for i in l:
                self.debug_stream('[IOStringArray::execute] received String' % in_data[out_index])
                out_data.append(in_data[i])
                self.debug_stream('[IOStringArray::execute] return String %s' %out_data[out_index])
                out_index += 1
            self.y = out_data
            return out_data

    #------------------------------------------------------------------
    # ATTRIBUTES
    #------------------------------------------------------------------

        def read_attr_hardware(self, data):
            self.debug_stream('In read_attr_hardware')

    #------------------------------------------------------------------

        def read_Long_attr(self, the_att):
            self.debug_stream('[PyDsExp::read_attr] attribute name Long_attr')

            # Before PyTango 7.0.0
            #PyTango.set_attribute_value(the_att, self.attr_long)

            # Now:
            the_att.set_value(self.attr_long)

    #------------------------------------------------------------------

        def read_Short_attr_rw(self, the_att):
            self.debug_stream('[PyDsExp::read_attr] attribute name Short_attr_rw')

            # Before PyTango 7.0.0
            #PyTango.set_attribute_value(the_att, self.attr_short_rw)
            
            # Now:
            the_att.set_value(self.attr_short_rw)

    #------------------------------------------------------------------

        def write_Short_attr_rw(self, the_att):
            self.debug_stream('In write_Short_attr_rw for attribute %s' % the_att.get_name())

            # Before PyTango 7.0.0
            #data = []
            #PyTango.get_write_value(the_att, data)

            # Now:
            data = the_att.get_write_value()
            self.attr_short_rw = data[0]

    #------------------------------------------------------------------
    # CLASS
    #------------------------------------------------------------------

    class PyDsExpClass(PyTango.DeviceClass):

        def __init__(self, name):
            PyTango.DeviceClass.__init__(self, name)
            self.set_type("TestDevice")
            print 'In PyDsExpClass __init__'

        cmd_list = { 'IOLong' : [ [ PyTango.ArgType.DevLong, "Number" ],
                                  [ PyTango.ArgType.DevLong, "Number * 2" ] ],
                     'IOStringArray' : [ [ PyTango.ArgType.DevVarStringArray, "Array of string" ],
                                         [ PyTango.ArgType.DevVarStringArray, "This reversed array"] ],
        }

        attr_list = { 'Long_attr' : [ [ PyTango.ArgType.DevLong ,
                                        PyTango.AttrDataFormat.SCALAR ,
                                        PyTango.AttrWriteType.READ],
                                      { 'min alarm' : 1000, 'max alarm' : 1500 } ],

                     'Short_attr_rw' : [ [ PyTango.ArgType.DevShort,
                                           PyTango.AttrDataFormat.SCALAR,
                                           PyTango.AttrWriteType.READ_WRITE ] ]
        }

    if __name__ == '__main__':
        try:
            util = PyTango.Util(sys.argv)
            
            # 
            # Deprecated: util.add_TgClass(PyDsExpClass, PyDsExp, 'PyDsExp')
            util.add_class(PyDsExpClass, PyDsExp, 'PyDsExp')
            
            U = PyTango.Util.instance()
            U.server_init()
            U.server_run()
        except PyTango.DevFailed,e:
            print '-------> Received a DevFailed exception:',e
        except Exception,e:
            print '-------> An unforeseen exception occured....',e

.. _IPython: http://ipython.scipy.org/