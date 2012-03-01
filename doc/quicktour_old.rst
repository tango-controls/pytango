.. _quick-tour-old:

A quick tour (original)
-----------------------

This quick tour will guide you through the first steps on using PyTango.
This is the original quick tour guide that uses a simple Python console.
There is a new version of this document which uses :ref:`itango` console in its
examples. You can find this new version :ref:`here <quick-tour>`.

Check PyTango version
~~~~~~~~~~~~~~~~~~~~~

Start a python console and type:

    >>> import PyTango
    >>> PyTango.__version__
    '7.1.2'
    >>> PyTango.__version_long__
    '7.1.2dev0'
    >>> PyTango.__version_number__
    712
    >>> PyTango.__version_description__
    'This version implements the C++ Tango 7.1 API.'

or alternatively:

    >>> import PyTango
    >>> PyTango.Release.version
    '7.1.2'
    >>> PyTango.Release.version_long
    '7.1.2dev0'
    >>> PyTango.Release.version_number
    712
    >>> PyTango.Release.version_description
    'This version implements the C++ Tango 7.1 API.'

Check Tango C++ version
~~~~~~~~~~~~~~~~~~~~~~~

From a client (This is only possible since PyTango 7.0.0)

    >>> import PyTango.constants
    >>> PyTango.constants.TgLibVers
    '7.1.1'
    
From a server you can alternatively do::
    
    u = PyTango.Util.instance()
    tg_cpp_lib_ver = u.get_tango_lib_release()
    

Test the connection to the Device and get it's current state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the most basic examples is to get a reference to a device and
determine if it is running or not::

    from PyTango import *
    import sys, os, time

    # Protect the script from Exceptions
    try:
            # Get proxy on the tangotest1 device
            print "Getting DeviceProxy "
            tangotest = DeviceProxy("tango/tangotest/1")

            # ping it
            print tangotest.ping()
            
            # get the state
            print tangotest.state()
            
            # First use the classical command_inout way to execute the DevString command
            # (DevString in this case is a command of the TangoTest device)

            result= tangotest.command_inout("DevString", "First hello to device")
            print "Result of execution of DevString command=", result

            # the same with a Device specific command
            result= tangotest.DevString("Second Hello to device")
            print "Result of execution of DevString command=", result

            # Please note that argin argument type is automagically managed by python
            result= tangotest.DevULong(12456)
            print "Result of execution of DevULong command=", result

    # Catch Tango and Systems  Exceptions
    except:
            print "Failed with exception !"
            print sys.exc_info()[0]

Execute commands with scalar arguments on a Device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As you can see in the following example, when scalar types are used, the Tango
binding automagically manages the data types, and writing scripts is quite easy::

    from PyTango import *
    import sys, os, time

    tangotest = DeviceProxy("tango/tangotest/1")

    # First use the classical command_inout way to execute the DevString command
    # (DevString in this case is a command of the TangoTest device)

    result= tangotest.command_inout("DevString", "First hello to device")
    print "Result of execution of DevString command=", result

    # the same with a Device specific command
    result= tangotest.DevString("Second Hello to device")
    print "Result of execution of DevString command=", result

    # Please note that argin argument type is automagically managed by python
    result= tangotest.DevULong(12456)
    print "Result of execution of DevULong command=", result

Execute commands with more complex types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case you have to use put your arguments data in the correct python
structures::

    from PyTango import *
    import sys, os, time

    print "Getting DeviceProxy "
    tango_test = DeviceProxy("tango/tangotest/1")
    # The input argument is a DevVarLongStringArray
    # so create the argin variable containing
    # an array of longs and an array of strings
    argin = ([1,2,3], ["Hello", "TangoTest device"])

    result= tango_test.DevVarLongArray(argin)
    print "Result of execution of DevVarLongArray command=", result

Reading and writing attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Basic read/write attribute operations::

    #Read a scalar attribute
    scalar=tangotest.read_attribute("long_scalar")

    #Read a spectrum attribute
    spectrum=tangotest.read_attribute("double_spectrum")

    # Write a scalar attribute
    scalar_value = 18
    tangotest.write_attribute("long_scalar", scalar_value)

    # Write a spectrum attribute
    spectrum_value = [1.2, 3.2, 12.3]
    tangotest.write_attribute("double_spectrum", spectrum_value)

    # Write an image attribute
    image_value = [ [1, 2], [3, 4] ]
    tangotest.write_attribute("long_image", image_value)


Note that if PyTango is compiled with numpy support the values got when reading
a spectrum or an image will be numpy arrays. This results in a faster and
more memory efficient PyTango. You can also use numpy to specify the values when
writing attributes, especially if you know the exact attribute type.::

    import PyTango, numpy

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

Defining devices in the Tango DataBase::

    from PyTango import *
    import sys, os, time

    #  A reference on the DataBase
    db = Database()

    # The 3 devices name we want to create
    # Note: these 3 devices will be served by the same DServer
    new_device_name1="px1/tdl/mouse1"
    new_device_name2="px1/tdl/mouse2"
    new_device_name3="px1/tdl/mouse3"

    # Define the Tango Class served by this  DServer
    new_device_info_mouse = DbDevInfo()
    new_device_info_mouse._class = "Mouse"
    new_device_info_mouse.server = "ds_Mouse/server_mouse"

    # add the first device
    print "Creation Device:" , new_device_name1
    new_device_info_mouse.name = new_device_name1
    db.add_device(new_device_info_mouse)

    # add the next device
    print "Creation Device:" , new_device_name2
    new_device_info_mouse.name = new_device_name2
    db.add_device(new_device_info_mouse)
    # add the third device
    print "Creation Device:" , new_device_name3
    new_device_info_mouse.name = new_device_name3
    db.add_device(new_device_info_mouse)


Setting up Device properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A more complex example using python subtilities.
The following python script example (containing some functions and instructions
manipulating a Galil motor axis device server) gives an idea of how the Tango
API should be accessed from Python::

    from PyTango import *
    import sys, os, time

    # connecting to the motor axis device
    axis1 = DeviceProxy ("microxas/motorisation/galilbox")

    # Getting Device Properties
    property_names = ["AxisBoxAttachement",
                      "AxisEncoderType",
                      "AxisNumber",
                      "CurrentAcceleration",
                      "CurrentAccuracy",
                      "CurrentBacklash",
                      "CurrentDeceleration",
                      "CurrentDirection",
                      "CurrentMotionAccuracy",
                      "CurrentOvershoot",
                      "CurrentRetry",
                      "CurrentScale",
                      "CurrentSpeed",
                      "CurrentVelocity",
                      "EncoderMotorRatio",
                      "logging_level",
                      "logging_target",
                      "UserEncoderRatio",
                      "UserOffset"]
    axis_properties = axis1.get_property(property_names)
    for prop in axis_properties.keys():
        print "%s: %s" % (prop, axis_properties[prop][0])

    # Changing Properties
    axis_properties["AxisBoxAttachement"] = ["microxas/motorisation/galilbox"]
    axis_properties["AxisEncoderType"] = ["1"]
    axis_properties["AxisNumber"] = ["6"]
    axis1.put_property(axis_properties)

    # Reading attributes
    att_list = axis.get_attribute_list()
    for att in att_list:
        att_val = axis.read_attribute(att)
        print "%s: %s" % (att, att_val.value)

    # Changing some attribute values
    axis1.write_attribute("AxisBackslash", 0.5)
    axis1.write_attribute("AxisDirection", 1.0)
    axis1.write_attribute("AxisVelocity", 1000.0)
    axis1.write_attribute("AxisOvershoot", 500.0)

    # Testing some device commands
    pos1=axis1.read_attribute("AxisCurrentPosition")
    axis1.command_inout("AxisBackward")
    while pos1.value > 1000.0:
        pos1=axis1.read_attribute("AxisCurrentPosition")
        print "position axis 1 = ",pos1.value
    axis1.command_inout("AxisStop")

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
