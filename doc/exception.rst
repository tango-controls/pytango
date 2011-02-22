.. currentmodule:: PyTango

.. highlight:: python
   :linenothreshold: 4

Exception Handling
==================

Exception definition
--------------------

All the exceptions that can be thrown by the underlying Tango C++ API are available
in the PyTango python module. Hence a user can catch one of the following
exceptions:

    - :class:`DevFailed`
    - :class:`ConnectionFailed`
    - :class:`CommunicationFailed`
    - :class:`WrongNameSyntax`
    - :class:`NonDbDevice`
    - :class:`WrongData`
    - :class:`NonSupportedFeature`
    - :class:`AsynCall`
    - :class:`AsynReplyNotArrived`
    - :class:`EventSystemFailed`
    - :class:`NamedDevFailedList`
    - :class:`DeviceUnlocked`

When an exception is caught, the sys.exc_info() function returns a tuple of three
values that give information about the exception that is currently being handled.
The values returned are (type, value, traceback).
Since most functions don't need access to the traceback, the best solution is to
use something like exctype, value = sys.exc_info()[:2] to extract only the exception
type and value. If one of the Tango exceptions is caught, the exctype will be class
name of the exception (DevFailed, .. etc) and the value a tuple of dictionary objects
all of which containing the following kind of key-value pairs:

- **reason**: a string describing the error type (more readable than the associated error code)
- **desc**: a string describing in plain text the reason of the error.
- **origin**: a string giving the name of the (C++ API) method which thrown the exception
- **severity**: one of the strings WARN, ERR, PANIC giving severity level of the error.

::

    #  Protect the script from Exceptions raised by the Tango or python itself
    try:
        # Get proxy on the tangotest1 device
        print "Getting DeviceProxy "
        tangotest = DeviceProxy("tango/tangotest/1")

    #Catch Tango and Systems  Exceptions
    except DevFailed:
        exctype , value = sys.exc_info()[:2]
        print "Failed with exception ! " , exctype
        for err in value:
            print " reason" , err.reason
            print " description" , err.desc
            print " origin" , err.origin
            print " severity" , err.severity

Throwing exception in a device server
-------------------------------------

The C++ Tango::Except class with its most important methods have been wrapped to Python.
Therefore, in a Python device server, you have the following methods to throw, re-throw or
print a Tango::DevFailed exception :

- *throw_exception()* which is a static method
- *re_throw_exception()* which is also a static method
- *print_exception()* which is also a static method

The following code is an example of a command method requesting a command on a sub-device and re-throwing
the exception in case of::

    try:
        dev.command_inout("SubDevCommand")
    except PyTango.DevFailed, e:
        PyTango.Except.re_throw_exception(e,
            "MyClass_CommandFailed",
            "Sub device command SubdevCommand failed",
            "Command()")

:line 2: Send the command to the sub device in a try/catch block
:line 4-6: Re-throw the exception and add a new level of information in the exception stack


Exception API
-------------

.. autoclass:: PyTango.Except
   :show-inheritance:
   :members:
    
    .. staticmethod:: print_exception(exception)
        
        Prints the Tango exception on the standard output

        :param exception: tango exception
        :type exception: :class:`DevFailed`
        :rtype: None


.. autoclass:: PyTango.DevError
   :show-inheritance:
   :members:

.. autoexception:: PyTango.DevFailed
   :show-inheritance:
   :members:

.. autoexception:: PyTango.ConnectionFailed
   :show-inheritance:
    
    This exception is thrown when a problem occurs during the connection 
    establishment between the application and the device. The API is stateless. 
    This means that DeviceProxy constructors filter most of the exception 
    except for cases described in the following table. 
    
    The desc DevError structure field allows a user to get more precise information. These informations are :
    
    **DB_DeviceNotDefined**
        The name of the device not defined in the database 
    **API_CommandFailed** 
        The device and command name 
    **API_CantConnectToDevice** 
        The device name 
    **API_CorbaException** 
        The name of the CORBA exception, its reason, its locality, its completed 
        flag and its minor code 
    **API_CantConnectToDatabase** 
        The database server host and its port number 
    **API_DeviceNotExported** 
        The device name


.. autoexception:: PyTango.CommunicationFailed
   :show-inheritance:
    
    This exception is thrown when a communication problem is detected during 
    the communication between the client application and the device server. It 
    is a two levels Tango::DevError structure. In case of time-out, the DevError
    structures fields are: 

    +-------+--------------------+-------------------------------------------------+----------+
    | Level |      Reason        |                   Desc                          | Severity |
    +=======+====================+=================================================+==========+
    |   0   | API_CorbaException | CORBA exception fields translated into a string |   ERR    |
    +-------+--------------------+-------------------------------------------------+----------+
    |   1   | API_DeviceTimedOut | String with time-out value and device name      |   ERR    |
    +-------+--------------------+-------------------------------------------------+----------+

    For all other communication errors, the DevError structures fields are: 

    +-------+-------------------------+----------------------------------------------------+----------+
    | Level |         Reason          |                     Desc                           | Severity |
    +=======+=========================+====================================================+==========+
    |   0   | API_CorbaException      |   CORBA exception fields translated into a string  |   ERR    |
    +-------+-------------------------+----------------------------------------------------+----------+
    |   1   | API_CommunicationFailed | String with device, method, command/attribute name |   ERR    |
    +-------+-------------------------+----------------------------------------------------+----------+


.. autoexception:: PyTango.WrongNameSyntax
   :show-inheritance:

This exception has only one level of Tango::DevError structure. The possible 
value for the reason field are :

    **API_UnsupportedProtocol**
        This error occurs when trying to build a DeviceProxy or an AttributeProxy 
        instance for a device with an unsupported protocol. Refer to the appendix 
        on device naming syntax to get the list of supported database modifier 
    **API_UnsupportedDBaseModifier**
        This error occurs when trying to build a DeviceProxy or an AttributeProxy 
        instance for a device/attribute with a database modifier unsupported. 
        Refer to the appendix on device naming syntax to get the list of 
        supported database modifier 
    **API_WrongDeviceNameSyntax**
        This error occurs for all the other error in device name syntax. It is 
        thrown by the DeviceProxy class constructor. 
    **API_WrongAttributeNameSyntax**
        This error occurs for all the other error in attribute name syntax. It 
        is thrown by the AttributeProxy class constructor. 
    **API_WrongWildcardUsage**
        This error occurs if there is a bad usage of the wildcard character 

.. autoexception:: PyTango.NonDbDevice
   :show-inheritance:

    This exception has only one level of Tango::DevError structure. The reason 
    field is set to API_NonDatabaseDevice. This exception is thrown by the API 
    when using the DeviceProxy or AttributeProxy class database access for 
    non-database device. 

.. autoexception:: PyTango.WrongData
   :show-inheritance:

    This exception has only one level of Tango::DevError structure. 
    The possible value for the reason field are :

    **API_EmptyDbDatum**
        This error occurs when trying to extract data from an empty DbDatum 
        object 
    **API_IncompatibleArgumentType**
        This error occurs when trying to extract data with a type different 
        than the type used to send the data 
    **API_EmptyDeviceAttribute**
        This error occurs when trying to extract data from an empty 
        DeviceAttribute object 
    **API_IncompatibleAttrArgumentType**
        This error occurs when trying to extract attribute data with a type 
        different than the type used to send the data 
    **API_EmptyDeviceData**
        This error occurs when trying to extract data from an empty DeviceData 
        object 
    **API_IncompatibleCmdArgumentType**
        This error occurs when trying to extract command data with a type 
        different than the type used to send the data 

.. autoexception:: PyTango.NonSupportedFeature
   :show-inheritance:

    This exception is thrown by the API layer when a request to a feature 
    implemented in Tango device interface release n is requested for a device 
    implementing Tango device interface n-x. There is one possible value for 
    the reason field which is API_UnsupportedFeature. 

.. autoexception:: PyTango.AsynCall
   :show-inheritance:

    This exception is thrown by the API layer when a the asynchronous model id
    badly used. This exception has only one level of Tango::DevError structure. 
    The possible value for the reason field are :

    **API_BadAsynPollId**
        This error occurs when using an asynchronous request identifier which is not 
        valid any more. 
    **API_BadAsyn**
        This error occurs when trying to fire callback when no callback has been 
        previously registered 
    **API_BadAsynReqType**
        This error occurs when trying to get result of an asynchronous request with 
        an asynchronous request identifier returned by a non-coherent asynchronous 
        request (For instance, using the asynchronous request identifier returned 
        by a command_inout_asynch() method with a read_attribute_reply() attribute). 

.. autoexception:: PyTango.AsynReplyNotArrived
   :show-inheritance:

    This exception is thrown by the API layer when:

        - a request to get asynchronous reply is made and the reply is not yet arrived
        - a blocking wait with timeout for asynchronous reply is made and the timeout expired.

    There is one possible value for the reason field which is API_AsynReplyNotArrived. 

.. autoexception:: PyTango.EventSystemFailed
   :show-inheritance:

    This exception is thrown by the API layer when subscribing or unsubscribing 
    from an event failed. This exception has only one level of Tango::DevError 
    structure. The possible value for the reason field are :

    **API_NotificationServiceFailed**
        This error occurs when the subscribe_event() method failed trying to 
        access the CORBA notification service 
    **API_EventNotFound**
        This error occurs when you are using an incorrect event_id in the 
        unsubscribe_event() method 
    **API_InvalidArgs**
        This error occurs when NULL pointers are passed to the subscribe or 
        unsubscribe event methods 
    **API_MethodArgument**
        This error occurs when trying to subscribe to an event which has already 
        been subsribed to 
    **API_DSFailedRegisteringEvent**
        This error means that the device server to which the device belongs to 
        failed when it tries to register the event. Most likely, it means that 
        there is no event property defined 
    **API_EventNotFound**
        Occurs when using a wrong event identifier in the unsubscribe_event 
        method 


.. autoexception:: PyTango.DeviceUnlocked
   :show-inheritance:

    This exception is thrown by the API layer when a device locked by the 
    process has been unlocked by an admin client. This exception has two levels 
    of Tango::DevError structure. There is only possible value for the reason 
    field which is

    **API_DeviceUnlocked**
        The device has been unlocked by another client (administration client) 

    The first level is the message reported by the Tango kernel from the server 
    side. The second layer is added by the client API layer with informations on
    which API call generates the exception and device name. 

.. autoexception:: PyTango.NotAllowed
   :show-inheritance:


.. autoexception:: PyTango.NamedDevFailedList
   :show-inheritance:

    This exception is only thrown by the DeviceProxy::write_attributes() 
    method. In this case, it is necessary to have a new class of exception 
    to transfer the error stack for several attribute(s) which failed during 
    the writing. Therefore, this exception class contains for each attributes
    which failed :

        - The name of the attribute
        - Its index in the vector passed as argumen tof the write_attributes() method
        - The error stack


