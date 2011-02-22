
Group
-----

.. currentmodule:: PyTango


.. GroupElement is the base class of Group, but is not the base of
   anything else. So, I don't include it in the documentation but just
   add its functions into Group by using :inherited-members:

Group class
~~~~~~~~~~~

.. autoclass:: PyTango.Group
    :show-inheritance:
    :inherited-members:
    :members:


GroupReply classes
~~~~~~~~~~~~~~~~~~

Group member functions do not return the same as their DeviceProxy counterparts,
but objects that contain them. This is:
    - *write attribute* family returns PyTango.GroupReplyList
    - *read attribute* family returns PyTango.GroupAttrReplyList
    - *command inout* family returns PyTango.GroupCmdReplyList

The Group*ReplyList objects are just list-like objects containing
:class:`~PyTango.GroupReply`, :class:`~PyTango.GroupAttrReply` and
:class:`~GroupCmdReply` elements that will be described now.

Note also that GroupReply is the base of GroupCmdReply and GroupAttrReply.

.. autoclass:: PyTango.GroupReply
    :members:

.. autoclass:: PyTango.GroupAttrReply
    :show-inheritance:
    :members:

.. autoclass:: GroupCmdReply
    :show-inheritance:
    :members:

