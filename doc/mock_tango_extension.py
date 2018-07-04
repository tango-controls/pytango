"""Mock the tango._tango extension module.

This is useful to build the documentation without building the extension.
However this is a bit tricky since the python side relies on what the
extension exposes. Here is the list of the mocking aspects that require
special attention:

  - __doc__ should not contain the mock documentation
  - __mro__ is required for autodoc
  - __name__ attribute is required
  - Device_6Impl class should not be accessible
  - the __base__ attribute for Device_[X]Impl is required
  - it shoud be possible to set __init__, __getattr__ and __setattr__ methods
  - tango.base_types.__document_enum needs to be patched before it is called
  - tango.base_types.__document_method needs to be patched before it is called
  - the mocks should not have any public methods such as assert_[...]
  - _tango.constants.TgLibVers is required (e.g. '9.2.2')
  - _tango._get_tango_lib_release function is required (e.g. lambda: 922)
  - tango._tango AND tango.constants modules have to be patched
  - autodoc requires a proper inheritance for the device impl classes

Patching tango._tango using sys.modules does not seem to work for python
version older than 3.5 (failed with 2.7 and 3.4)
"""

# Imports
import sys
from mock import MagicMock

__all__ = ('tango',)


# Constants
TANGO_VERSION = '9.2.2'
TANGO_VERSION_INT = int(TANGO_VERSION[::2])


# Extension mock class
class ExtensionMock(MagicMock):

    # Remove the mock documentation
    __doc__ = None

    # The method resolution order is required for autodoc
    __mro__ = object,

    @property
    def __name__(self):
        # __name__ is used for some objects
        if self._mock_name is None:
            return ''
        return self._mock_name.split('.')[-1]

    def __getattr__(self, name):
        # Limit device class discovery
        if name == 'Device_6Impl':
            raise AttributeError
        # Regular mock behavior
        return MagicMock.__getattr__(self, name)

    def __setattr__(self, name, value):
        # Ignore unsupported magic methods
        if name in ["__init__", "__getattr__", "__setattr__",
                    "__str__", "__repr__"]:
            return
        # Hook as soon as possible and patch the documentation methods
        if name == 'side_effect' and self.__name__ == 'AccessControlType':
            import tango.utils
            import tango.base_types
            import tango.device_server
            import tango.connection
            tango.utils.__dict__['document_enum'] = document_enum
            tango.utils.__dict__['document_method'] = document_method
            tango.base_types.__dict__['__document_enum'] = document_enum
            tango.device_server.__dict__['__document_method'] = document_method
            tango.connection.__dict__['__document_method'] = document_method
            tango.connection.__dict__['__document_static_method'] = document_method
        MagicMock.__setattr__(self, name, value)


# Remove all public methods
for name in dir(ExtensionMock):
    if not name.startswith('_') and \
       callable(getattr(ExtensionMock, name)):
        setattr(ExtensionMock, name, None)


# Patched version of document_enum
def document_enum(klass, enum_name, desc, append=True):
    getattr(klass, enum_name).__doc__ = desc


# Patched version of document_enum
def document_method(klass, name, doc, add=True):
    method = lambda self: None
    method.__doc__ = doc
    method.__name__ = name
    setattr(klass, name, method)


# Use empty classes for device impl inheritance scheme
def set_device_implementations(module):
    attrs = {'__module__': module.__name__}
    module.DeviceImpl = type('DeviceImpl', (object,), attrs)
    module.Device_2Impl = type('Device_2Impl', (module.DeviceImpl,), attrs)
    module.Device_3Impl = type('Device_3Impl', (module.Device_2Impl,), attrs)
    module.Device_4Impl = type('Device_4Impl', (module.Device_3Impl,), attrs)
    module.Device_5Impl = type('Device_5Impl', (module.Device_4Impl,), attrs)


# Use empty classes for device proxy inheritance scheme
def set_device_proxy_implementations(module):
    attrs = {'__module__': module.__name__}
    module.Connection = type('Connection', (object,), attrs)
    module.DeviceProxy = type('DeviceProxy', (module.Connection,), attrs)


# Patch the extension module
_tango = ExtensionMock(name='_tango')
_tango.constants.TgLibVers = TANGO_VERSION
_tango._get_tango_lib_release.return_value = TANGO_VERSION_INT
set_device_implementations(_tango)
set_device_proxy_implementations(_tango)


# Patch modules
sys.modules['tango._tango'] = _tango
sys.modules['tango.constants'] = _tango.constants
print('Mocking tango._tango extension module')


# Try to import
import tango
