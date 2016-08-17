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
- the mocks should not have any public methods such as assert_[...]
- _tango.constants.TgLibVers is required (e.g. '9.2.2')
- _tango._get_tango_lib_release function is required (e.g. lambda: 922)
- tango._tango AND tango.constants modules have to be patched

Patching tango._tango using sys.modules does not seem to work for python
version older than 3.5 (failed with 2.7 and 3.4)
"""

__all__ = ['tango']

# Imports
import sys
from mock import MagicMock

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
        return self._mock_name.split('.')[-1]

    def __getattr__(self, name):
        # Limit device class discovery
        if name == 'Device_6Impl':
            raise AttributeError
        # Emulate device class inheritance
        if name == '__base__':
            return {
                'Device_5Impl': _tango.Device_4Impl,
                'Device_4Impl': _tango.Device_3Impl,
                'Device_3Impl': _tango.Device_2Impl,
                'Device_2Impl': _tango.DeviceImpl,
                'DeviceImpl': object}[self.__name__]
        # Regular mock behavior
        return MagicMock.__getattr__(self, name)

    def __setattr__(self, name, value):
        # Ignore unsupported magic methods
        if name in ["__init__", "__getattr__", "__setattr__"]:
            return
        # Hook in tango.base_types to patch document_enum
        if name == '__getinitargs__' and self.__name__ == 'AttributeInfo':
            import tango.base_types
            tango.base_types.__dict__['__document_enum'] = document_enum
        # Regular mock behavior
        MagicMock.__setattr__(self, name, value)


# Remove all public methods
for name in dir(ExtensionMock):
    if not name.startswith('_') and \
       callable(getattr(ExtensionMock, name)):
        setattr(ExtensionMock, name, None)


# Patched version of document_enum
def document_enum(klass, enum_name, desc, append=True):
    getattr(klass, enum_name).__doc__ = desc


# Patch the extension module
_tango = ExtensionMock(name='_tango')
_tango.constants.TgLibVers = TANGO_VERSION
_tango._get_tango_lib_release.return_value = TANGO_VERSION_INT


# Patch modules
sys.modules['tango._tango'] = _tango
sys.modules['tango.constants'] = _tango.constants
print('Mocking tango._tango extension module')


# Try to import
import tango
