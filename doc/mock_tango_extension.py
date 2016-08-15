__all__ = ['tango']

# Imports
import sys
from mock import MagicMock


# Extension mock class
class ExtensionMock(MagicMock):
    __doc__ = None
    __mro__ = ()

    @property
    def __name__(self):
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
_tango.constants.TgLibVers = "9.2.2"
_tango._get_tango_lib_release.return_value = 922


# Patch modules
sys.modules['tango._tango'] = _tango
sys.modules['tango.constants'] = _tango.constants
print('Mocking tango._tango extension module')

# Try to import
import tango
import tango.futures
import tango.gevent
