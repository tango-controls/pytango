import sys
import inspect
import unittest
import TangoRunner
import PyTango
from sets import Set
import types

# restore points
_restore_points = Set()

def restore_hash(cls, name):
    if isinstance(cls, (type, types.ClassType)):
        # the tearDownClass method case
        return cls.__name__ + name
    else:
        # the test methods case
        return cls.__class__.__name__ + name

def restore_set(cls, name):
    _restore_points.add(restore_hash(cls, name))
    
def restore_unset(cls, name):
    # TODO: consider catching exceptions for silent execution
    _restore_points.remove(restore_hash(cls, name))
    
def is_restore_set(cls, name):
    return restore_hash(cls, name) in _restore_points
    

class AttrMisc(TangoRunner.TangoTestCase):
    @classmethod
    def setUpClass(self):
        self.device1_name = 'dev/pytomasz/1'
        self.device1 = PyTango.DeviceProxy(self.device1_name)
    
    @classmethod
    def tearDownClass(self):
        if is_restore_set(self, 'Float_spec_attr_rw'):
            ai = self.device1.get_attribute_config('Float_spec_attr_rw')
            ai.min_value = 'NaN'
            ai.max_value = 'NaN'
            ai.alarms.min_alarm = 'NaN'
            ai.alarms.max_alarm = 'NaN'
            self.device1.set_attribute_config(ai)
        
    def zzztest(self):
        self.device1.state()
        
    def test_GetAttributeConfig(self):
        ai = self.device1.get_attribute_config('Float_spec_attr_rw')
        old_ai = PyTango.AttributeInfoEx(ai)
        assert(ai.min_value == 'Not specified')
        assert(ai.max_value == 'Not specified')
        assert(ai.alarms.min_alarm == 'Not specified')
        assert(ai.alarms.max_alarm == 'Not specified')
        
        ai.min_value = '3.5'
        ai.max_value = '15.5'
        ai.alarms.min_alarm = '4.2'
        ai.alarms.max_alarm = '13.5'
        self.device1.set_attribute_config(ai)
        restore_set(self, 'Float_spec_attr_rw')
        
        new_ai = self.device1.get_attribute_config('Float_spec_attr_rw')
        assert(new_ai.name == ai.name)
        assert(new_ai.data_type == ai.data_type)
        assert(new_ai.data_format == ai.data_format)
        assert(new_ai.max_dim_x == ai.max_dim_x)
        assert(new_ai.max_dim_y == ai.max_dim_y)
        assert(new_ai.writable == ai.writable)
        assert(new_ai.label == ai.label)
        assert(new_ai.description == ai.description)
        assert(new_ai.alarms.min_alarm == ai.alarms.min_alarm)
        assert(new_ai.alarms.max_alarm == ai.alarms.max_alarm)
        assert(new_ai.min_value == ai.min_value)
        assert(new_ai.max_value == ai.max_value)
        
        ai.min_value = 'NaN'
        ai.max_value = 'NaN'
        ai.alarms.min_alarm = 'NaN'
        ai.alarms.max_alarm = 'NaN'
        self.device1.set_attribute_config(ai)
        restore_unset(self, 'Float_spec_attr_rw')
        
        new_ai = self.device1.get_attribute_config('Float_spec_attr_rw')
        assert(new_ai.name == old_ai.name)
        assert(new_ai.data_type == old_ai.data_type)
        assert(new_ai.data_format == old_ai.data_format)
        assert(new_ai.max_dim_x == old_ai.max_dim_x)
        assert(new_ai.max_dim_y == old_ai.max_dim_y)
        assert(new_ai.writable == old_ai.writable)
        assert(new_ai.label == old_ai.label)
        assert(new_ai.description == old_ai.description)
        assert(new_ai.alarms.min_alarm == old_ai.alarms.min_alarm)
        assert(new_ai.alarms.max_alarm == old_ai.alarms.max_alarm)
        assert(new_ai.min_value == old_ai.min_value)
        assert(new_ai.max_value == old_ai.max_value)
        
        new_ai.min_value = '3.5'
        new_ai.max_value = '15.5'
        self.device1.set_attribute_config(new_ai)
        restore_set(self, 'Float_spec_attr_rw')
        
        # TODO: choose one variant
        # variant 1
        with self.assertRaises(PyTango.DevFailed) as cm:
            self.device1.write_attribute('Float_spec_attr_rw',[3.6,3.3,3.7])
        assert(cm.exception.args[0].reason == 'API_WAttrOutsideLimit')
        # variant 2
        self.assertRaisesRegexp(PyTango.DevFailed,'reason = API_WAttrOutsideLimit',self.device1.write_attribute,'Float_spec_attr_rw',[3.6,3.3,3.7])
        
        self.assertRaisesRegexp(PyTango.DevFailed,'reason = API_WAttrOutsideLimit',self.device1.write_attribute,'Float_spec_attr_rw',[17.6])

        new_ai.min_value = 'NaN'
        new_ai.max_value = 'NaN'
        new_ai.alarms.min_alarm = '6.0'
        self.device1.set_attribute_config(new_ai)
        
        state = self.device1.state()
        assert(state == PyTango.DevState.ALARM)
        status = self.device1.status()
        self.assertRegexpMatches(status,'ALARM')
        self.assertRegexpMatches(status,'Float_spec_attr_rw')
        
        da = self.device1.read_attribute('Float_spec_attr_rw')
        assert(da.quality == PyTango.AttrQuality.ATTR_ALARM)
        
        new_ai.alarms.min_alarm = 'NaN'
        self.device1.set_attribute_config(new_ai)
        
        state = self.device1.state()
        assert(state == PyTango.DevState.ON)
        
        da = self.device1.read_attribute('Float_spec_attr_rw')
        assert(da.quality == PyTango.AttrQuality.ATTR_VALID)
        

if __name__ == '__main__':
    # automatically detect tests (alphabetical order)
    suites = []
    testClasses = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    for name, test in testClasses:
        if issubclass(test,TangoRunner.TangoTestCase):
            suites.append(unittest.TestLoader().loadTestsFromTestCase(test))
    
#    # alternatively add test suite names here
#    tests = [MyTest__loop]
#    for test in tests:
#        suites.append(unittest.TestLoader().loadTestsFromTestCase(test))

    suite = TangoRunner.TangoTestSuite(suites)
    TangoRunner.TangoTestRunner(loopSuite=2, loop=2).run(suite)