import sys
import inspect
import unittest
import TangoRunner
import PyTango
from sets import Set
import types
import os


# =====================================================================================================================
# Test suites ---------------------------------------------------------------------------------------------------------

# Test Case example ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# this is a Test Case which aggregates Test Units of a similar domain
# append a '__loop' suffix to the Test Case name to execute it multiple times - this will only be done if
# a '--loopSuite=' command line parameter is defined and has a numeric value
class TestCaseExample__loop(TangoRunner.TangoTestCase):
    @classmethod
    def setUpClass(self):
        # get command line parameters
        # parameters are defined by name, e.g. 'myparam' and provided in the command line as '--myparam=myparamvalue'
        # you can provide description of the parameter but it is optional
        # values of the parameters are returned as strings
        # get_param() defines and returns value of a Mandatory Parameter which has to be provided in the command line,
        # otherwise the execution terminates
        #self.myparam = get_param('myparam','description of what myparam is')
        # get_param_opt() defines and returns value of an Optional Parameter which may but does not have to be provided
        # in the command line
        #self.myparamopt = get_param_opt('loop','number of times the Unit Test suffixed with "__loop" will be executed')
        # to correctly process command line parameters always append this line
        validate_args()
    
    @classmethod
    def tearDownClass(self):
        if is_restore_set(self, 'my_restore_point'):
            # fix here what your unit test could break upon unpredicted termination
            pass
    # this is a Unit Test, to make the framework interpret a method as a Unit Test append the 'test_' suffix to its name 
    def test_MyUnitTest(self):
        # set a restore point if you modify the configuration of the device on which the Test Suite is executed
        # even if the Unit Test terminates unexpectedly, the configuration will be restored if you use
        # is_restore_set() method in tearDownClass()
        restore_set(self, 'my_restore_point')
        # write your test here
        # to get more information about available assertions refer to unittest tutorial
        assert(1 == 1)
        self.assertRegexpMatches('string to contain a word','word')
        # if you bring all the device configuration to defaults, unset the restore point
        restore_unset(self, 'my_restore_point')

    # this Unit Test will be executed several times in a loop if the '--loop=' parameter is defined and has a numeric
    # value; to declare a Unit Test to be executed in a loop append the '__looop' suffix to its name
    def test_MyUnitTest1__loop(self):
        assert(True)

# Attr Misc Test Case ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class AttrMisc(TangoRunner.TangoTestCase):
    @classmethod
    def setUpClass(self):
        # get parameters
        self.device1_name = get_param('device1','device on which the Test Suite executes')
        validate_args()
        
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

# End of Test suites --------------------------------------------------------------------------------------------------
# =====================================================================================================================



# =====================================================================================================================
# Restore points internal functions -----------------------------------------------------------------------------------

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

# End of Restore points internal functions ----------------------------------------------------------------------------
# =====================================================================================================================



# =====================================================================================================================
# Arguments parsing ---------------------------------------------------------------------------------------------------

params = {}
params_opt = {}
args_valid = True

def get_param(param,desc='user defined mandatory parameter'):
    '''Get mandatory parameters'''
    if param not in params:
        params[param] = desc
    return find_param(param)

def get_param_opt(param,desc='user defined mandatory parameter'):
    '''Get mandatory parameters'''
    if param not in params:
        params_opt[param] = desc
    return find_param(param)

def validate_args():
    '''Validate parameters'''
    global args_valid
    if args_valid == False:
        usage = 'Usage: ' + os.path.basename(__file__) + ' '
        params_str = 'Mandatory Parameters:\n'
        params_opt_str = 'Optional Parameters:\n'
        for param in params:
            usage += '--' + param + '= '
            params_str += '\t--' + param + '= - ' + params[param] + '\n'
        for param in params_opt:
            usage += '[--' + param + '=] '
            params_opt_str += '\t--' + param + '= - ' + params_opt[param] + '\n'
        print(usage + '\n')
        if len(params) != 0:
            print(params_str)
        if len(params_opt) != 0:
            print(params_opt_str)
        sys.exit(1)
            
def find_param(param):
    param_full = '--' + param + '='
    for arg in sys.argv:
        if arg[:len(param_full)] == param_full:
            return arg[len(param_full):]
    global args_valid
    args_valid = False
    return ''

def get_param_if_registered(param):
    if param in params:
        return get_param(param)
    elif param in params_opt:
        return get_opt_param(param)
    else:
        return ''

       
# End of Arguments parsing -------------------------------------------------------------------------------------------- 
# =====================================================================================================================



# =====================================================================================================================
# Test suite execution ------------------------------------------------------------------------------------------------
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
    
    # check loop parameters
    def get_loop(param):
        param_full = '--' + param + '='
        for arg in sys.argv:
            if arg[:len(param_full)] == param_full:
                try:
                    loop_value = int(arg[len(param_full):])
                except:
                    loop_value = 1
                return loop_value
        return 1

    loopSuite_param = get_loop('loopSuite')
    loop_param = get_loop('loop')

    # execute the Test Suite
    suite = TangoRunner.TangoTestSuite(suites)
    TangoRunner.TangoTestRunner(loopSuite=loopSuite_param, loop=loop_param).run(suite)

# End of Test suite execution -----------------------------------------------------------------------------------------
# =====================================================================================================================