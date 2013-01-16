import sys
import inspect
import unittest
import TangoRunner

class MyTestSuite1(TangoRunner.TangoTestCase):
    @classmethod
    def setUpClass(self):
        pass
        
    @classmethod
    def tearDownClass(cls):
        pass
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def testMyTestCase1(self):
        pass
    
    def testMyTestCase2__loop(self):
        pass


class MyTestSuite2__loop(TangoRunner.TangoTestCase):
    def testMyTestCase3(self):
        pass
        

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
    TangoRunner.TangoTestRunner().run(suite)