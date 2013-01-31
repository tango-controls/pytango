"""Running tests"""

import sys
import time

from unittest import result
from unittest.signals import registerResult

from unittest import TestCase, TestSuite, suite, util
from copy import deepcopy


__unittest = True

# loop parameters
_loop = 1
_loopSuite = 1
_loopSuffix = '__loop'


def _printDict(obj):
    for key, value in sorted(obj.__dict__.items()):
        print("\t" + str(key) + " : " + str(value))
        
def _hasFailed(result):
    '''Checks if any failure occurred'''
    if result.__class__.__name__ == 'TangoTestResult' and (len(result.errors) != 0 or len(result.failures) != 0):
        return True
    return False

def formatName(name):
    newName = ''
    for letter in name:
        if letter.isupper():
            newName += ' '
        newName += letter
    return newName

class TangoTestSuite(TestSuite):
    '''Tango-tailored Test Suite class'''
    def __init__(self, tests=()):
        super(TangoTestSuite, self).__init__(tests)
        
    def __call__(self, *args, **kwds):
        if len(args) > 0:
            result = args[0]
            if not _hasFailed(result):
                # loop count down
                self.loop = getattr(self,'loop', _loopSuite)
                result.loopSuite = 0
                # flag to indicate if the test suite has finished its loop execution
                result.loopSuiteDone = False
                # determine if the suite consists of test cases for the same class
                suiteClass = next(iter(self), None).__class__
                className = suiteClass.__name__
                if suiteClass != None.__class__ and all(isinstance(test, TangoTestCase) and test.__class__ == suiteClass for test in self):
                    # print test suite name (only once), truncate the '__loop' suffix and show number of iterations
                    if self.loop == _loopSuite:
                        suiteName = formatName(className)
                        if suiteName.endswith(_loopSuffix):
                            suiteName = suiteName[:-len(_loopSuffix)]
                            if _loopSuite > 1:
                                suiteName += ' [' + str(_loopSuite) + ' iter]'
                        result.stream.writeln("\n" + suiteName + "\n")
                    # execute test suites with suffix '__loop' multiple times
                    if className.endswith(_loopSuffix) and self.loop > 1:
                        # treat test methods executed in a loop as one test run
                        testsRun = result.testsRun
                        self.loop -= 1
                        # TODO: check efficiency
                        suite = deepcopy(self)
                        suite(*args, **kwds)
                        result.testsRun = testsRun
                if not _hasFailed(result):
                    result.loopSuite += 1
                    # at the last iteration of the suite loop set the flag to True
                    if not className.endswith(_loopSuffix) or _loopSuite <= 1 or result.loopSuite == _loopSuite:
                        result.loopSuiteDone = True
                    return super(TangoTestSuite, self).run(*args, **kwds)

class TangoTestCase(TestCase):
    '''Tango-tailored Test Case class'''
    def __init__(self, methodName='runTest'):
        super(TangoTestCase, self).__init__(methodName)
        
    def __call__(self, *args, **kwds):
        if len(args) > 0:
            result = args[0]
            if not _hasFailed(result):
                # loop count down
                self.loop = getattr(self,'loop', _loop)
                result.loop = 0
                # print test case name (only once), truncate the '__loop' suffix and show number of iterations
                if self.loop == _loop and result.loopSuiteDone:
                    caseName = formatName(self._testMethodName)
                    if caseName.startswith('test_'):
                        caseName = caseName[len('test_'):]
                    if caseName.endswith(_loopSuffix):
                        caseName = caseName[:-len(_loopSuffix)]
                        if _loop > 1:
                            caseName += ' [' + str(_loop) + ' iter]'
                    caseName = '\t' + caseName
                    result.stream.write(caseName)
                # run test methods with suffix '__loop' multiple times
                if self._testMethodName.endswith(_loopSuffix) and self.loop > 1:
                    # treat test methods executed in a loop as one test run
                    testsRun = result.testsRun
                    self.loop -= 1
                    self(*args, **kwds)
                    result.testsRun = testsRun
                
                if not _hasFailed(result):
                    result.loop += 1
                    returnResult = super(TangoTestCase, self).run(*args, **kwds)
                    # print OK information only after the last successful execution of the test case loop and as well as test suite loop
                    if not _hasFailed(result) and getattr(result, 'loopSuiteDone', False) and (not self._testMethodName.endswith(_loopSuffix)  or _loop <= 1 or result.loop == _loop):
                        result.stream.writeln(" --> OK")
                    return returnResult

class _WritelnDecorator(object):
    """Used to decorate file-like objects with a handy 'writeln' method"""
    def __init__(self,stream):
        self.stream = stream

    def __getattr__(self, attr):
        if attr in ('stream', '__getstate__'):
            raise AttributeError(attr)
        return getattr(self.stream,attr)

    def writeln(self, arg=None):
        if arg:
            self.write(arg)
        self.write('\n') # text-mode streams translate to \r\n if needed


class TangoTestResult(result.TestResult):
    """A test result class that can print formatted text results to a stream.

    Used by TangoTestRunner.
    """
    separator1 = '=' * 70
    separator2 = '-' * 70

    def __init__(self, stream, descriptions, verbosity):
        super(TangoTestResult, self).__init__()
        self.stream = stream
        self.showAll = verbosity > 2
        self.dots = verbosity == 2
        self.tangoPrint = verbosity == 1
        self.descriptions = descriptions
        self.loop = 0
        self.loopSuite = 0

    def getDescription(self, test):
        testString = str(test).split(' ')
        if len(testString) is 2:
            testName = testString[0]
            testClass = testString[1][1:-1]
            if self.loop > 1:
                loop = ' [' + str(self.loop) + ' iter]'
            else:
                loop = ''
            if self.loopSuite > 1:
                loopSuite = ' [' + str(self.loopSuite) + ' iter]'
            else:
                loopSuite = ''
            return str(testClass + loopSuite + ' :: ' + testName + loop)
        else:
            return str(test)
        
        doc_first_line = test.shortDescription()
        if self.descriptions and doc_first_line:
            return '\n'.join((str(test), doc_first_line))
        else:
            return str(test)

    def startTest(self, test):
        super(TangoTestResult, self).startTest(test)
        if self.showAll:
            self.stream.write(self.getDescription(test))
            self.stream.write(" ... ")
            self.stream.flush()

    def addSuccess(self, test):
        super(TangoTestResult, self).addSuccess(test)
        if self.showAll:
            self.stream.writeln("ok")
        elif self.dots:
            self.stream.write('.')
            self.stream.flush()

    def addError(self, test, err):
        super(TangoTestResult, self).addError(test, err)
        if self.showAll:
            self.stream.writeln("ERROR")
        elif self.dots:
            self.stream.write('E')
            self.stream.flush()

    def addFailure(self, test, err):
        super(TangoTestResult, self).addFailure(test, err)
        if self.showAll:
            self.stream.writeln("FAIL")
        elif self.dots:
            self.stream.write('F')
            self.stream.flush()

    def addSkip(self, test, reason):
        super(TangoTestResult, self).addSkip(test, reason)
        if self.showAll:
            self.stream.writeln("skipped {0!r}".format(reason))
        elif self.dots:
            self.stream.write("s")
            self.stream.flush()

    def addExpectedFailure(self, test, err):
        super(TangoTestResult, self).addExpectedFailure(test, err)
        if self.showAll:
            self.stream.writeln("expected failure")
        elif self.dots:
            self.stream.write("x")
            self.stream.flush()

    def addUnexpectedSuccess(self, test):
        super(TangoTestResult, self).addUnexpectedSuccess(test)
        if self.showAll:
            self.stream.writeln("unexpected success")
        elif self.dots:
            self.stream.write("u")
            self.stream.flush()

    def printErrors(self):
        if self.dots or self.showAll:
            self.stream.writeln()
        self.printErrorList('ERROR', self.errors)
        self.printErrorList('FAIL', self.failures)

    def printErrorList(self, flavour, errors):
        for test, err in errors:
            self.stream.writeln()
            self.stream.writeln(self.separator1)
            self.stream.writeln("%s: %s" % (flavour,self.getDescription(test)))
            self.stream.writeln(self.separator2)
            self.stream.writeln("%s" % err)


class TangoTestRunner(object):
    """A test runner class that displays results in textual form.

    It prints out the names of tests as they are run, errors as they
    occur, and a summary of the results at the end of the test run.
    """
    resultclass = TangoTestResult

    def __init__(self, stream=sys.stderr, descriptions=True, verbosity=1,
                 failfast=False, buffer=False, resultclass=None, loopSuite=1, loop=1):
        self.stream = _WritelnDecorator(stream)
        self.descriptions = descriptions
        self.verbosity = verbosity
        self.failfast = failfast
        self.buffer = buffer
        # set loop parameters
        global _loopSuite, _loop
        _loopSuite = loopSuite
        _loop = loop
        if resultclass is not None:
            self.resultclass = resultclass

    def _makeResult(self):
        return self.resultclass(self.stream, self.descriptions, self.verbosity)

    def run(self, test):
        "Run the given test case or test suite."

        # convert test classes to Tango Test Suite compliant
        def convertToTango(test):
            try:
                iter(test)
            except TypeError:
                test.__class__.__bases__ = (TangoTestCase, )
            else:
                test.__class__ = TangoTestSuite
                for t in test:
                    convertToTango(t)
        convertToTango(test)
        
        result = self._makeResult()
        registerResult(result)
        result.failfast = self.failfast
        result.buffer = self.buffer
        startTime = time.time()
        startTestRun = getattr(result, 'startTestRun', None)
        if startTestRun is not None:
            startTestRun()
        try:
            test(result)
        finally:
            stopTestRun = getattr(result, 'stopTestRun', None)
            if stopTestRun is not None:
                stopTestRun()
        stopTime = time.time()
        timeTaken = stopTime - startTime
        result.printErrors()
        if hasattr(result, 'separator2'):
            self.stream.writeln(result.separator2)
        run = result.testsRun
        self.stream.writeln("Ran %d test%s in %.3fs" %
                            (run, run != 1 and "s" or "", timeTaken))
        self.stream.writeln()

        expectedFails = unexpectedSuccesses = skipped = 0
        try:
            results = map(len, (result.expectedFailures,
                                result.unexpectedSuccesses,
                                result.skipped))
        except AttributeError:
            pass
        else:
            expectedFails, unexpectedSuccesses, skipped = results

        infos = []
        if not result.wasSuccessful():
            self.stream.write("FAILED")
            failed, errored = map(len, (result.failures, result.errors))
            if failed:
                infos.append("failures=%d" % failed)
            if errored:
                infos.append("errors=%d" % errored)
        else:
            self.stream.write("OK")
        if skipped:
            infos.append("skipped=%d" % skipped)
        if expectedFails:
            infos.append("expected failures=%d" % expectedFails)
        if unexpectedSuccesses:
            infos.append("unexpected successes=%d" % unexpectedSuccesses)
        if infos:
            self.stream.writeln(" (%s)" % (", ".join(infos),))
        else:
            self.stream.write("\n")
        return result
