from random import randint
import sys
import time
import os

exitCode = randint(0,1)

T = time.asctime(time.localtime(time.time()))
testString = 'Failure! ' + T
successMessage = 'FAILED!'

if exitCode == 1:
    testString = 'Success! ' + T
    successMessage = 'SUCCEEDED!'

directory = '/tmp/jenkins/jobs/Test/'

if not os.path.exists(directory):
    os.makedirs(directory)

file = open(directory + 'testfile', 'w')
file.write(testString)
file.close()

print(successMessage)

exit(exitCode)