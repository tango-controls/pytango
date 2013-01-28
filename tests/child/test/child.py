import sys

directory = '/tmp/jenkins/jobs/Test/'

try:
    testString = open(directory + 'testfile', 'r').read()
except:
    print('File doesnt exist!')
    exit(1)

print(testString)

exit(0)