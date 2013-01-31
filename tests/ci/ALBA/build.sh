#!/bin/bash

export PYTHONPATH=/tmp/jenkins/jobs/PyTango

cd ../..

python DevTest.py pytomasz &

if [ $? != 0 ]
then
	exit $?
fi

python TestSuite.py --device1=dev/pytomasz/1

expor EX = $?

ps -ef | grep DevTest.py | awk '{print $2}' | xargs kill -9

exit $EX































echo LOG4TANGO_ROOT
echo $LOG4TANGO_ROOT
echo OMNI_ROOT
echo $OMNI_ROOT
echo BOOST_ROOT
echo $BOOST_ROOT
echo ZMQ_ROOT
echo $ZMQ_ROOT

echo C_INCLUDE_PATH
echo $C_INCLUDE_PATH

echo CPLUS_INCLUDE_PATH
echo $CPLUS_INCLUDE_PATH

echo CPATH
echo $CPATH

export CPATH=/tmp/jenkins/jobs/TangoLib/include
export C_INCLUDE_PATH=/tmp/jenkins/jobs/TangoLib/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/tmp/jenkins/jobs/TangoLib/include:$CPLUS_INCLUDE_PATH

export TANGO_ROOT=/tmp/jenkins/jobs/TangoLib


export LD_LIBRARY_PATH=/tmp/jenkins/jobs/TangoLib/lib:$LD_LIBRARY_PATH


echo $LD_LIBRARY_PATH
echo $CPLUS_INCLUDE_PATH


cd ../..

pwd

make user=1 prefix=/tmp/jenkins/jobs install

make user=1 prefix=/tmp/jenkins/jobs install

export PYTHONPATH=/tmp/jenkins/jobs/PyTango:$PYTHONPATH

echo $PYTHONPATH

python tests/DevTest.py pytomasz &

python tests/TestSuite.py

ps -ef | awk '/DevTest.py/ {print$2}' | xargs kill -9
