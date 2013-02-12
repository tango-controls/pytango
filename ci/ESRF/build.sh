#!/bin/bash

if [ ! -z "$NODE_NAME" -a -f "$NODE_NAME" ]
then
	source "$NODE_NAME"
else
	echo "The settings file for the node $NODE_NAME does not exist!"
	echo "Create ci/$INSTITUTE/$NODE_NAME file."
	exit 1
fi

cd ../..
#used system by default
python setup.py build

if [ $? != 0 ]
then
    /segfs/bliss/bin/python2.6 setup.py build
elif [ $? != 0 ]
then
	exit $?
fi

case "${realos}" in
	"debian6_64")
		python setup.py install --prefix=/tmp/jenkins/jobs/PyTango
		;; 
	"redhate4_32")
		PYTHONPATH=/segfs/bliss/source/python/PyTango/PyTango8/redhate4/lib/python2.6/site-packages/ /segfs/bliss/bin/python2.6 setup.py install --prefix=/tmp/jenkins/jobs/PyTango
		;;
	"redhate5_64")
		PYTHONPATH=/segfs/bliss/source/python/PyTango/PyTango8/redhate5/lib/python2.6/site-packages/ /segfs/bliss/bin/python2.6 setup.py install --prefix=/tmp/jenkins/jobs/PyTango
		;;
	"W7_64")
        ;;
	*)
		echo "Not supporting operating system: " ${OSTYPE}
        exit $?
		;;
esac

exit $?





























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
