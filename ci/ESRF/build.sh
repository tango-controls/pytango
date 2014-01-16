#!/bin/bash
#export CYGWIN="${CYGWIN} nodosfilewarning"
INSTALL_DIR=/segfs/tango/ci/PyTango

if [ ! -z "$NODE_NAME" -a -f "$NODE_NAME" ]
then
	echo $NODE_NAME
	if [ $NODE_NAME != "ct32windows7" ]
	then
        source "$NODE_NAME"
    fi
else
    echo "The settings file for the node $NODE_NAME does not exist!"
    echo "Create ci/$INSTITUTE/$NODE_NAME file."
fi

cd ../..
#used system by default

rm -rf build/*

print $realos
case "${realos}" in
	"debian"*)
		#debian6/7
        /usr/bin/python setup.py build
		;; 
	"redhate"*)
        #redhate4/5
        /segfs/bliss/bin/python2.6 setup.py build
        ;;
	"windows7")
		cd /cygdrive/c
        export PATH=$PATH:/cygdrive/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio\ 9.0/Common7/IDE 
        #export PATH=$PATH:/cygdrive/c/Windows/Microsoft.Net/Framework64/v4.0.30319
		/bin/rm -rf /cygdrive/c/pytango
		LIBNAME="PyTango.vcproj"
		pyVersion="py26 py27"
		platform="Win32 x64"
		SLN="C:\jenkins\workspace\PyTango\OperatingSystems\Windows64-VC10\win\PyTango_VS9\PyTango.sln"
		LIBNAME="PyTango.vcproj"
		for pv in $pyVersion
		do
		    for p in $platform
		    do
		        pyN=${pv:2:4}
			    MODE=$pv"_bopystatic_tangostatic_release|"$p
			    OUTFILE="c:/Temp/log_"${p}"_"${pyN}
                echo "BUILD_"$pv"_"$p
                if [ "$pyN" -eq 33 ]
                then
                    SLN="C:\jenkins\workspace\PyTango\OperatingSystems\Windows64-VC10\win\PyTango_VS10"
					MODE=$pv"_bopystatic_tangostatic_release"
					MAKE_CMD="MSBuild.exe $SLN/PyTango.vcxproj /p:Platform=$p /t:rebuild /p:Configuration=$MODE /v:quiet /flp:LogFile=$OUTFILE;Summary;ShowCommandLine;Verbosity=minimal"
                    #MsBuild.exe $SLN /t:Rebuild /p:Configuration=py33_bopystatic_tangostatic_release /p:Platform=x86 /flp:LogFile=C:/Temp/log_
                else          
			        MAKE_CMD="devenv.exe $SLN /project PyTango.vcproj /rebuild $MODE /projectconfig $MODE /out $OUTFILE"
                fi
			    $MAKE_CMD
		    done
		done
        ;;
	*)
		echo "Build - Not supporting operating system: " ${OSTYPE}
        exit $?
		;;
esac

if [ $? != 0 ]
then
	exit $?
fi

case "${realos}" in
	"debian6_64")
        rm -rf $INSTALL_DIR/debian6/*
		/usr/bin/python setup.py install --prefix=$INSTALL_DIR/debian6
		;; 
	"debian7_64")
        rm -rf $INSTALL_DIR/debian7/*
		/usr/bin/python setup.py install --prefix=$INSTALL_DIR/debian7
		;; 
	"redhate4_32")
        rm -rf $INSTALL_DIR/redhate4/*
		/segfs/bliss/bin/python2.6 setup.py install --prefix=$INSTALL_DIR/redhate4
		;;
	"redhate5_64")
        rm -rf $INSTALL_DIR/redhate5/*
		/segfs/bliss/bin/python2.6 setup.py install --prefix=$INSTALL_DIR/redhate5
		;;
    "windows7"*)
		INSTALL_DIR="//unixhome/segfs/tango/ci/PyTango/windows7"
		/bin/rm -rf $INSTALL_DIR/* || echo "Error executing rm command"
        /bin/cp -rf /cygdrive/c/pytango/build_8.1.2_tg8.1.2_boost1.53.0/lib  $INSTALL_DIR || echo "Error executing cp command, LIB"
		/bin/cp -rf /cygdrive/c/pytango/build_8.1.2_tg8.1.2_boost1.53.0/dist  $INSTALL_DIR || echo "Error executing cp command, DIST"
		/bin/chmod -R 755  $INSTALL_DIR || echo "Error executing chmod command"
        ;;
	*)
		echo "Install - Not supporting operating system: " ${OSTYPE}
        exit $?
		;;
esac


exit $?




