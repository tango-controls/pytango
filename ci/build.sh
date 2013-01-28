#!/bin/bash

pwd


if [ ! -z "$INSTITUTE" -a -d "$INSTITUTE" -a -f "$INSTITUTE/build.sh" ]
then
	echo "Executing build.sh for $INSTITUTE"
	cd "$INSTITUTE"
	./build.sh
	exit 0
else
	if [ ! -z "$INSTITUTE" ]
	then
		echo "Failed to execute ci/$INSTITUTE/build.sh !"
		echo "Make sure ci/$INSTITUTE/build.sh exists"
	else
		echo "Mr Jenkins needs additional configuration!"
		echo "Go to Jenkins dashboard -> Manage Jenkins -> Global Properties, tick Environment Variables and add a key-value pair: name - INSTITUTE, value - YourInstituteName."
		echo "Check out the project. Go to the 'ci' directory and create a 'YourInstituteName' subdirectory. In the 'YourInstituteName' subdirectory create a 'build.sh' file which will contain the recipe how to build your project. Make the 'build.sh' file executable. Commit changes."
	fi
	exit -1
fi
