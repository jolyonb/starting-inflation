#!/bin/bash
# Script to use SCP to get a file from the server
# Usage:
# ./get_file.sh filename [location]
# If no location is specified, the current directory is used

# Check that a filename was supplied
if [ -z "$1" ]
then
    echo "Get file SCP script"
    echo "Usage: ./get_file.sh filename [location]"
    exit 0
fi

# Figure out the location
if [ -z "$2" ]
then
    location="."
else
    location=$2
fi

# Get the DNS to use
dns=$(./getdns.sh)

# scp the appropriate file
scp -i ~/.ssh/aws.pem ec2-user@$dns:$1 $location
