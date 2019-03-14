#!/bin/bash
# Script to use SCP to push a file to the server
# Usage:
# ./push_file.sh filename [location]
# If no location is specified, ~/ is used

# Check that a filename was supplied
if [ -z "$1" ]
then
    echo "Init file SCP script"
    echo "Usage: ./pushinit.sh filename [location]"
    exit 0
fi

# Check that the filename exists
if [ ! -f $1 ]
then
    echo "Error: $file not found!"
    exit 1
fi

# Check that the filename exists
if [ -z "$2" ]
then
    location="~/"
else
    location=$2
fi

# Get the DNS to use
dns=$(./getdns.sh)

# scp the appropriate file
scp -i ~/.ssh/aws.pem $1 ec2-user@$dns:$location
