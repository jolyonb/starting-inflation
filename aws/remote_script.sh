#!/bin/bash
# Script to use SSH into the server and run a script remotely
# Specify the script file to use in command line arguments

# Check that an argument was supplied
if [ -z "$1" ]
then
    echo "Run a script on a remote server via SSH"
    echo "Usage: ./remote_script.sh filename"
    exit 0
fi

# Check that the filename exists
if [ ! -f $1 ]
then
    echo "Error: $file not found!"
    exit 1
fi

# Get the DNS to use
dns=$(./getdns.sh)

# Do the ssh
cat $1 | ssh -T -i ~/.ssh/aws.pem ec2-user@$dns
