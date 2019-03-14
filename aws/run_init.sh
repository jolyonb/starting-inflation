#!/bin/bash
# SCP an initialization file to the server, and then run
# the python script run_aws.py on it
# Usage: ./run_init.sh filename

# Check that an argument was supplied
if [ -z "$1" ]
then
    echo "Init file SCP script"
    echo "Usage: ./run_init.sh filename"
    exit 0
fi

# Check that the filename exists
if [ ! -f $1 ]
then
    echo "Error: $file not found!"
    exit 1
fi

# Get the DNS to use
echo "Getting DNS address..."
dns=$(./getdns.sh)

# Ensure that the drive is mounted
echo "Mounting volume..."
./setup_instance.sh > /dev/null

# scp the appropriate file
echo "Copying initialization file..."
scp -i ~/.ssh/aws.pem $1 ec2-user@$dns:/ebs/init/ > /dev/null

# Run the script using nohup to run it after SSH closes
filename=$(basename $1)
echo "Running SSH..."
ssh -i ~/.ssh/aws.pem ec2-user@$dns "sh -c 'source /ebs/starting-inflation/.env/bin/activate; cd /ebs; nohup ./starting-inflation/run_aws.py ./init/$filename > /dev/null 2>&1 &'"

echo "Done!"
