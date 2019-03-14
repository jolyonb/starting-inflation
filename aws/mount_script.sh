#!/bin/bash
# Ensures that the data volume is mounted
# Note that this is intended to be run remotely!

mountpoint -q /ebs
if [ $? == 0 ]
then
echo "/ebs is a mountpoint"
else
sudo mount /dev/xvdf /ebs/
echo "/ebs has been mounted"
fi
