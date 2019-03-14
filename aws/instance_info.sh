#!/bin/bash
# Script to get EC2 instance info

# Get the AWS Instance ID
instance=$(<instance.txt)

# Get the information
aws ec2 describe-instance-status --instance-ids $instance --output table
