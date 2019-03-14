#!/bin/bash
# Script to start an EC2 instance

# Get the AWS Instance ID
instance=$(<instance.txt)

# Start the instance
aws ec2 start-instances --instance-ids $instance
