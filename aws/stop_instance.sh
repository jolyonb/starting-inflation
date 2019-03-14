#!/bin/bash
# Script to stop an EC2 instance

# Get the AWS Instance ID
instance=$(<instance.txt)

# Stop the instance
aws ec2 stop-instances --instance-ids $instance
