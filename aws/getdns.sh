#!/bin/bash
# Script to get the server DNS

# Get the AWS Instance ID
instance=$(<instance.txt)
# Grab the instance DNS
aws ec2 describe-instances --instance-ids $instance --query 'Reservations[].Instances[].PublicDnsName' --output text
