#!/bin/bash
# Script to use SSH into the server
# All arguments are passed to the SSH terminal

# Get the DNS to use
dns=$(./getdns.sh)

# Run the SSH command
ssh -i ~/.ssh/aws.pem ec2-user@$dns $*
