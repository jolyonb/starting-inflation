#!/bin/bash
# Script to set up mount information for the remote instance

# Get the DNS to use
dns=$(./getdns.sh)

# Copy the mount script
scp -i ~/.ssh/aws.pem mount_script.sh ec2-user@$dns:~/

# Run the mount script remotely
ssh -o StrictHostKeyChecking=no -T -i ~/.ssh/aws.pem ec2-user@$dns "chmod +x mount_script.sh && ./mount_script.sh"
