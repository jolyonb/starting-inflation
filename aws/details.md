# Information for setting up instances

Here are the notes that I made while setting up an instance.


# Launching an instance

Follows https://towardsdatascience.com/cloud-computing-aws-1-introduction-ec2-instance-creation-via-console-5177a2b43359

Go to https://console.aws.amazon.com
Navigate to Services, EC2
Click "Launch Instance"
Choose an OS image. I suggest "Amazon Linux 2 AMI (HVM)", which has a minimal installation.
Select instance type. Only one type is free.
Click Review and Launch
Here, you can specify a security group, if you have one already set up. Otherwise, default works.
Specify the key pair you want to use to SSH into the system. Make a new one if you haven't set one up already. I recommend downloading the associated pem file and storing it in ~/.ssh/aws.pem so you can use it for ssh at later times. You will also need to "chmod 400 aws.pem" to change its permissions before using it.
Click "Launch Instances"
Go to the "Instances" page to see your instance running.
Wait until the Status says "2/2 checks passed".

# Connecting to your instance

Start by setting up the aws command line instructions (awscli):
pip install awscli --upgrade --user
Note the public_dns_name for your instance.
SSH in with the following command:

ssh -i ~/.ssh/aws.pem ec2-user@[public_dns_name]
Type "yes" when prompted.

I need to type:
ssh -i ~/.ssh/aws.pem ec2-user@ec2-184-72-135-250.compute-1.amazonaws.com

Update everything with
sudo yum update

Now, let's see what's installed on this box...
python2.7 is available, but python3 isn't. Let's install it.
sudo yum install python3

This also installs pip3.

# Installing a virtual environment

Navigate to the directory you want to put the virtual environment in (for python3).
python3 -m venv .env

Activate it using
source .env/bin/activate

pip can now be accessed from the virtual environment
pip install numpy scipy matplotlib
pip freeze > requirements.txt


# Installing git

sudo yum install git

Set up details for git:
git config --global user.name "Jolyon Bloomfield"
git config --global user.email "jkb84@cornell.edu"
git config --list

We can now use https to clone from github!


# Attaching an EBS drive

We want to have one drive that contains the code and data. We'll attach it to whichever instance is doing computing. We follow the steps here:

https://devopscube.com/mount-ebs-volume-ec2-instance/

In "Instances", go to "Elastic Block Store", "Volumes".
Click on Create Volume
Choose the appropriate parameters. Make sure that it's in the same zone as the instance though!
Righ click on your volume, and attach it to your instance.

In ssh, see available disks using
lsblk

Check that your device has a file system by running
sudo file -s /dev/xvdf
(Replace xvdf with the appropriate device name)

Make a directory to mount the device in
sudo mkdir /ebs

Mount the drive
sudo mount /dev/xvdf /ebs/
(Replace xvdf with the appropriate device name)

Check the device is working:
cd /ebs
df -h .

Give yourself access to the device:
sudo chown ec2-user /ebs
(should only need to do this the first time it mounts)

To unmount, use
sudo umount /dev/xvdf
(Replace xvdf with the appropriate device name)
Make sure you're not in the directory, or it will complain!


# Terminating your instance

Make sure to go to your instances page, and use actions to terminate your instances, because otherwise they'll run up charges! (Terminating will clear the disk and mean that you can't launch this instance again later; if you just want to pause things, stop them instead, and you can relaunch them later)

Note that we're storing everything on the ebs drive, which is not ephemeral, so nothing will be deleted. Note also that the installation of stuff should be stored on another ebs drive, so long as you created an AMI backed by EBS:
"Basically, root volume (your entire virtual system disk) is ephemeral, but only if you choose to create AMI backed by Amazon EC2 instance store. If you choose to create AMI backed by EBS then your root volume is backed by EBS and everything you have on your root volume will be saved between reboots."


# Copying files into the drive

scp -i <key-pair-file> <data-file> <ec2-username>@<ec2-public-dns>:<ec2-directory>

So:
scp -i ~/.ssh/aws.pem <filename> ec2-user@ec2-184-72-135-250.compute-1.amazonaws.com:/ebs/init/

scp -i ~/.ssh/aws.pem sample.json ec2-user@ec2-184-72-135-250.compute-1.amazonaws.com:/ebs/init/


# Cloning the repo

git clone https://github.com/jolyonb/starting-inflation.git
cd starting-inflation
python3 -m venv .env
source ./.env/bin/activate
pip install -r requirements.txt


# Setting up scripts

Need to make an access key. Go to the IAM console.
Follow the steps here in 1B) and 8.
https://medium.com/@junseopark/from-zero-to-aws-ec2-for-data-science-62e7a22d4579


# Alarms

Seems simple enough to set up an alarm to automatically turn off an instance if CPU usage drops for 5 minutes. Do so in the instance console by clicking on the "alarm bell" add icon for your instance.
