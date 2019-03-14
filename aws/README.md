# AWS Commands

This folder contains a number of scripts to assist in running commands on a remote AWS instance.

* `instance.txt`: Text file that stores the instance id of the remote AWS instance. This must be set.
* `getdns.sh`: Gets the DNS address of the instance; used by most other scripts.
* `start_instance.sh`: Starts the instance.
* `stop_instance.sh`: Stops the instance.
* `instance_info.sh`: Prints out information on the instance.
* `mount_script.sh`: This is a script that is intended to be run on the instance. It ensures that the data volume is mounted.
* `setup_instance.sh`: This script copies `mount_script.sh` to the instance and executes it.
* `push_file.sh`: This pushes a file via SCP to the instance.
* `remote-script.sh`: This script allows a specified script (eg, `remote-script`) to be run remotely via SSH.
* `my_ssh.sh`: Starts an SSH session.
* `run_init.sh`: This is the workhorse script. Given a running instance, it ensures that the data volume is mounted, copies a given initialization file to the instance, then executes `run_aws.py` on that file in the background, before returning.
