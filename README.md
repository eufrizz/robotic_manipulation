Requirements:
- Run MuJoCo
- Run on Ufactory Lite6
- Run Dockerised on AGX Orin or MacOS

Should training be adaptable to MPS or CUDA?
Should sim run on Orin for CUDA?

Dependencies for MuJoCo:
- python3
- pytorch
- CUDA
- mujoco

Dependencies for hardware:
- python3
- pytorch
- CUDA
- realsense
- ROS?


Orin setup
1st via USB
Check it shows up with `ls /dev/cu.usbmodem`
https://developer.nvidia.com/embedded/learn/get-started-jetson-agx-orin-devkit

### Share internet
To share network via USB, need RNDIS, not supported by default. 3rd party [Horndis](https://www.joshuawise.com/horndis) is no longer supported/doesn't work on Ventura.
To share over ethernet:
1. Go to System prefs > general > sharing
2. Click (i) next to "Internet sharing" (make sure it is turned off)
3. Configure sharing as desired
4. Done
5. Turn on internet sharing
6. Check the hostname at the bottom of the page (ping Eugenes-MacBook-Air.local)
7. On the jetson, `ping Eugenes-MacBook-Air.local`
8. Use the IP shown as the gateway. `sudo nmcli connection modify Wired\ connection\ 1 ipv4.gateway 192.168.2.1`
9. Keep in mind that the gateway needs to be on the same subnet as the IP address

I set laptop to 192.168.2.1, and Orin to 192.168.2.3

### Wifi
`sudo nmcli device wifi connect <name> password <pw>`

### Setting up X11 forwarding:
On mac, install xquartz: `brew install --cask xquartz`\
`ssh -X ...`\
Check `X11Forwarding yes` in `/etc/ssh/sshd_config`\
Should work as normal user now\
To enable running as root, on server:\
`xauth list $DISPLAY`\
Copy this output\
`sudo touch /root/.Xauthority`\
Paste here:\
`sudo xauth add <paste>`\
e.g.\
`sudo xauth add orin/unix:10  MIT-MAGIC-COOKIE-1  00fbc134e596dbc6404a98632a0dc44a`
Or as a one liner: `sudo xauth add $(xauth list $DISPLAY)`
Seems this needs to be done on every new login

### Port forwarding
To access the arm web interface through the connected Xavier, we can port forward over ssh.
`ssh -L local_socket:host:hostport username@host`\
i.e.\
`ssh -L 18333:192.168.1.185:18333 eugene@192.168.4.176`\

### NVPMODEL
Select power modes. Predefined choices are MAXN, 15W, 30W (default), 50W.
They can be chosen with `sudo nvpmodel -m <0/1/2/3>` respectively

I found realsense-viewer and glxgears laggy on 30W mode. Smooth on 50W mode.

### Using bluetooth mouse
Could pair successfully, but could not get to mouse or keyboard to respond.
Tried installing solaar as suggested [here](https://askubuntu.com/questions/1206369/logitech-options-on-linux), even getting the latest version from by adding their repo PPA, but no dice.
Also cloned, built and installed logiops, ran with a MX Master 3 config, but to no avail.

Kernel module for HID is not configured, is this why?
```
$ zcat /proc/config.gz | grep 'CONFIG_USB_HIDDEV'
CONFIG_USB_HIDDEV is not set
```

### Realsense
Turns out, only up to L4T 35.1 is supported. Currently 36.2 installed (no kernel patch script)
Can be installed via apt, but doesn't seem to work? Also can't get python bindings without build from source for Jetson.
```
[ WARN:0@0.263] global cap_gstreamer.cpp:2784 handleMessage OpenCV | GStreamer warning: Embedded video playback halted; module source reported: Could not read from resource.
[ WARN:0@0.264] global cap_gstreamer.cpp:1679 open OpenCV | GStreamer warning: unable to start pipeline
[ WARN:0@0.264] global cap_gstreamer.cpp:1164 isPipelinePlaying OpenCV | GStreamer warning: GStreamer: pipeline have not been created
Error opening camera
```

Alternative: [libuvc installation](https://github.com/IntelRealSense/librealsense/blob/master/doc/libuvc_installation.md)
Didn't work. Also says this method is deprecated.

Alternative: RSUSB backend compilation, from [here](https://dev.intelrealsense.com/docs/nvidia-jetson-tx2-installation).
Make sure to add NVCC to path as shown [below](#cudanvcc-path)
```
sudo apt-get install git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev -y
git clone https://github.com/IntelRealSense/librealsense.git
./scripts/setup_udev_rules.sh  
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=true -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=release -DFORCE_RSUSB_BACKEND=true -DBUILD_WITH_CUDA=true && make && sudo make install
```
Thanks to this [GitHub issue ](https://github.com/IntelRealSense/librealsense/issues/12566)

#### Python
Couldn't get python to work.
Had to copy across the pyrealsense*.so
Which I fixed by adding this to the python CMakelists:
```
 install(TARGETS pyrealsense2
    LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
 )
 install(FILES pyrealsense2/__init__.py DESTINATION ${PYTHON_INSTALL_DIR})
```

#### Viewer
Forwarding OpenGL/realsense viewer over SSH doesn't work (not withot the pain of setting up external rendering), so just stick to connecting to a screen. Interestingly, glxgears works though.

Inside the container, I had to run ldconfig to get the librealsense to show up

```
/dev/video3 - Intel_R__RealSense_TM__Depth_Camera_435i_Intel_R__RealSense_TM__Depth_Camera_435i_944123050641
/dev/video1 - Intel_R__RealSense_TM__Depth_Camera_435i_Intel_R__RealSense_TM__Depth_Camera_435i_944123050641
/dev/video2 - Intel_R__RealSense_TM__Depth_Camera_435i_Intel_R__RealSense_TM__Depth_Camera_435i_944123050641
/dev/video0 - Intel_R__RealSense_TM__Depth_Camera_435i_Intel_R__RealSense_TM__Depth_Camera_435i_944123050641
/dev/input/event6 - Intel_R__RealSense_TM__Depth_Camera_435i_Intel_R__RealSense_TM__Depth_Camera_435i_944123050641
/dev/video5 - Intel_R__RealSense_TM__Depth_Camera_435i_Intel_R__RealSense_TM__Depth_Camera_435i_944123050641
/dev/video4 - Intel_R__RealSense_TM__Depth_Camera_435i_Intel_R__RealSense_TM__Depth_Camera_435i_944123050641
```

After a week or two, I came back to it, only to find that it gave this error when running rs-hello-realsense or a C++ test program to capture a frame:
```
RealSense error calling rs2_pipeline_wait_for_frames(pipe:0xaaaac507dba0):
    Frame didn't arrive within 15000
```
However, another brand new realsense worked. I noticed that the other was on fw version 5.12.7.150. I had previously updated it to the latest 5.16.0.1, and it worked fine. Anyway, I downgraded mine to 5.12.7.100 and it started working. What the hell changed?
Reflashing to 5.16.0.1 gave the same issue. The previous version, 5.15.1, worked, so I'll stick with that.

### CUDA/NVCC Path
NVCC was not in the path by default. Add it to path via bashrc (taken from [here](https://forums.developer.nvidia.com/t/nvcc-command-not-found-after-installing-nvidia-jetpack/224536)):
```
# Add CUDA bin directory into $PATH so that NVCC and others tools can be found
export PATH=/usr/local/cuda/bin:$PATH

# Add CUDA lib directory into the list of places for searching dynamic libraries
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### JTOP
To view GPU usage, install jtop with `sudo pip3 install -U jetson-stats`
Run with `jtop` after logout

### SSH forwarding
On client, `ssh-add`
On server, uncomment `AllowAgentForwarding` in /etc/ssh/sshd_config


## ROS
Latest ROS LTS release is Jazzy Jalisco, but dusty hasn't added it to jetson-containers yet

https://github.com/dusty-nv/jetson-containers?tab=readme-ov-file


## Docker
Didn't come preinstalled, so installed it via apt with the [instructions](https://docs.docker.com/engine/install/ubuntu/) on Docker's website
```
Jun 18 22:56:00 orin dockerd[2983]: time="2024-06-18T22:56:00.038445060-07:00" level=info msg="Loading containers: start."
Jun 18 22:56:00 orin dockerd[2983]: time="2024-06-18T22:56:00.343995273-07:00" level=info msg="stopping event stream following graceful shutdown" error="<nil>" module=libcontainerd namespace=moby
Jun 18 22:56:00 orin dockerd[2983]: failed to start daemon: Error initializing network controller: error obtaining controller instance: failed to register "bridge" driver: unable to add return rule in DOCKER>
Jun 18 22:56:00 orin dockerd[2983]:  (exit status 4))
Jun 18 22:56:00 orin dockerd[2983]: time="2024-06-18T22:56:00.344784307-07:00" level=info msg="stopping event stream following graceful shutdown" error="context canceled" module=libcontainerd namespace=plugi>
Jun 18 22:56:00 orin systemd[1]: docker.service: Main process exited, code=exited, status=1/FAILURE
Jun 18 22:56:00 orin systemd[1]: docker.service: Failed with result 'exit-code'.
Jun 18 22:56:00 orin systemd[1]: Failed to start Docker Application Container Engine.
```
Fix from [here](https://forums.developer.nvidia.com/t/docker-gives-error-after-upgrading-ubuntu/283563/9)
was to
```
sudo update-alternatives --set iptables /usr/sbin/iptables-legacy
sudo apt reinstall docker-ce
```

To build images for jetson, use this repo: https://github.com/dusty-nv/jetson-containers/tree/master


Then to fix the classic docker socket problem, `sudo usermod -a -G docker $USER` and log back in

Add "default-runtime": "nvidia" to your /etc/docker/daemon.json configuration file before attempting to build the containers:
```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "default-runtime": "nvidia"
}
```
Then restart the Docker service, or reboot your system before proceeding: `sudo systemctl restart docker`

### SSD
I couldn't be bothered to make the jetson boot on SSD, so I decided to just mount the SSD and place docker data and other heavy things there. Also means I'll make better use of the 64GB eMMC cause otherwise I'd just do everything on SSD.
Format with sudo gnome-disks (over ssh -X)
Check `lsblk` - should be nvme0 or something similar. `lsblk -f` tells you uuid
```
sudo mkdir /mount/ssd
sudo mount /dev/nvme0n1 /media/ssd
sudo chmod 755 /media/ssd
sudo vim /etc/fstab
# Add line: UUID=<uuid>  /media/ssd  ext4  defaults  0  2

# Test the fstab config
sudo umount /dev/nvme0n1
sudo mount -a
```
Make sure no errors are thrown, then you're good to restart and it will be mounted by default

Docker transfer:
```
sudo cp -r /var/lib/docker /media/ssd/docker # to transfer the cache
sudo vim /etc/docker/daemon.json
```
add "data-root": "/mnt/docker"
```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "default-runtime": "nvidia",
    "data-root": "/mnt/docker"
}
```
confirm the changes by looking under `docker info`

### Neovim
Apt version of neovim is v0.6 whilst snap version is v0.10, which allows us to use lazy
So `sudo snap install nvim` (apt remove neovim if you already installed it with apt)

Then follow:
https://www.lazyvim.org/installation

To get rid of question marks, install a [Nerd Font](https://www.nerdfonts.com/font-downloads) on the laptop, and set iTerm to use it in Settings > Profiles > Text > Font

Press space to get a list of commands.
Space > e will open up the neo-tree sidebar showing directory structure

### ROS + Realsense in Docker
Instead of installing ROS, I decided to just pull the image ros-jazzy-vision-opencv.
Mounted librealsense, and could compile, but it was unable to connect with the camera, giving the `Frame didn't arrive within 15000` error, even though it worked fine outside the container. Tested with rs-hello-realsense and a super basic test program compiled from C++, but to no avail.
Pulled the dustynv ros-humble image (iron didn't have a premade L4T 36), and it failed because it couldn't find libusb. Apt search for libusb returned nothing. In hindsight, probably could have got it installed but anyway I moved on...
Cloned the jetson-containers repo and built l4t-r36.2.0-ros_iron-ros-core image (at least we're using later ROS than humble). Jazzy is for Ubuntu 24. There was an issue with numpy 2 which just got released last week breaking the build, but thankfully managed to patch it myself by pip downgrading numpy (https://github.com/dusty-nv/jetson-containers/issues/561#issuecomment-2181686554).
Make sure to use the `opencv:deb` version of OpenCV when building the jetson-containers image

rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys "librealsense2 opencv" -y

Oops! Turns out ros-core is the most minimal, ros-base has useful packages. A lot of ament packages were missing in ros-core ...and they were still missing in ros_base.

```
ros-iron-rosidl-default-generators ros-iron-rosidl-core-generators ros-iron-rosidl-cmake etc.
```
Ended up being way too many packages. The next idea was to use the opencv base image and just install ros from apt. Turns out it's a very easy install. 
TODO: add source /opt/ros/iron/setup.bash to bashrc

To install realsense ROS there were issues.
1. Not all the .so files were available, so had to build librealsense from source in the image
2. apt still broken with opencv - requires apt download <pkg> and dpkg --force-all -i <pkg>
    a. Turns out you can edit /var/lib/dpkg/status manually!

'build-essential', 'cuda:12.2', 'cudnn:8.9', 'python', 'tensorrt', 'numpy', 'opencv:deb', 'cmake', 'ros:iron-ros-core'
Seems that the OpenCV installation fails to have CUDA anyway?


## ROS MacOS
Docker on macOS can't do net=host, you can only expose specified ports. It was really hard to find what ports DDS used (for ROS2 discovery), and I spent a bit of time playing with it but with no result, so I decided to forego docker. EDIT: I also didn't set ROS_DOMAIN_ID, so that could have screwed it up altogether

This meant building ROS, which is the only was to get it on mac and is not very well supported. Didn't end up getting it to work as the builtin_interfaces package would throw the error "could not import generator.py" or something like that. Here were the notes anyway

>This was useful: https://github.com/mawson-rovers/ros2_mac_setup
>Use Python 3.11 in venv
>Make sure to install numpy==1.26.4 (last version before 2)
>PATH had python12 in it
>
>For vcs: `open /Applications/Python\ 3.11/Install\ Certificates.command`
>
> Maybe this can help? https://github.com/dcedyga/ros2docker-mac-network
>
>To try:
>lsof -i to see ports
>VM on mac to install ROS
>Different version of ROS more compatible with Mac?
>Run with screen attached

I then remembered Foxglove! There is a macOS version. Simply run the foxglove_bridge node on the jetson, and connect via the websocket on mac. Works pretty well!

## ROS 2
https://roboticsbackend.com/ros1-vs-ros2-practical-overview/
Need to set ROS_DOMAIN_ID

Realsense publishing:
while loop: 34% CPU, 27Hz
timer: 35% CPU, 25Hz

setting config to 640*480, 30fps: same

colcon build places install and build in the folder where you first build (if setup.bash has not already been sourced)
