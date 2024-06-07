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

Couldn't get pytho

Forwarding OpenGL/realsense viewer over SSH doesn't work (not withot the pain of setting up external rendering), so just stick to connecting to a screen. Interestingly, glxgears works though.

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