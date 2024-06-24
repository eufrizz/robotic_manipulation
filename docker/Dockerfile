FROM ros_iron:l4t-r36.2.0-numpy 

RUN adduser user

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null


RUN apt-get update && apt-get install -y ros-dev-tools ros-iron-ros-base

RUN apt-get update && apt-get install -y libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev  python-is-python3 vim

USER user
WORKDIR /home/user

RUN git clone https://github.com/IntelRealSense/librealsense.git

USER root
RUN  cd librealsense && mkdir build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) -DBUILD_PYTHON_BINDINGS=true -DCMAKE_BUILD_TYPE=RelWithDebInfo -DFORCE_RSUSB_BACKEND=true -DBUILD_WITH_CUDA=true && make -j10  &&  make install

USER user
WORKDIR ros_ws

RUN mkdir src && cd src && git clone --branch 4.55.1 https://github.com/IntelRealSense/realsense-ros.git 

USER root
RUN usermod -aG video user
#RUN . /opt/ros/iron/setup.sh && rosdep init && rosdep update && rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y
#RUN . /opt/ros/iron/setup.sh && rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y


# TODO: fix apt breaking due to custom OpenCV installation. This is a hacky workaround which installs all the needed packages but leaves it in a broken state
#USER user
#RUN 

#USER root
#RUN apt-get download ros-iron-cv-bridge ros-iron-diagnostic-updater ros-iron-image-transport ros-iron-xacro && dpkg --force-all -i ros-iron*

#USER user
#RUN . /opt/ros/iron/setup.sh && colcon build

