FROM nvidia/cuda:12.8.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=x11_display
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV ROS_DISTRO=foxy

RUN apt update && apt install -y \
		locales && \
	locale-gen en_US en_US.UTF-8 && \
	update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
	export LANG=en_US.UTF-8

RUN apt install -y \
		software-properties-common && \
	add-apt-repository universe

RUN apt update && apt install -y \
		curl && \
	curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
	tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt update && apt upgrade -y && \
	apt install -y \
		ros-foxy-desktop \
		ros-foxy-gazebo-ros-pkgs \
		python3-colcon-common-extensions \
		python3-argcomplete

RUN curl -sSL http://get.gazebosim.org | sh

RUN apt install -y \
		ros-foxy-xacro \
		ros-foxy-robot-state-publisher \
		ros-foxy-joint-state-publisher \
		ros-foxy-gazebo-ros2-control \
		ros-foxy-ros2-control \
		ros-foxy-joint-state-broadcaster \
		ros-foxy-joint-trajectory-controller 

RUN apt-get update && \
	apt-get install -y \ 
		vim && \
	echo "set ts=4" > ~/.vimrc && \
	rm -rf /var/lib/apt/lists/*
