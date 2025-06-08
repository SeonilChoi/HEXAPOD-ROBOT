sudo xhost +local:root
sudo docker run -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/gentlemonster/HEXAPOD/colcon_ws:/colcon_ws \
  --net=host \
  --privileged \
  --rm \
  --gpus=all \
  --name ros_foxy_hexapod \
  -it ros/foxy:hexapod bash
