#!/bin/bash

######### BASH TERMINAL COLORS ###############################################
# Black        0;30     Dark Gray     1;30
# Red          0;31     Light Red     1;31
# Green        0;32     Light Green   1;32
# Brown/Orange 0;33     Yellow        1;33
# Blue         0;34     Light Blue    1;34
# Purple       0;35     Light Purple  1;35
# Cyan         0;36     Light Cyan    1;36
# Light Gray   0;37     White         1;37

RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
BROWN='\033[0;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

printf "\n${PURPLE}*** Script ID: $(basename "$0") ***${NC}\n\n"

######### DIRECTORIES & FILES #################################################
BAGS_DIR=${DATA}/bags
DOCKER_IMAGE=sps-project-1
LOG_DIR=None

model_weights_pth=/sps/best_models/420_601.ckpt
######### ENVIRONMENTAL VARIABLES  ############################################
LOCALIZER=hdl
LIDAR_FRAME=os_sensor
TOPIC_ODOM=/odometry_node/odometry
TOPIC_CLOUD=/os_cloud_node/points
PLAY_RATE=1
DURATION=100
NUM_OF_EXP=1

EXP_ID=test

SEQS=(20220420 20220601 20220608 20220629 20220714)

odom_frame=map
filter_method=( sps )

######### LOG DIRECTORY #######################################################
# create log dir
log() { #params are: (1)dir 
    LOG_DIR=$1/log_$2
    mkdir -p $LOG_DIR
    printf "log dir: ${BLUE}${LOG_DIR}${NC}\n"
}

######### Clean up code ######################################################
kill_ros_nodes(){
    rosnode kill -a >/dev/null # kill all the nodes 
    sleep 2
}

# Function to handle the kill signal
cleanup() {
    echo "Received Ctrl+C. Cleaning up before exiting..."
    # Add your cleanup actions here (if any)
    kill_ros_nodes 
    exit 1
}

# Set up the trap to call the cleanup function when the script receives a kill signal
trap cleanup SIGINT

##############################################################################
sps(){
    docker exec ${DOCKER_IMAGE} \
        /bin/bash -c "source /opt/ros/noetic/setup.bash && \
                      source /sps/c_ws/devel/setup.bash && \
                      roslaunch sps_filter sps.launch raw_cloud:=$TOPIC_CLOUD odom_frame:=$odom_frame epsilon:=0.84 model_weights_pth:=$model_weights_pth" &> "$LOG_DIR/sps_log.txt" &
}

mos4d(){ 
    docker exec ${DOCKER_IMAGE} \
        /bin/bash -c "source /opt/ros/noetic/setup.bash && \
                      source /sps/c_ws/devel/setup.bash && \
                      roslaunch mos4d mos4d.launch raw_cloud:=$TOPIC_CLOUD odom_frame:=$odom_frame" &> "$LOG_DIR/mos4d_log.txt" &
}

mapmos(){ 
    docker exec ${DOCKER_IMAGE} \
        /bin/bash -c "source /opt/ros/noetic/setup.bash && \
                      source /sps/c_ws/devel/setup.bash && \
                      roslaunch mapmos mapmos.launch raw_cloud:=$TOPIC_CLOUD odom_frame:=$odom_frame" &> "$LOG_DIR/mapmos_log.txt" &
}

lts(){ 
    docker exec ${DOCKER_IMAGE} \
        /bin/bash -c "source /opt/ros/noetic/setup.bash && \
                      source /sps/c_ws/devel/setup.bash && \
                      roslaunch lts_filter filter.launch raw_cloud:=$TOPIC_CLOUD epsilon_0:=0 epsilon_1:=0.84 lidar:=vlp-16" &> "$LOG_DIR/lts_log.txt" &
}

mask(){
    docker exec ${DOCKER_IMAGE} \
        /bin/bash -c "source /opt/ros/noetic/setup.bash && \
                      source /sps/c_ws/devel/setup.bash && \
                      roslaunch sps_filter mask.launch raw_cloud:=$TOPIC_CLOUD odom_frame:=$odom_frame epsilon:=2" &> "$LOG_DIR/mask_log.txt" &
}

raw(){
    docker exec ${DOCKER_IMAGE} \
        /bin/bash -c "source /opt/ros/noetic/setup.bash && \
                      source /sps/c_ws/devel/setup.bash && \
                      roslaunch sps_filter sps.launch raw_cloud:=$TOPIC_CLOUD odom_frame:=$odom_frame epsilon:=2" &> "$LOG_DIR/sps_log.txt" &
}

##############################################################################
#NOTE: Please update the following paths based on the correct paths on your machine 
hdl(){
    rviz -d /home/ibrahim/SPS/config/rviz/hdl.rviz &>$LOG_DIR/rviz.txt& 
    source /home/ibrahim/Neptune/catkin_ws/devel/setup.bash
    roslaunch hdl_localization hdl_localization.launch odom_child_frame_id:=$LIDAR_FRAME points_topic:=$TOPIC_CLOUD downsample_resolution:=0.2 &> "$LOG_DIR/hdl_log.txt" &
}

##############################################################################
run_exp() {
    local filter="$1"
    local bag_dir_play="$2"
    local bag_dir_record="$3"

    if [[ "$bag_dir_play" == "$bag_dir_record" ]]; then
        echo "${bag_dir_play} and ${bag_dir_record} are equal."
        cleanup
    fi

    printf "${CYAN}Running ${LOCALIZER} in background with $filter filter ... ${NC}\n"
    $LOCALIZER; $filter; sleep 10 # give it time to load the map and initialize the filter

    printf "${CYAN}Recording to $bag_dir_record ${NC}\n"
    rosbag record -O "$bag_dir_record" $TOPIC_ODOM &> "$LOG_DIR/rosbag_record.txt" &
    sleep 1

    printf "${CYAN}Cropping LiDAR points to $MAX_RANGE ${NC}\n"

    printf "${CYAN}Playing $bag_dir_play ${NC}\n"
    rosbag play "$bag_dir_play" --clock -r "$PLAY_RATE" -u "$DURATION"  #>/dev/null
    sleep 5

    kill_ros_nodes
}

calculate_metrics() {
    local gt_traj_pth="$1"
    local exp_dir="$2"
    local exp_num="$3"

    local traj_bag_pth="$exp_dir/$exp_num.bag"
    local odom_traj_pth="$exp_dir/$exp_num.tum"
    local plot_pth="$exp_dir/$exp_num.pdf"
    local res_table_pth="$exp_dir/$exp_num.zip"

    printf "Processing: ${BROWN}${traj_bag_pth}${NC} ...\n"

    evo_traj bag "$traj_bag_pth" "$TOPIC_ODOM" --save_as_tum
    mv *.tum "$odom_traj_pth"

    rm -f "$plot_pth" "$res_table_pth"
    
    evo_ape tum "$gt_traj_pth" "$odom_traj_pth" --plot_mode xy --save_results "$res_table_pth" --save_plot "$plot_pth" --t_max_diff 0.1 #--n_to_align 10 -a
}

######### Main program #######################################################
for exp_num in $(seq 0 "$NUM_OF_EXP"); do
    for seq in "${SEQS[@]}"; do
        gt_traj_pth="$DATA/gt/$seq.tum"
        bag_play_pth="$DATA/bags/$seq.bag"

        for filter in "${filter_method[@]}"; do
            exp_dir="$DATA/$EXP/$LOCALIZER/$seq/${filter}"
            mkdir -p "$exp_dir"
            
            printf "Exp dir: ${BLUE}${exp_dir}${NC}, exp num: ${exp_num}\n"
            log "$exp_dir" "$exp_num"
            
            bag_record_pth="$exp_dir/$exp_num.bag"
            
            run_exp "$filter" "$bag_play_pth" "$bag_record_pth"
            # calculate_metrics "$gt_traj_pth" "$exp_dir" "$exp_num"
            
            echo ""
            echo "----------------------------------------"
            echo "----------------------------------------"
            echo ""
        done
    done
done
