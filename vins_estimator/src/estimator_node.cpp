#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"
#include "loop-closure/loop_closure.h"
#include "loop-closure/keyframe.h"
#include "loop-closure/keyframe_database.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

Estimator estimator;
LoopClosure *loop_closure;
KeyFrameDatabase keyframe_database;

std::condition_variable con;
double g_current_time = -1;
double g_latest_time = -1;
int g_sum_of_wait = 0;
int g_frame_cnt = 0;

queue<sensor_msgs::ImuConstPtr> g_imu_buf;
queue<sensor_msgs::PointCloudConstPtr> g_feature_buf;
queue<int> g_optimize_posegraph_buf;
queue<KeyFrame*> g_keyframe_buf;
queue<RetriveData> g_retrive_data_buf;
queue<pair<cv::Mat, double>> g_image_buf;

std::mutex mtx_posegraph_buf;
std::mutex mtx_buf;
std::mutex mtx_state;
std::mutex mtx_i_buf;
std::mutex mtx_loop_drift;
std::mutex mtx_keyframedatabase_resample;
std::mutex mtx_update_visualization;
std::mutex mtx_keyframe_buftion;
std::mutex mtx_retrive_data_buf;

Eigen::Vector3d g_tmp_P;
Eigen::Quaterniond g_tmp_Q;
Eigen::Vector3d g_tmp_V;
Eigen::Vector3d g_tmp_Ba;
Eigen::Vector3d g_tmp_Bg;
Eigen::Vector3d g_acc_0;
Eigen::Vector3d g_gyr_0;

//camera param
camodocal::CameraPtr m_camera;
vector<int> g_erase_index;
std_msgs::Header g_cur_header;
Eigen::Vector3d g_relocalize_t{ Eigen::Vector3d(0, 0, 0) };
Eigen::Matrix3d g_relocalize_r{ Eigen::Matrix3d::Identity() };

void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dt = t - g_latest_time;
    g_latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = g_tmp_Q * (g_acc_0 - g_tmp_Ba - g_tmp_Q.inverse() * estimator.g);
    Eigen::Vector3d un_gyr = 0.5 * (g_gyr_0 + angular_velocity) - g_tmp_Bg;
    g_tmp_Q = g_tmp_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = g_tmp_Q * (linear_acceleration - g_tmp_Ba - g_tmp_Q.inverse() * estimator.g);
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    g_tmp_P = g_tmp_P + dt * g_tmp_V + 0.5 * dt * dt * un_acc;
    g_tmp_V = g_tmp_V + dt * un_acc;

    g_acc_0 = linear_acceleration;
    g_gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    g_latest_time = g_current_time;
    g_tmp_P = g_relocalize_r * estimator.Ps[WINDOW_SIZE] + g_relocalize_t;
    g_tmp_Q = g_relocalize_r * estimator.Rs[WINDOW_SIZE];
    g_tmp_V = estimator.Vs[WINDOW_SIZE];
    g_tmp_Ba = estimator.Bas[WINDOW_SIZE];
    g_tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    g_acc_0 = estimator.acc_0;
    g_gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = g_imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
GetMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
    while (true){
        if (g_imu_buf.empty() || g_feature_buf.empty())
            return measurements;

        if (!(g_imu_buf.back()->header.stamp > g_feature_buf.front()->header.stamp)){
            ROS_WARN("wait for imu, only should happen at the beginning");
            g_sum_of_wait++;
            return measurements;
        }

        if (!(g_imu_buf.front()->header.stamp < g_feature_buf.front()->header.stamp)){
            ROS_WARN("throw img, only should happen at the beginning");
            g_feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = g_feature_buf.front();
        g_feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (g_imu_buf.front()->header.stamp <= img_msg->header.stamp){
            IMUs.emplace_back(g_imu_buf.front());
            g_imu_buf.pop();
        }

        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void ImuCallback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    mtx_buf.lock();
    g_imu_buf.push(imu_msg);
    mtx_buf.unlock();
    con.notify_one();

    {
        std::lock_guard<std::mutex> lg(mtx_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(g_tmp_P, g_tmp_Q, g_tmp_V, header);
    }
}

void RawImageCallback(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImagePtr img_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    //image_pool[img_msg->header.stamp.toNSec()] = img_ptr->image;
    if(LOOP_CLOSURE)
    {
        mtx_i_buf.lock();
        g_image_buf.push(make_pair(img_ptr->image, img_msg->header.stamp.toSec()));
        mtx_i_buf.unlock();
    }
}

void FeatureCallback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    mtx_buf.lock();
    g_feature_buf.push(feature_msg);
    mtx_buf.unlock();
    con.notify_one();
}

void ProcessIMU(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (g_current_time < 0)
        g_current_time = t;
    double dt = t - g_current_time;
    g_current_time = t;

    double ba[]{0.0, 0.0, 0.0};
    double bg[]{0.0, 0.0, 0.0};

    double dx = imu_msg->linear_acceleration.x - ba[0];
    double dy = imu_msg->linear_acceleration.y - ba[1];
    double dz = imu_msg->linear_acceleration.z - ba[2];

    double rx = imu_msg->angular_velocity.x - bg[0];
    double ry = imu_msg->angular_velocity.y - bg[1];
    double rz = imu_msg->angular_velocity.z - bg[2];
    //ROS_DEBUG("IMU %f, dt: %f, acc: %f %f %f, gyr: %f %f %f", t, dt, dx, dy, dz, rx, ry, rz);

    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
}

//thread:loop detection
void ProcessLoopDetection()
{
    if(loop_closure == NULL)
    {
        const char *voc_file = VOC_FILE.c_str();
        TicToc t_load_voc;
        ROS_DEBUG("loop start loop");
        cout << "voc file: " << voc_file << endl;
        loop_closure = new LoopClosure(voc_file, IMAGE_COL, IMAGE_ROW);
        ROS_DEBUG("loop load vocbulary %lf", t_load_voc.toc());
        loop_closure->initCameraModel(CAM_NAMES);
    }

    while(LOOP_CLOSURE)
    {
        KeyFrame* cur_kf = NULL; 
        mtx_keyframe_buftion.lock();
        while(!g_keyframe_buf.empty())
        {
            if(cur_kf!=NULL)
                delete cur_kf;
            cur_kf = g_keyframe_buf.front();
            g_keyframe_buf.pop();
        }
        mtx_keyframe_buftion.unlock();
        if (cur_kf != NULL)
        {
            cur_kf->global_index = g_frame_cnt;
            mtx_keyframedatabase_resample.lock();
            keyframe_database.add(cur_kf);
            mtx_keyframedatabase_resample.unlock();

            cv::Mat current_image;
            current_image = cur_kf->image;   

            bool loop_succ = false;
            int old_index = -1;
            vector<cv::Point2f> cur_pts;
            vector<cv::Point2f> old_pts;
            TicToc t_brief;
            cur_kf->extractBrief(current_image);
            //printf("loop extract %d feature using %lf\n", cur_kf->keypoints.size(), t_brief.toc());
            TicToc t_loopdetect;
            loop_succ = loop_closure->startLoopClosure(cur_kf->keypoints, cur_kf->descriptors, cur_pts, old_pts, old_index);
            double t_loop = t_loopdetect.toc();
            ROS_DEBUG("t_loopdetect %f ms", t_loop);
            if(loop_succ)
            {
                KeyFrame* old_kf = keyframe_database.getKeyframe(old_index);
                if (old_kf == NULL)
                {
                    ROS_WARN("NO such frame in keyframe_database");
                    ROS_BREAK();
                }
                ROS_DEBUG("loop succ %d with %drd image", g_frame_cnt, old_index);
                assert(old_index!=-1);
                
                Vector3d T_w_i_old, PnP_T_old;
                Matrix3d R_w_i_old, PnP_R_old;

                old_kf->getPose(T_w_i_old, R_w_i_old);
                std::vector<cv::Point2f> measurements_old;
                std::vector<cv::Point2f> measurements_old_norm;
                std::vector<cv::Point2f> measurements_cur;
                std::vector<int> features_id_matched;  
                cur_kf->findConnectionWithOldFrame(old_kf, measurements_old, measurements_old_norm, PnP_T_old, PnP_R_old, m_camera);
                measurements_cur = cur_kf->measurements_matched;
                features_id_matched = cur_kf->features_id_matched;
                // send loop info to VINS relocalization
                int loop_fusion = 0;
                if( (int)measurements_old_norm.size() > MIN_LOOP_NUM && g_frame_cnt - old_index > 35 && old_index > 30)
                {

                    Quaterniond PnP_Q_old(PnP_R_old);
                    RetriveData retrive_data;
                    retrive_data.cur_index = cur_kf->global_index;
                    retrive_data.header = cur_kf->header;
                    retrive_data.P_old = T_w_i_old;
                    retrive_data.R_old = R_w_i_old;
                    retrive_data.relative_pose = false;
                    retrive_data.relocalized = false;
                    retrive_data.measurements = measurements_old_norm;
                    retrive_data.features_ids = features_id_matched;
                    retrive_data.loop_pose[0] = PnP_T_old.x();
                    retrive_data.loop_pose[1] = PnP_T_old.y();
                    retrive_data.loop_pose[2] = PnP_T_old.z();
                    retrive_data.loop_pose[3] = PnP_Q_old.x();
                    retrive_data.loop_pose[4] = PnP_Q_old.y();
                    retrive_data.loop_pose[5] = PnP_Q_old.z();
                    retrive_data.loop_pose[6] = PnP_Q_old.w();
                    mtx_retrive_data_buf.lock();
                    g_retrive_data_buf.push(retrive_data);
                    mtx_retrive_data_buf.unlock();
                    cur_kf->detectLoop(old_index);
                    old_kf->is_looped = 1;
                    loop_fusion = 1;

                    mtx_update_visualization.lock();
                    keyframe_database.addLoop(old_index);
                    CameraPoseVisualization* posegraph_visualization = keyframe_database.getPosegraphVisualization();
                    pubPoseGraph(posegraph_visualization, g_cur_header);  
                    mtx_update_visualization.unlock();
                }


                // visualization loop info
                if(0 && loop_fusion)
                {
                    int COL = current_image.cols;
                    //int ROW = current_image.rows;
                    cv::Mat gray_img, loop_match_img;
                    cv::Mat old_img = old_kf->image;
                    cv::hconcat(old_img, current_image, gray_img);
                    cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
                    cv::Mat loop_match_img2;
                    loop_match_img2 = loop_match_img.clone();
                    /*
                    for(int i = 0; i< (int)cur_pts.size(); i++)
                    {
                        cv::Point2f cur_pt = cur_pts[i];
                        cur_pt.x += COL;
                        cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
                    }
                    for(int i = 0; i< (int)old_pts.size(); i++)
                    {
                        cv::circle(loop_match_img, old_pts[i], 5, cv::Scalar(0, 255, 0));
                    }
                    for (int i = 0; i< (int)old_pts.size(); i++)
                    {
                        cv::Point2f cur_pt = cur_pts[i];
                        cur_pt.x += COL ;
                        cv::line(loop_match_img, old_pts[i], cur_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
                    }
                    ostringstream convert;
                    convert << "/home/tony-ws/raw_data/loop_image/"
                            << cur_kf->global_index << "-" 
                            << old_index << "-" << loop_fusion <<".jpg";
                    cv::imwrite( convert.str().c_str(), loop_match_img);
                    */
                    for(int i = 0; i< (int)measurements_cur.size(); i++)
                    {
                        cv::Point2f cur_pt = measurements_cur[i];
                        cur_pt.x += COL;
                        cv::circle(loop_match_img2, cur_pt, 5, cv::Scalar(0, 255, 0));
                    }
                    for(int i = 0; i< (int)measurements_old.size(); i++)
                    {
                        cv::circle(loop_match_img2, measurements_old[i], 5, cv::Scalar(0, 255, 0));
                    }
                    for (int i = 0; i< (int)measurements_old.size(); i++)
                    {
                        cv::Point2f cur_pt = measurements_cur[i];
                        cur_pt.x += COL ;
                        cv::line(loop_match_img2, measurements_old[i], cur_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
                    }

                    ostringstream convert2;
                    convert2 << "/home/tony-ws/raw_data/loop_image/"
                            << cur_kf->global_index << "-" 
                            << old_index << "-" << loop_fusion <<"-2.jpg";
                    cv::imwrite( convert2.str().c_str(), loop_match_img2);
                }
                  
            }
            //release memory
            cur_kf->image.release();
            g_frame_cnt++;

            if (t_loop > 1000 || keyframe_database.size() > MAX_KEYFRAME_NUM)
            {
                mtx_keyframedatabase_resample.lock();
                g_erase_index.clear();
                keyframe_database.downsample(g_erase_index);
                mtx_keyframedatabase_resample.unlock();
                if(!g_erase_index.empty())
                    loop_closure->eraseIndex(g_erase_index);
            }
        }
        std::chrono::milliseconds dura(10);
        std::this_thread::sleep_for(dura);
    }
}

//thread: pose_graph optimization
void ProcessPoseGraph()
{
    while(true)
    {
        mtx_posegraph_buf.lock();
        int index = -1;
        while (!g_optimize_posegraph_buf.empty())
        {
            index = g_optimize_posegraph_buf.front();
            g_optimize_posegraph_buf.pop();
        }
        mtx_posegraph_buf.unlock();
        if(index != -1)
        {
            Vector3d correct_t = Vector3d::Zero();
            Matrix3d correct_r = Matrix3d::Identity();
            TicToc t_posegraph;
            keyframe_database.optimize4DoFLoopPoseGraph(index,
                                                    correct_t,
                                                    correct_r);
            ROS_DEBUG("t_posegraph %f ms", t_posegraph.toc());
            mtx_loop_drift.lock();
            g_relocalize_r = correct_r;
            g_relocalize_t = correct_t;
            mtx_loop_drift.unlock();
            mtx_update_visualization.lock();
            keyframe_database.updateVisualization();
            CameraPoseVisualization* posegraph_visualization = keyframe_database.getPosegraphVisualization();
            mtx_update_visualization.unlock();
            pubOdometry(estimator, g_cur_header, g_relocalize_t, g_relocalize_r);
            pubPoseGraph(posegraph_visualization, g_cur_header); 
            nav_msgs::Path refine_path = keyframe_database.getPath();
            updateLoopPath(refine_path);
        }

        std::chrono::milliseconds dura(5000);
        std::this_thread::sleep_for(dura);
    }
}

// thread: visual-inertial odometry
void ProcessVIO()
{
    while (1){
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(mtx_buf);
        con.wait(lk, [&]{ return (measurements = GetMeasurements()).size() != 0; });
        lk.unlock();

        for (auto &measurement : measurements){
			// 1. 处理IMU
            for (auto &imu_msg : measurement.first)
                ProcessIMU(imu_msg);
			
			// 2. 处理image
            auto img_msg = measurement.second;
            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());
            TicToc t_s;
            map<int, vector<pair<int, Vector3d>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++){
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                ROS_ASSERT(z == 1);
                image[feature_id].emplace_back(camera_id, Vector3d(x, y, z));
            }
            estimator.processImage(image, img_msg->header);

			// start build keyframe database for loop closure
            if(LOOP_CLOSURE){
                // remove previous loop
                vector<RetriveData>::iterator it = estimator.retrive_data_vector.begin();
                for(; it != estimator.retrive_data_vector.end(); ){
                    if ((*it).header < estimator.Headers[0].stamp.toSec())
                        it = estimator.retrive_data_vector.erase(it);
                    else
                        it++;
                }
                
                mtx_retrive_data_buf.lock();
                while(!g_retrive_data_buf.empty()){
                    RetriveData tmp_retrive_data = g_retrive_data_buf.front();
                    g_retrive_data_buf.pop();
                    estimator.retrive_data_vector.push_back(tmp_retrive_data);
                }
                mtx_retrive_data_buf.unlock();
				
                //WINDOW_SIZE - 2 is key frame
                if(estimator.marginalization_flag == 0 && estimator.solver_flag == estimator.NON_LINEAR){   
                    Vector3d vio_T_w_i = estimator.Ps[WINDOW_SIZE - 2];
                    Matrix3d vio_R_w_i = estimator.Rs[WINDOW_SIZE - 2];
                    mtx_i_buf.lock();
                    while(!g_image_buf.empty() && g_image_buf.front().second < estimator.Headers[WINDOW_SIZE - 2].stamp.toSec()){
                        g_image_buf.pop();
                    }
                    mtx_i_buf.unlock();
					
                    //assert(estimator.Headers[WINDOW_SIZE - 1].stamp.toSec() == g_image_buf.front().second);
                    // relative_T   i-1_T_i relative_R  i-1_R_i
                    cv::Mat KeyFrame_image;
                    KeyFrame_image = g_image_buf.front().first;
                    
                    const char *pattern_file = PATTERN_FILE.c_str();
                    Vector3d cur_T;
                    Matrix3d cur_R;
                    cur_T = g_relocalize_r * vio_T_w_i + g_relocalize_t;
                    cur_R = g_relocalize_r * vio_R_w_i;
                    KeyFrame* keyframe = new KeyFrame(estimator.Headers[WINDOW_SIZE - 2].stamp.toSec(), vio_T_w_i, vio_R_w_i, cur_T, cur_R, g_image_buf.front().first, pattern_file);
                    keyframe->setExtrinsic(estimator.tic[0], estimator.ric[0]);
                    keyframe->buildKeyFrameFeatures(estimator, m_camera);
                    mtx_keyframe_buftion.lock();
                    g_keyframe_buf.push(keyframe);
                    mtx_keyframe_buftion.unlock();
                    // update loop info
                    if (!estimator.retrive_data_vector.empty() && estimator.retrive_data_vector[0].relative_pose){
                        if(estimator.Headers[0].stamp.toSec() == estimator.retrive_data_vector[0].header){
                            KeyFrame* cur_kf = keyframe_database.getKeyframe(estimator.retrive_data_vector[0].cur_index);                            
                            if (abs(estimator.retrive_data_vector[0].relative_yaw) > 30.0 || estimator.retrive_data_vector[0].relative_t.norm() > 20.0){
                                ROS_DEBUG("Wrong loop");
                                cur_kf->removeLoop();
                            }else {
                                cur_kf->updateLoopConnection( estimator.retrive_data_vector[0].relative_t, 
                                                              estimator.retrive_data_vector[0].relative_q, 
                                                              estimator.retrive_data_vector[0].relative_yaw);
                                mtx_posegraph_buf.lock();
                                g_optimize_posegraph_buf.push(estimator.retrive_data_vector[0].cur_index);
                                mtx_posegraph_buf.unlock();
                            }
                        }
                    }
                }
            }
            
            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";
            g_cur_header = header;
            mtx_loop_drift.lock();
            if (estimator.relocalize){
                g_relocalize_t = estimator.relocalize_t;
                g_relocalize_r = estimator.relocalize_r;
            }
            pubOdometry(estimator, header, g_relocalize_t, g_relocalize_r);
            pubKeyPoses(estimator, header, g_relocalize_t, g_relocalize_r);
            pubCameraPose(estimator, header, g_relocalize_t, g_relocalize_r);
            pubPointCloud(estimator, header, g_relocalize_t, g_relocalize_r);
            pubTF(estimator, header, g_relocalize_t, g_relocalize_r);
            mtx_loop_drift.unlock();
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        
        mtx_buf.lock();
        mtx_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        mtx_state.unlock();
        mtx_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, ImuCallback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, FeatureCallback);
    ros::Subscriber sub_raw_image = n.subscribe(IMAGE_TOPIC, 2000, RawImageCallback);

	// VIO main thread
    std::thread measurement_process{ ProcessVIO };
	
    std::thread loop_detection, pose_graph;
    if (LOOP_CLOSURE){
        ROS_WARN("LOOP_CLOSURE true");
        loop_detection = std::thread( ProcessLoopDetection );   
        pose_graph = std::thread( ProcessPoseGraph );
        m_camera = CameraFactory::instance()->generateCameraFromYamlFile( CAM_NAMES );
    }
    ros::spin();

    return 0;
}
