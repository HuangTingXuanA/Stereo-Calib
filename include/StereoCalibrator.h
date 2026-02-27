/**
 * @file StereoCalibrator.h
 * @brief 双目相机标定器类定义
 * 
 * 提供基于圆形标定板的双目相机标定功能
 * 使用YOLO分割模型进行标定板ROI检测
 */

#ifndef STEREO_CALIBRATOR_H
#define STEREO_CALIBRATOR_H

#include "CalibrationTypes.h"
#include "BoardConfig.h"
#include "ellipse_detector.hpp"
#include "YoloSegmentor.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

/**
 * @brief 双目相机标定器类
 * 
 * 使用YOLO分割获取标定板ROI，结合椭圆检测算法提取圆心，
 * 完成张正友标定法的双目相机标定
 */
class StereoCalibrator {
public:
    /**
     * @brief 构造函数（使用配置文件）
     * @param config_path 标定板配置文件路径
     */
    explicit StereoCalibrator(const std::string& config_path);
    
    /**
     * @brief 检查分割器是否可用
     */
    bool hasSegmentor() const { return segmentor_ && segmentor_->isLoaded(); }
    
    /**
     * @brief 从指定目录加载双目图像对
     * @param root_dir 包含 /left 和 /right 子文件夹的根目录
     */
    bool loadImages(const std::string& root_dir);
    
    /**
     * @brief 检测所有图像对中的圆心（使用YOLO分割）
     */
    bool detectCircles();
    
    /**
     * @brief 执行双目标定
     */
    bool calibrate();
    
    /**
     * @brief 保存标定结果到 YAML 文件
     */
    bool saveResults(const std::string& yaml_path);
    
    /**
     * @brief 输出 3D 点云到 TXT 文件
     */
    bool export3DPoints(const std::string& txt_path);
    
    /**
     * @brief 获取标定结果
     */
    CalibrationResult getResult() const { return result_; }
    
    /**
     * @brief 设置调试模式 (输出所有调试图像)
     */
    void setDebugMode(bool debug) { debug_mode_ = debug; }
    
private:
    // ========== 辅助方法 ==========
    
    /** @brief 在单张图像中检测椭圆 */
    std::vector<Ellipse> detectEllipsesInImage(const cv::Mat& image, const cv::Rect& roi = cv::Rect());
    
    /** @brief 从椭圆列表中提取圆心坐标 */
    std::vector<cv::Point2f> extractCenters(const std::vector<Ellipse>& ellipses);
    
    /** @brief 生成所有圆的世界坐标 
     *  @param flatten_z 是否将 Z 设为 0（true: 初始标定用, false: 精确标定用）
     */
    std::vector<cv::Point3f> generateWorldCoordinates(bool flatten_z = true);
    
    /** @brief 计算重投影误差 */
    void computeReprojectionErrors();

    /** @brief 评估几何一致性（第一性原理误差） */
    void evaluateGeometricConsistency();

    /** @brief 查找特征大圆 */
    bool findAnchors(const std::vector<Ellipse>& ellipses, std::vector<Ellipse>& out_anchors, 
                     const std::string& image_name = "", std::string* error_msg = nullptr);
    
    /** @brief 保存带掩码的调试图像 */
    void saveDebugImageWithMask(
        const cv::Mat& image,
        const std::vector<Ellipse>& ellipses,
        const cv::Mat& mask,
        const cv::Rect& roi,
        const std::string& output_path,
        float confidence = -1.0f,
        const std::vector<Ellipse>& anchors = {});
    
    /** @brief 初始化YOLO分割器 */
    bool initSegmentor(const std::string& model_path, float conf_threshold = 0.5f, int target_size = 1024);
    
private:
    // ========== 成员变量 ==========
    
    // 标定板配置
    BoardConfig board_config_;
    
    // YOLO分割器
    std::unique_ptr<YoloSegmentor> segmentor_;
    
    // 图像数据
    std::vector<ImagePairData> image_pairs_;
    cv::Size image_size_;
    
    // 标定结果
    CalibrationResult result_;
    std::vector<cv::Mat> rvecs_left_;
    std::vector<cv::Mat> tvecs_left_;
    std::vector<cv::Mat> rvecs_right_;
    std::vector<cv::Mat> tvecs_right_;
    
    // 调试模式
    bool debug_mode_ = false;
};

#endif // STEREO_CALIBRATOR_H
