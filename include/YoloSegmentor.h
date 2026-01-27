/**
 * @file YoloSegmentor.h
 * @brief YOLO分割模型推理类
 * 
 * 使用ONNX Runtime加载分割模型，检测标定板区域
 */

#ifndef YOLO_SEGMENTOR_H
#define YOLO_SEGMENTOR_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

/**
 * @brief 分割结果结构
 */
struct SegmentResult {
    cv::Rect roi;           // 掩码边界框（用于裁剪ROI）
    cv::Mat mask;           // 二值掩码（原图尺寸）
    float confidence;       // 置信度
    bool valid;             // 是否检测到目标
    
    SegmentResult() : confidence(0.0f), valid(false) {}
};

/**
 * @brief YOLO分割模型推理类
 */
class YoloSegmentor {
public:
    /**
     * @brief 构造函数
     * @param model_path ONNX模型路径
     * @param conf_threshold 置信度阈值
     * @param target_size 推理尺寸（长边）
     */
    explicit YoloSegmentor(const std::string& model_path, 
                           float conf_threshold = 0.5f,
                           int target_size = 1024);
    
    ~YoloSegmentor() = default;
    
    /**
     * @brief 执行分割推理
     * @param image 输入图像（BGR格式）
     * @return 分割结果（包含ROI和掩码）
     */
    SegmentResult segment(const cv::Mat& image);
    
    /**
     * @brief 检查模型是否加载成功
     */
    bool isLoaded() const { return model_loaded_; }
    
    /**
     * @brief 设置置信度阈值
     */
    void setConfThreshold(float threshold) { conf_threshold_ = threshold; }

private:
    // ONNX Runtime组件
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    
    // 配置
    float conf_threshold_;
    int target_size_;
    bool model_loaded_;
    
    // 预处理（保持纵横比）
    cv::Mat preprocess(const cv::Mat& image, cv::Size& original_size, 
                       float& scale, int& pad_w, int& pad_h,
                       int& input_w, int& input_h);
    
    // 后处理
    SegmentResult postprocess(const std::vector<Ort::Value>& outputs,
                              const cv::Size& original_size,
                              float scale, int pad_w, int pad_h,
                              int input_w, int input_h);
};

#endif // YOLO_SEGMENTOR_H
