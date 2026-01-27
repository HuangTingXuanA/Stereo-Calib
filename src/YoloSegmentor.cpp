/**
 * @file YoloSegmentor.cpp
 * @brief YOLO分割模型推理实现
 * 
 * 使用ONNX Runtime进行YOLOv8-seg分割推理
 */

#include "YoloSegmentor.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>

YoloSegmentor::YoloSegmentor(const std::string& model_path, float conf_threshold, int target_size)
    : conf_threshold_(conf_threshold), target_size_(target_size), model_loaded_(false) {
    
    try {
        // 创建ONNX Runtime环境
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YoloSegmentor");
        
        // 配置会话选项
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // 尝试使用CUDA
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        } catch (...) {
            // CUDA不可用，使用CPU
        }
        
        // 创建会话
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
        
        model_loaded_ = true;
        std::cout << "[YoloSegmentor] 模型加载成功: " << model_path << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "[YoloSegmentor] ONNX Runtime错误: " << e.what() << std::endl;
        model_loaded_ = false;
    }
}

cv::Mat YoloSegmentor::preprocess(const cv::Mat& image, cv::Size& original_size,
                                   float& scale, int& pad_w, int& pad_h,
                                   int& input_w, int& input_h) {
    original_size = image.size();
    
    // 处理灰度图：转换为3通道
    cv::Mat color_img;
    if (image.channels() == 1) {
        cv::cvtColor(image, color_img, cv::COLOR_GRAY2BGR);
    } else {
        color_img = image;
    }
    
    // 计算保持纵横比的缩放因子
    scale = std::min(static_cast<float>(target_size_) / original_size.width, 
                     static_cast<float>(target_size_) / original_size.height);
    
    // 缩放后的尺寸
    int scaled_w = std::round(original_size.width * scale);
    int scaled_h = std::round(original_size.height * scale);
    
    // 缩放图像
    cv::Mat resized;
    cv::resize(color_img, resized, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);
    
    // 计算padding (Minimum Rectangle)
    // 1. 确保长边为 target_size_ (实际上前面的缩放已经保证了这一点)
    // 2. 确保宽高都是stride(32)的倍数
    int stride = 32;
    input_w = (scaled_w + stride - 1) / stride * stride;
    input_h = (scaled_h + stride - 1) / stride * stride;
    
    int dw = input_w - scaled_w;
    int dh = input_h - scaled_h;
    
    pad_w = dw / 2;
    pad_h = dh / 2;
    
    // 添加补边 (顶，底，左，右)
    int top = static_cast<int>(std::round(dh - 0.1) / 2);
    int bottom = static_cast<int>(std::round(dh + 0.1) / 2);
    int left = static_cast<int>(std::round(dw - 0.1) / 2);
    int right = static_cast<int>(std::round(dw + 0.1) / 2);
    
    pad_w = left;
    pad_h = top;

    cv::Mat letterboxed;
    cv::copyMakeBorder(resized, letterboxed, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    // BGR to RGB (OpenCV默认读入是BGR，YOLO训练通常用RGB)
    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    
    // 转换为float并归一化
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);
    
    return float_img;
}

SegmentResult YoloSegmentor::postprocess(const std::vector<Ort::Value>& outputs,
                                          const cv::Size& original_size,
                                          float scale, int pad_w, int pad_h,
                                          int input_w, int input_h) {
    SegmentResult result;
    result.valid = false;
    
    if (outputs.size() < 2) {
        return result;
    }
    
    // 获取检测输出 [1, 300, 38]
    auto& det_output = outputs[0];
    auto det_shape = det_output.GetTensorTypeAndShapeInfo().GetShape();
    const float* det_data = det_output.GetTensorData<float>();
    
    // 获取proto mask [1, 32, mask_h, mask_w]
    auto& proto_output = outputs[1];
    auto proto_shape = proto_output.GetTensorTypeAndShapeInfo().GetShape();
    const float* proto_data = proto_output.GetTensorData<float>();
    
    int num_detections = det_shape[1];
    int det_dim = det_shape[2];
    
    int mask_dim = proto_shape[1];
    int proto_h = proto_shape[2];
    int proto_w = proto_shape[3];
    
    // 找到最高置信度的检测
    float max_conf = 0;
    int best_idx = -1;
    
    int num_candidates = 0;
    float true_max_conf = 0;
    for (int i = 0; i < num_detections; i++) {
        const float* det = det_data + i * det_dim;
        float conf = det[4];
        
        if (conf > true_max_conf) {
            true_max_conf = conf;
        }
        
        if (conf > conf_threshold_) {
            num_candidates++;
            if (conf > max_conf) {
                max_conf = conf;
                best_idx = i;
            }
        }
    }
    
    if (best_idx < 0) {
        std::cout << "[YoloSegmentor] No valid detection. Max conf: " << true_max_conf 
                  << " (Threshold: " << conf_threshold_ << ")" << std::endl;
        return result;
    }
    
    // std::cout << "[YoloSegmentor] Detection success. Conf: " << max_conf << std::endl;
    
    // std::cout << "[Debug] Found " << num_candidates << " candidates. Best conf: " << max_conf << std::endl;
    
    const float* best_det = det_data + best_idx * det_dim;
    
    // 解析边界框 (x1, y1, x2, y2) - 在输入图像坐标系
    float x1 = best_det[0];
    float y1 = best_det[1];
    float x2 = best_det[2];
    float y2 = best_det[3];
    
    // 获取mask系数
    std::vector<float> mask_coef(mask_dim);
    for (int i = 0; i < mask_dim; i++) {
        mask_coef[i] = best_det[6 + i];
    }
    
    // 计算mask: proto @ mask_coef
    // proto_data: [1, 32, proto_h, proto_w]
    cv::Mat mask_logits(proto_h, proto_w, CV_32F, cv::Scalar(0));
    
    for (int h = 0; h < proto_h; h++) {
        for (int w = 0; w < proto_w; w++) {
            float sum = 0;
            for (int k = 0; k < mask_dim; k++) {
                int idx = k * proto_h * proto_w + h * proto_w + w;
                sum += mask_coef[k] * proto_data[idx];
            }
            mask_logits.at<float>(h, w) = sum;
        }
    }
    
    // sigmoid
    cv::Mat mask_sigmoid;
    cv::exp(-mask_logits, mask_sigmoid);
    mask_sigmoid = 1.0 / (1.0 + mask_sigmoid);
    
    // 缩放mask到输入图像尺寸
    cv::Mat mask_input;
    cv::resize(mask_sigmoid, mask_input, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);
    
    // 用边界框裁剪掩码
    int bx1 = std::clamp(static_cast<int>(x1), 0, input_w - 1);
    int by1 = std::clamp(static_cast<int>(y1), 0, input_h - 1);
    int bx2 = std::clamp(static_cast<int>(x2), 0, input_w - 1);
    int by2 = std::clamp(static_cast<int>(y2), 0, input_h - 1);
    
    cv::Mat mask_cropped = cv::Mat::zeros(input_h, input_w, CV_32F);
    if (bx2 > bx1 && by2 > by1) {
        cv::Rect roi(bx1, by1, bx2 - bx1, by2 - by1);
        mask_input(roi).copyTo(mask_cropped(roi));
    }
    
    // 二值化
    cv::Mat mask_binary;
    cv::threshold(mask_cropped, mask_binary, 0.5, 255, cv::THRESH_BINARY);
    mask_binary.convertTo(mask_binary, CV_8U);
    
    // 移除padding并缩放回原图尺寸
    int scaled_w = static_cast<int>(original_size.width * scale);
    int scaled_h = static_cast<int>(original_size.height * scale);
    
    int crop_x = std::min(pad_w, input_w - 1);
    int crop_y = std::min(pad_h, input_h - 1);
    int crop_w = std::min(scaled_w, input_w - crop_x);
    int crop_h = std::min(scaled_h, input_h - crop_y);
    
    if (crop_w <= 0 || crop_h <= 0) {
        return result;
    }
    
    cv::Mat mask_no_pad = mask_binary(cv::Rect(crop_x, crop_y, crop_w, crop_h));
    
    // 缩放到原图尺寸
    cv::Mat mask_original;
    cv::resize(mask_no_pad, mask_original, original_size, 0, 0, cv::INTER_LINEAR);
    cv::threshold(mask_original, mask_original, 127, 255, cv::THRESH_BINARY);
    
    // 计算ROI
    std::vector<cv::Point> points;
    cv::findNonZero(mask_original, points);
    
    if (points.empty()) {
        std::cout << "[Debug] Mask is empty after thresholding." << std::endl;
        return result;
    }
    
    result.roi = cv::boundingRect(points);
    result.mask = mask_original;
    result.confidence = max_conf;
    result.valid = true;
    
    return result;
}

SegmentResult YoloSegmentor::segment(const cv::Mat& image) {
    SegmentResult result;
    result.valid = false;
    
    if (!model_loaded_ || image.empty()) {
        return result;
    }
    
    // 预处理
    cv::Size original_size;
    float scale;
    int pad_w, pad_h, input_w, input_h;
    cv::Mat preprocessed = preprocess(image, original_size, scale, pad_w, pad_h, input_w, input_h);
    
    // 准备输入tensor
    std::vector<int64_t> input_dims = {1, 3, input_h, input_w};
    size_t input_size = 1 * 3 * input_h * input_w;
    
    // HWC float转CHW
    std::vector<float> input_data(input_size);
    int hw = input_h * input_w;
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < input_h; h++) {
            for (int w = 0; w < input_w; w++) {
                input_data[c * hw + h * input_w + w] = 
                    preprocessed.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    
    // 创建输入tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_size,
        input_dims.data(), input_dims.size()
    );
    
    // 运行推理
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0", "output1"};
    
    try {
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 2
        );
        
        return postprocess(outputs, original_size, scale, pad_w, pad_h, input_w, input_h);
        
    } catch (const Ort::Exception& e) {
        std::cerr << "[YoloSegmentor] 推理错误: " << e.what() << std::endl;
        return result;
    }
}
