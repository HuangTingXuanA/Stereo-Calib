/**
 * @file StereoCalibrator.cpp
 * @brief 双目相机标定器类实现
 */

#include "StereoCalibrator.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <cmath>
#include <chrono>
#include <map>

namespace fs = std::filesystem;

/**
 * @brief 构造函数（使用配置文件）
 */
StereoCalibrator::StereoCalibrator(const std::string& config_path) {
    std::cout << "=== 初始化标定器 ===" << std::endl;
    
    // 加载配置文件
    if (!board_config_.loadFromFile(config_path)) {
        throw std::runtime_error("无法加载配置文件: " + config_path);
    }
    
    // 从配置文件初始化分割器
    if (!board_config_.segmentor.model_path.empty()) {
        if (!initSegmentor(board_config_.segmentor.model_path, 
                           board_config_.segmentor.confidence,
                           board_config_.segmentor.inference_size)) {
            std::cerr << "警告: 无法加载分割模型，将使用全图检测" << std::endl;
        }
    } else {
        std::cerr << "警告: 配置文件未指定分割模型路径" << std::endl;
    }
}

/**

 * @brief 从指定目录加载双目图像对
 */
bool StereoCalibrator::loadImages(const std::string& root_dir) {
    // 1. 构建 left 和 right 目录路径
    std::string left_dir = root_dir + "/left";
    std::string right_dir = root_dir + "/right";
    
    // 检查目录是否存在
    if (!fs::exists(left_dir) || !fs::is_directory(left_dir)) {
        std::cerr << "错误: 左图目录不存在: " << left_dir << std::endl;
        return false;
    }
    if (!fs::exists(right_dir) || !fs::is_directory(right_dir)) {
        std::cerr << "错误: 右图目录不存在: " << right_dir << std::endl;
        return false;
    }
    
    // 2. 读取文件列表
    std::vector<std::string> left_files;
    std::vector<std::string> right_files;
    
    // 支持的图像格式
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    
    // 读取左图文件
    for (const auto& entry : fs::directory_iterator(left_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                left_files.push_back(entry.path().string());
            }
        }
    }
    
    // 读取右图文件
    for (const auto& entry : fs::directory_iterator(right_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                right_files.push_back(entry.path().string());
            }
        }
    }
    
    // 3. 排序确保配对
    std::sort(left_files.begin(), left_files.end());
    std::sort(right_files.begin(), right_files.end());
    
    // 4. 验证数量一致
    if (left_files.size() != right_files.size()) {
        std::cerr << "错误: 左右图像数量不一致 (左: " << left_files.size() 
                  << ", 右: " << right_files.size() << ")" << std::endl;
        return false;
    }
    
    if (left_files.empty()) {
        std::cerr << "错误: 未找到图像文件" << std::endl;
        return false;
    }
    
    // 5. 加载图像对
    for (size_t i = 0; i < left_files.size(); ++i) {
        ImagePairData pair;
        pair.left_path = left_files[i];
        pair.right_path = right_files[i];
        image_pairs_.push_back(pair);
    }
    
    std::cout << "共加载图像 " << left_files.size() << " 对图像" << std::endl;
    return true;
}

/**
 * @brief 检测所有图像对中的圆心（基于YOLO分割的精确检测）
 * 
 * 使用YOLO分割模型获取标定板ROI和掩码，在ROI内检测椭圆，
 * 并使用掩码过滤只保留标定板区域的椭圆
 */
bool StereoCalibrator::detectCircles() {
    int valid_count = 0;
    int expected_circles = board_config_.rows * board_config_.cols;
    
    // 生成世界坐标（所有图像对使用相同的世界坐标）
    std::vector<cv::Point3f> world_coords = generateWorldCoordinates();
    
    // 创建调试目录
    std::filesystem::create_directories("debug_img");
    
    // 检查分割器是否可用
    if (!segmentor_ || !segmentor_->isLoaded()) {
        std::cerr << "警告: YOLO分割器未初始化，将使用全图检测" << std::endl;
    }
    
    for (size_t i = 0; i < image_pairs_.size(); ++i) {
        // 提取图像名称
        std::string left_filename = std::filesystem::path(image_pairs_[i].left_path).stem().string();
        std::string right_filename = std::filesystem::path(image_pairs_[i].right_path).stem().string();
        
        // 读取图像
        cv::Mat left_img = cv::imread(image_pairs_[i].left_path, cv::IMREAD_GRAYSCALE);
        cv::Mat right_img = cv::imread(image_pairs_[i].right_path, cv::IMREAD_GRAYSCALE);
        
        if (left_img.empty() || right_img.empty()) {
            std::cout << "图像对 " << left_filename << "/" << right_filename << ": 加载失败 [跳过]" << std::endl;
            continue;
        }
        
        // 保存图像尺寸
        if (image_size_.width == 0) {
            image_size_ = left_img.size();
        }
        
        // ========== 处理左图 ==========
        std::vector<Ellipse> left_ellipses;
        cv::Rect left_roi;
        cv::Mat left_mask;
        float left_conf = -1.0f;
        bool left_has_mask = false;
        

        if (segmentor_ && segmentor_->isLoaded()) {
            SegmentResult left_seg = segmentor_->segment(left_img);
            if (left_seg.valid) {
                left_roi = left_seg.roi;
                left_mask = left_seg.mask;
                left_conf = left_seg.confidence;
                left_has_mask = true;
                
                // 在ROI内检测椭圆
                std::vector<Ellipse> roi_ellipses = detectEllipsesInImage(left_img, left_roi);
                
                // 掩码过滤：只保留圆心在掩码内的椭圆
                for (const auto& e : roi_ellipses) {
                    int x = static_cast<int>(e.center.x);
                    int y = static_cast<int>(e.center.y);
                    if (x >= 0 && x < left_mask.cols && y >= 0 && y < left_mask.rows) {
                        if (left_mask.at<uchar>(y, x) > 0) {
                            left_ellipses.push_back(e);
                        }
                    }
                }
            }
        }

        
        // 如果没有分割结果，使用全图检测
        if (left_ellipses.empty() && !left_has_mask) {
            left_ellipses = detectEllipsesInImage(left_img);
        }
        
        // ========== 处理右图 ==========
        std::vector<Ellipse> right_ellipses;
        cv::Rect right_roi;
        cv::Mat right_mask;
        float right_conf = -1.0f;
        bool right_has_mask = false;
        

        if (segmentor_ && segmentor_->isLoaded()) {
            SegmentResult right_seg = segmentor_->segment(right_img);
            if (right_seg.valid) {
                right_roi = right_seg.roi;
                right_mask = right_seg.mask;
                right_conf = right_seg.confidence;
                right_has_mask = true;
                
                // 在ROI内检测椭圆
                std::vector<Ellipse> roi_ellipses = detectEllipsesInImage(right_img, right_roi);
                
                // 掩码过滤
                for (const auto& e : roi_ellipses) {
                    int x = static_cast<int>(e.center.x);
                    int y = static_cast<int>(e.center.y);
                    if (x >= 0 && x < right_mask.cols && y >= 0 && y < right_mask.rows) {
                        if (right_mask.at<uchar>(y, x) > 0) {
                            right_ellipses.push_back(e);
                        }
                    }
                }
            }
        }

        
        // 如果没有分割结果，使用全图检测
        if (right_ellipses.empty() && !right_has_mask) {
            right_ellipses = detectEllipsesInImage(right_img);
        }

        // ========== 收集处理状态信息 ==========
        std::vector<std::string> errors;  // 错误信息列表
        
        // 先查找锚点，用于调试可视化和圆心排序
        std::vector<Ellipse> left_anchors, right_anchors;
        std::string left_anchor_err, right_anchor_err;
        bool left_anchor_ok = findAnchors(left_ellipses, left_anchors, "L", &left_anchor_err);
        bool right_anchor_ok = findAnchors(right_ellipses, right_anchors, "R", &right_anchor_err);
        
        if (!left_anchor_err.empty()) errors.push_back("  L: " + left_anchor_err);
        if (!right_anchor_err.empty()) errors.push_back("  R: " + right_anchor_err);

        // ========== 检查锚点识别和椭圆数量 ==========
        bool left_enough = (left_ellipses.size() >= static_cast<size_t>(expected_circles));
        bool right_enough = (right_ellipses.size() >= static_cast<size_t>(expected_circles));
        
        if (!left_enough) {
            errors.push_back("  L: 椭圆不足 (" + std::to_string(left_ellipses.size()) + "/" + std::to_string(expected_circles) + ")");
        }
        if (!right_enough) {
            errors.push_back("  R: 椭圆不足 (" + std::to_string(right_ellipses.size()) + "/" + std::to_string(expected_circles) + ")");
        }
        
        if (!left_anchor_ok || !right_anchor_ok || !left_enough || !right_enough) {
            // 输出图像对状态
            std::cout << "[" << left_filename << "] L:" << left_ellipses.size() 
                      << " R:" << right_ellipses.size() << " -> 失败" << std::endl;
            for (const auto& err : errors) {
                std::cout << err << std::endl;
            }
            
            // 保存调试图像
            if (!left_enough || !left_anchor_ok) {
                saveDebugImageWithMask(left_img, left_ellipses, left_mask, left_roi,
                                       "debug_img/L_" + left_filename + "_debug.png", left_conf, left_anchors);
            }
            if (!right_enough || !right_anchor_ok) {
                saveDebugImageWithMask(right_img, right_ellipses, right_mask, right_roi,
                                       "debug_img/R_" + right_filename + "_debug.png", right_conf, right_anchors);
            }
            continue;
        }

        // ========== 使用单应性排序圆心 ==========
        // 原理：
        //   1. 用锚点计算单应性 H (模型 -> 图像)
        //   2. 用 H^-1 将所有检测点投影到模型空间
        //   3. 根据投影位置分配网格坐标 (row, col)
        //   4. 按网格坐标排序，确保与世界坐标一一对应
        
        const float spacing = board_config_.circle_spacing;
        
        auto orderByHomography = [&](const std::vector<Ellipse>& ellipses, 
                                      const std::vector<Ellipse>& anchors,
                                      std::vector<cv::Point2f>& ordered_centers) -> bool {
            // 构建锚点的模型坐标和检测坐标
            std::vector<cv::Point2f> model_pts(5), detected_pts(5);
            
            // 根据坐标模式获取锚点的模型坐标
            if (board_config_.auto_generate_coords) {
                // 自动生成模式：从 anchors (row, col) 映射计算模型坐标
                for (const auto& [id, grid_pos] : board_config_.anchors) {
                    if (id >= 0 && id < 5) {
                        model_pts[id] = cv::Point2f(grid_pos.x * spacing, grid_pos.y * spacing);
                    }
                }
            } else {
                // 外部文件模式：从 anchor_labels 查找对应的3D坐标
                std::map<std::string, cv::Point3f> label_to_coord;
                for (const auto& pt : board_config_.world_coords) {
                    label_to_coord[pt.label] = pt.coord;
                }
                
                for (size_t id = 0; id < 5; ++id) {
                    const std::string& label = board_config_.anchor_labels[id];
                    auto it = label_to_coord.find(label);
                    if (it == label_to_coord.end()) {
                        return false;
                    }
                    // 使用 X, Y 作为2D模型坐标
                    model_pts[id] = cv::Point2f(it->second.x, it->second.y);
                }
            }
            
            for (int k = 0; k < 5; ++k) {
                detected_pts[k] = cv::Point2f(anchors[k].center.x, anchors[k].center.y);
            }
            
            // 计算单应性
            cv::Mat H = cv::findHomography(model_pts, detected_pts, 0);
            if (H.empty()) return false;
            
            cv::Mat H_inv = H.inv();
            if (H_inv.empty()) return false;
            
            // 将所有检测点投影到模型空间
            std::vector<cv::Point2f> all_detected;
            for (const auto& e : ellipses) {
                all_detected.emplace_back(e.center.x, e.center.y);
            }
            
            std::vector<cv::Point2f> back_projected;
            cv::perspectiveTransform(all_detected, back_projected, H_inv);
            
            // 准备模型空间中的目标网格点
            std::vector<cv::Point2f> model_grid_pts;
            double min_dist_between_pts = 1e9;
            
            if (board_config_.auto_generate_coords) {
                // 自动生成模式：生成规则网格
                for (int r = 0; r < board_config_.rows; ++r) {
                    for (int c = 0; c < board_config_.cols; ++c) {
                        model_grid_pts.emplace_back(c * spacing, r * spacing);
                    }
                }
                min_dist_between_pts = spacing;
            } else {
                // 外部文件模式：使用预加载的坐标
                for (const auto& pt : board_config_.world_coords) {
                    model_grid_pts.emplace_back(pt.coord.x, pt.coord.y);
                }
                // 计算点间最小距离
                for (size_t i = 0; i < model_grid_pts.size(); ++i) {
                    for (size_t j = i + 1; j < model_grid_pts.size(); ++j) {
                        double d = cv::norm(model_grid_pts[i] - model_grid_pts[j]);
                        if (d > 1e-6 && d < min_dist_between_pts) {
                            min_dist_between_pts = d;
                        }
                    }
                }
            }
            
            // 网格匹配容差
            const double GRID_TOLERANCE = min_dist_between_pts / 3.0;
            
            // 为每个投影点找最近的网格点
            struct GridPoint {
                cv::Point2f image_pt;
                int grid_idx;  // 网格点索引
                double dist_to_grid;
            };
            std::vector<GridPoint> grid_points;
            
            for (size_t k = 0; k < back_projected.size(); ++k) {
                const auto& bp = back_projected[k];
                
                // 找最近的网格点
                double min_dist = 1e9;
                int best_idx = -1;
                for (size_t g = 0; g < model_grid_pts.size(); ++g) {
                    double d = cv::norm(bp - model_grid_pts[g]);
                    if (d < min_dist) {
                        min_dist = d;
                        best_idx = static_cast<int>(g);
                    }
                }
                
                // 只接受距离足够近的点
                if (min_dist < GRID_TOLERANCE && best_idx >= 0) {
                    grid_points.push_back({all_detected[k], best_idx, min_dist});
                }
            }
            
            // 检查是否有足够的点
            int expected = static_cast<int>(model_grid_pts.size());
            if (static_cast<int>(grid_points.size()) < expected * 0.9) {
                // 不在这里输出，由调用处统一处理
                return false;
            }
            
            // 去重：同一个网格位置只保留距离最近的点
            std::map<int, GridPoint> unique_grid;
            for (const auto& gp : grid_points) {
                if (unique_grid.find(gp.grid_idx) == unique_grid.end() || 
                    gp.dist_to_grid < unique_grid[gp.grid_idx].dist_to_grid) {
                    unique_grid[gp.grid_idx] = gp;
                }
            }
            
            // 检查是否覆盖了所有网格点
            if (static_cast<int>(unique_grid.size()) != expected) {
                // 不在这里输出，由调用处统一处理
                return false;
            }
            
            // 按网格索引顺序输出（与世界坐标生成顺序一致）
            ordered_centers.clear();
            ordered_centers.reserve(expected);
            for (int idx = 0; idx < expected; ++idx) {
                auto it = unique_grid.find(idx);
                if (it == unique_grid.end()) {
                    // 不在这里输出，由调用处统一处理
                    return false;
                }
                ordered_centers.push_back(it->second.image_pt);
            }
            
            return true;
        };
        
        std::vector<cv::Point2f> left_centers, right_centers;
        
        bool left_order_ok = orderByHomography(left_ellipses, left_anchors, left_centers);
        bool right_order_ok = orderByHomography(right_ellipses, right_anchors, right_centers);
        
        if (!left_order_ok || !right_order_ok) {
            std::cout << "[" << left_filename << "] L:" << left_ellipses.size() 
                      << " R:" << right_ellipses.size() << " -> 排序失败" << std::endl;
            if (!left_order_ok) std::cout << "  L: 网格匹配失败" << std::endl;
            if (!right_order_ok) std::cout << "  R: 网格匹配失败" << std::endl;
            
            // 保存调试图像
            if (!left_order_ok) {
                saveDebugImageWithMask(left_img, left_ellipses, left_mask, left_roi,
                                       "debug_img/L_" + left_filename + "_debug.png", left_conf, left_anchors);
            }
            if (!right_order_ok) {
                saveDebugImageWithMask(right_img, right_ellipses, right_mask, right_roi,
                                       "debug_img/R_" + right_filename + "_debug.png", right_conf, right_anchors);
            }
            continue;
        }
        
        // 保存结果
        image_pairs_[i].left_centers = left_centers;
        image_pairs_[i].right_centers = right_centers;
        image_pairs_[i].world_points = world_coords;
        
        std::cout << "[" << left_filename << "] L:" << left_centers.size() 
                  << " R:" << right_centers.size() << " -> 成功" << std::endl;
        
        valid_count++;
    }
    
    std::cout << "\n有效图像对: " << valid_count << "/" << image_pairs_.size() << std::endl;
    
    if (valid_count < 3) {
        std::cerr << "错误: 有效图像对不足3对，无法标定" << std::endl;
        return false;
    }
    
    return true;
}

/**
 * @brief 保存带掩码的调试图像
 */


void StereoCalibrator::saveDebugImageWithMask(
    const cv::Mat& image,
    const std::vector<Ellipse>& ellipses,
    const cv::Mat& mask,
    const cv::Rect& roi,
    const std::string& output_path,
    float confidence,
    const std::vector<Ellipse>& anchors) {
    (void)confidence;
    
    cv::Mat vis;
    cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
    
    // 绘制掩码（半透明绿色）
    if (!mask.empty()) {
        cv::Mat mask_color = cv::Mat::zeros(vis.size(), CV_8UC3);
        mask_color.setTo(cv::Scalar(0, 255, 0), mask);
        cv::addWeighted(vis, 0.7, mask_color, 0.3, 0, vis);
    }
    
    // 绘制ROI边框（黄色）
    if (roi.width > 0 && roi.height > 0) {
        cv::rectangle(vis, roi, cv::Scalar(0, 255, 255), 2);
    }
    
    // 绘制所有相关椭圆（红色点）
    for (const auto& e : ellipses) {
        cv::Point center(cvRound(e.center.x), cvRound(e.center.y));
        cv::circle(vis, center, 2, cv::Scalar(0, 0, 255), -1);
    }
    
    // 绘制锚点ID (绿色大号字体)
    for (size_t i = 0; i < anchors.size(); ++i) {
        const auto& e = anchors[i];
        cv::Point center(cvRound(e.center.x), cvRound(e.center.y));
        
        // 绘制ID
        cv::putText(vis, std::to_string(i), center + cv::Point(15, 15), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);
    }
    
    // 显示统计信息
    std::stringstream ss;
    ss << "Detected: " << ellipses.size();
    if (!anchors.empty()) ss << " | Anchors: " << anchors.size();
    cv::putText(vis, ss.str(), cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    
    cv::imwrite(output_path, vis);
}

bool StereoCalibrator::findAnchors(const std::vector<Ellipse>& ellipses, std::vector<Ellipse>& out_anchors, 
                                   const std::string& image_name, std::string* error_msg) {
    (void)image_name;  // 参数仅用于日志，现改为通过 error_msg 返回
    /**
     * RANSAC风格锚点匹配算法 (透视变换鲁棒版)
     * 
     * 算法思路:
     *   1. 取面积最大的前K个椭圆作为候选锚点
     *   2. 枚举所有 C(K,5) 种组合，每种尝试 5! 种排列
     *   3. 对每种排列，计算单应性 H (模型->图像)
     *   4. 反向验证：用 H^-1 将所有检测椭圆投影到模型空间
     *   5. 检查投影点是否落在网格点附近（内点验证）
     *   6. 选择内点数最多的排列
     * 
     * 这种方法的优点:
     *   - 不依赖像素距离比例（对透视变换鲁棒）
     *   - 使用所有检测点进行验证（对误检鲁棒）
     *   - 通过内点数自然处理椭圆数量不确定的情况
     */
    
    if (ellipses.size() < 5) return false;
    
    // 1. 按面积降序排序，取前K个候选
    std::vector<Ellipse> sorted = ellipses;
    std::sort(sorted.begin(), sorted.end(), [](const Ellipse& a, const Ellipse& b) {
        return (a.a * a.b) > (b.a * b.b);
    });
    
    // 增加候选数到15，应对更多误检
    int n_cand = std::min(static_cast<int>(sorted.size()), 7);
    std::vector<Ellipse> candidates(sorted.begin(), sorted.begin() + n_cand);
    
    // 2. 准备锚点的模型坐标
    std::vector<cv::Point2f> anchor_model_pts(5);
    
    if (board_config_.auto_generate_coords) {
        // 自动生成模式：从 anchors (row, col) 映射计算模型坐标
        if (board_config_.anchors.size() != 5) return false;
        
        const float spacing = board_config_.circle_spacing;
        for (const auto& [id, grid_pos] : board_config_.anchors) {
            if (id >= 0 && id < 5) {
                // 注意: board.yaml中 [row, col] -> 模型坐标 (col*spacing, row*spacing)
                anchor_model_pts[id] = cv::Point2f(grid_pos.x * spacing, grid_pos.y * spacing);
            }
        }
    } else {
        // 外部文件模式：从 anchor_labels 查找对应的3D坐标
        if (board_config_.anchor_labels.size() != 5) return false;
        
        // 构建标签到坐标的映射
        std::map<std::string, cv::Point3f> label_to_coord;
        for (const auto& pt : board_config_.world_coords) {
            label_to_coord[pt.label] = pt.coord;
        }
        
        for (size_t id = 0; id < 5; ++id) {
            const std::string& label = board_config_.anchor_labels[id];
            auto it = label_to_coord.find(label);
            if (it == label_to_coord.end()) {
                std::cerr << "[findAnchors] 找不到锚点标签: " << label << std::endl;
                return false;
            }
            // 使用 X, Y 作为2D模型坐标（Z通常接近0）
            anchor_model_pts[id] = cv::Point2f(it->second.x, it->second.y);
        }
    }
    
    // 3. 准备所有检测椭圆的中心点
    std::vector<cv::Point2f> all_detected;
    all_detected.reserve(ellipses.size());
    for (const auto& e : ellipses) {
        all_detected.emplace_back(e.center.x, e.center.y);
    }
    
    // 4. RANSAC风格暴力搜索
    // 复杂度: C(15,5) * 5! = 3003 * 120 = 360,360 次迭代（可接受）
    
    int best_inliers = -1;
    double best_fit_error = 1e9;
    std::vector<Ellipse> best_anchors;
    
    // 准备模型空间中的所有目标点
    std::vector<cv::Point2f> model_grid_pts;
    double min_dist_between_pts = 1e9;  // 用于计算容差
    
    if (board_config_.auto_generate_coords) {
        // 自动生成模式：生成规则网格
        const float spacing = board_config_.circle_spacing;
        for (int r = 0; r < board_config_.rows; ++r) {
            for (int c = 0; c < board_config_.cols; ++c) {
                model_grid_pts.emplace_back(c * spacing, r * spacing);
            }
        }
        min_dist_between_pts = spacing;
    } else {
        // 外部文件模式：使用预加载的坐标
        for (const auto& pt : board_config_.world_coords) {
            model_grid_pts.emplace_back(pt.coord.x, pt.coord.y);
        }
        // 计算点间最小距离
        for (size_t i = 0; i < model_grid_pts.size(); ++i) {
            for (size_t j = i + 1; j < model_grid_pts.size(); ++j) {
                double d = cv::norm(model_grid_pts[i] - model_grid_pts[j]);
                if (d > 1e-6 && d < min_dist_between_pts) {
                    min_dist_between_pts = d;
                }
            }
        }
    }
    
    // 内点阈值：与最近网格点的距离 < min_dist / 4
    const double GRID_TOLERANCE = min_dist_between_pts / 4.0;
    
    // 生成组合的位掩码
    std::vector<int> combo(n_cand, 0);
    std::fill(combo.begin(), combo.begin() + 5, 1);
    
    do {
        // 提取当前组合的5个候选
        std::vector<Ellipse> subset;
        subset.reserve(5);
        for (int i = 0; i < n_cand; ++i) {
            if (combo[i]) subset.push_back(candidates[i]);
        }
        
        // 尝试所有 5! 种排列
        std::vector<int> perm = {0, 1, 2, 3, 4};
        do {
            // 构建当前排列的检测点
            std::vector<cv::Point2f> detected_pts(5);
            for (int k = 0; k < 5; ++k) {
                detected_pts[k] = cv::Point2f(subset[perm[k]].center.x, subset[perm[k]].center.y);
            }
            
            // 计算单应性: 模型 -> 图像
            cv::Mat H = cv::findHomography(anchor_model_pts, detected_pts, 0);
            if (H.empty()) continue;
            
            // 快速验证：5点拟合误差
            std::vector<cv::Point2f> reproj;
            cv::perspectiveTransform(anchor_model_pts, reproj, H);
            double fit_error = 0;
            for (int k = 0; k < 5; ++k) {
                fit_error += cv::norm(reproj[k] - detected_pts[k]);
            }
            if (fit_error > 10.0) continue;  // 5点拟合误差应该很小
            
            // 反向投影验证：将所有检测点投影到模型空间
            cv::Mat H_inv = H.inv();
            if (H_inv.empty()) continue;
            
            std::vector<cv::Point2f> back_projected;
            cv::perspectiveTransform(all_detected, back_projected, H_inv);
            
            // 计算内点数：检查投影点是否落在预定义网格点附近
            int inliers = 0;
            for (const auto& bp : back_projected) {
                // 找最近的网格点
                double min_dist = 1e9;
                for (const auto& gp : model_grid_pts) {
                    double d = cv::norm(bp - gp);
                    if (d < min_dist) min_dist = d;
                }
                
                if (min_dist < GRID_TOLERANCE) {
                    inliers++;
                }
            }
            
            // 更新最佳结果
            // 优先选择内点多的，内点相同时选择拟合误差小的
            bool update = false;
            if (inliers > best_inliers) {
                update = true;
            } else if (inliers == best_inliers && fit_error < best_fit_error) {
                update = true;
            }
            
            if (update) {
                best_inliers = inliers;
                best_fit_error = fit_error;
                best_anchors.resize(5);
                for (int k = 0; k < 5; ++k) {
                    best_anchors[k] = subset[perm[k]];
                }
            }
            
        } while (std::next_permutation(perm.begin(), perm.end()));
        
    } while (std::prev_permutation(combo.begin(), combo.end()));
    
    // 5. 输出结果
    if (best_inliers < 0) {
        if (error_msg) *error_msg = "未找到有效的锚点组合";
        return false;
    }
    
    // 要求至少有50%的检测点是内点
    // 这个阈值比较宽松，因为可能有误检和漏检
    int min_required = static_cast<int>(ellipses.size() * 0.5);
    if (best_inliers < min_required) {
        if (error_msg) {
            *error_msg = "锚点内点过少: " + std::to_string(best_inliers) + "/" + 
                         std::to_string(ellipses.size()) + " (需要至少" + std::to_string(min_required) + ")";
        }
        return false;
    }
    
    out_anchors = best_anchors;
    return true;
}

/**
 * @brief 执行双目标定
 * 
 * 采用两阶段标定策略：
 *   阶段 1: 使用平面坐标 (Z=0) 获得初始内参
 *   阶段 2: 使用真实 3D 坐标 (含 Z) + CALIB_USE_INTRINSIC_GUESS 进行精确标定
 * 
 * 原理: OpenCV 的 calibrateCamera 对于非平面点需要初始内参矩阵，
 *       因为无法从非平面单张图像恢复唯一的内参解。
 */
bool StereoCalibrator::calibrate() {
    // 1. 准备标定数据
    std::vector<std::vector<cv::Point3f>> object_points_flat;  // 平面化坐标 (Z=0)
    std::vector<std::vector<cv::Point3f>> object_points_real;  // 真实 3D 坐标
    std::vector<std::vector<cv::Point2f>> left_image_points;
    std::vector<std::vector<cv::Point2f>> right_image_points;
    
    // 生成平面化和真实的世界坐标
    std::vector<cv::Point3f> world_coords_flat = generateWorldCoordinates(true);   // Z=0
    std::vector<cv::Point3f> world_coords_real = generateWorldCoordinates(false);  // 真实 Z
    
    for (const auto& pair : image_pairs_) {
        if (pair.isValid()) {
            object_points_flat.push_back(world_coords_flat);
            object_points_real.push_back(world_coords_real);
            left_image_points.push_back(pair.left_centers);
            right_image_points.push_back(pair.right_centers);
        }
    }
    
    if (object_points_flat.size() < 3) {
        std::cerr << "错误: 有效图像对不足3对" << std::endl;
        return false;
    }
    
    // 检查是否需要两阶段标定（外部文件模式且包含非零 Z）
    bool need_two_stage = !board_config_.auto_generate_coords;
    if (need_two_stage) {
        // 检查是否真的有非零 Z
        double max_z = 0;
        for (const auto& pt : world_coords_real) {
            max_z = std::max(max_z, static_cast<double>(std::abs(pt.z)));
        }
        need_two_stage = (max_z > 0.01);  // Z 偏差 > 0.01mm 才启用两阶段
    }
    
    std::cout << "\n双目标定中..." << std::endl;
    if (need_two_stage) {
        std::cout << "  [使用两阶段标定: 先平面初始化，再精确标定]" << std::endl;
    }

    // ========== 阶段 1: 使用平面坐标获得初始内参 ==========
    auto start_left = std::chrono::high_resolution_clock::now();
    
    // 左相机初始标定
    result_.rms_left = cv::calibrateCamera(
        object_points_flat, 
        left_image_points, 
        image_size_,
        result_.camera_matrix_left, 
        result_.dist_coeffs_left,
        rvecs_left_, 
        tvecs_left_,
        cv::CALIB_RATIONAL_MODEL + cv::CALIB_FIX_K3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 150, 1e-6)
    );
    auto end_left = std::chrono::high_resolution_clock::now();
    auto duration_left = std::chrono::duration_cast<std::chrono::milliseconds>(end_left - start_left);
    
    // 右相机初始标定
    auto start_right = std::chrono::high_resolution_clock::now();
    result_.rms_right = cv::calibrateCamera(
        object_points_flat, 
        right_image_points, 
        image_size_,
        result_.camera_matrix_right, 
        result_.dist_coeffs_right,
        rvecs_right_, 
        tvecs_right_,
        cv::CALIB_RATIONAL_MODEL + cv::CALIB_FIX_K3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 150, 1e-6)
    );
    auto end_right = std::chrono::high_resolution_clock::now();
    auto duration_right = std::chrono::duration_cast<std::chrono::milliseconds>(end_right - start_right);
    
    // ========== 阶段 2: 使用真实 3D 坐标精确标定 ==========
    if (need_two_stage) {
        std::cout << "  [阶段 2: 使用真实 3D 坐标精确标定]" << std::endl;
        
        // 左相机精确标定
        result_.rms_left = cv::calibrateCamera(
            object_points_real, 
            left_image_points, 
            image_size_,
            result_.camera_matrix_left, 
            result_.dist_coeffs_left,
            rvecs_left_, 
            tvecs_left_,
            cv::CALIB_USE_INTRINSIC_GUESS + cv::CALIB_RATIONAL_MODEL + cv::CALIB_FIX_K3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 150, 1e-6)
        );
        
        // 右相机精确标定
        result_.rms_right = cv::calibrateCamera(
            object_points_real, 
            right_image_points, 
            image_size_,
            result_.camera_matrix_right, 
            result_.dist_coeffs_right,
            rvecs_right_, 
            tvecs_right_,
            cv::CALIB_USE_INTRINSIC_GUESS + cv::CALIB_RATIONAL_MODEL + cv::CALIB_FIX_K3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 150, 1e-6)
        );
    }
    
    // ========== 双目标定 ==========
    // 使用真实 3D 坐标（如果可用）进行双目标定
    auto& object_points_stereo = need_two_stage ? object_points_real : object_points_flat;
    
    auto start_stereo = std::chrono::high_resolution_clock::now();
    result_.rms_stereo = cv::stereoCalibrate(
        object_points_stereo, 
        left_image_points, 
        right_image_points,
        result_.camera_matrix_left, 
        result_.dist_coeffs_left,
        result_.camera_matrix_right, 
        result_.dist_coeffs_right,
        image_size_, 
        result_.R, 
        result_.T, 
        result_.E, 
        result_.F,
        cv::CALIB_USE_INTRINSIC_GUESS + 
        cv::CALIB_RATIONAL_MODEL + 
        cv::CALIB_FIX_K3 + 
        cv::CALIB_FIX_K4 + 
        cv::CALIB_FIX_K5,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 150, 1e-6)
    );
    auto end_stereo = std::chrono::high_resolution_clock::now();
    auto duration_stereo = std::chrono::duration_cast<std::chrono::milliseconds>(end_stereo - start_stereo);
    
    // 5. 计算误差
    computeReprojectionErrors();
    
    return true;
}

/**
 * @brief 保存标定结果到 YAML 文件
 */
bool StereoCalibrator::saveResults(const std::string& yaml_path) {
    std::cout << "\n=== 保存标定结果 ===" << std::endl;
    
    cv::FileStorage fs(yaml_path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "错误: 无法打开文件 " << yaml_path << std::endl;
        return false;
    }
    
    // 保存左相机参数
    fs << "camera_matrix_left" << result_.camera_matrix_left;
    fs << "dist_coeffs_left" << result_.dist_coeffs_left;
    
    // 保存右相机参数
    fs << "camera_matrix_right" << result_.camera_matrix_right;
    fs << "dist_coeffs_right" << result_.dist_coeffs_right;
    
    // 保存双目外参
    fs << "R" << result_.R;
    fs << "T" << result_.T;
    fs << "E" << result_.E;
    fs << "F" << result_.F;
    
    // 保存标定质量指标
    fs << "rms_left" << result_.rms_left;
    fs << "rms_right" << result_.rms_right;
    fs << "rms_stereo" << result_.rms_stereo;
    
    // 保存标定板参数
    fs << "board_rows" << board_config_.rows;
    fs << "board_cols" << board_config_.cols;
    fs << "circle_spacing" << board_config_.circle_spacing;
    
    fs.release();
    
    std::cout << "标定结果已保存到: " << yaml_path << std::endl;
    return true;
}

/**
 * @brief 输出 3D 点云到 TXT 文件（基于双目标定结果的三角化重建）
 */
bool StereoCalibrator::export3DPoints(const std::string& txt_path) {
    std::cout << "\n=== 输出3D点云 ===" << std::endl;
    
    std::ofstream ofs(txt_path);
    if (!ofs.is_open()) {
        std::cerr << "错误: 无法打开文件 " << txt_path << std::endl;
        return false;
    }
    
    // 计算立体校正参数（用于三角化）
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(
        result_.camera_matrix_left, result_.dist_coeffs_left,
        result_.camera_matrix_right, result_.dist_coeffs_right,
        image_size_,
        result_.R, result_.T,
        R1, R2, P1, P2, Q
    );
    
    // 输出文件头
    ofs << "# 双目标定板3D点云（基于双目标定结果的三角化重建）" << std::endl;
    ofs << "# 格式: X Y Z (单位: 毫米)" << std::endl;
    ofs << "# 标定板: " << board_config_.rows << "行 x " << board_config_.cols << "列, 圆心间距 " 
        << board_config_.circle_spacing << "mm" << std::endl;
    ofs << "# 坐标系: 校正后的左相机坐标系" << std::endl;
    ofs << std::endl;
    
    int total_points = 0;
    int valid_image_count = 0;
    
    // 对每个有效图像对进行三角化
    for (size_t i = 0; i < image_pairs_.size(); ++i) {
        if (!image_pairs_[i].isValid()) {
            continue;
        }
        
        // 提取文件名
        std::string left_filename = std::filesystem::path(image_pairs_[i].left_path).stem().string();
        std::string right_filename = std::filesystem::path(image_pairs_[i].right_path).stem().string();
        
        ofs << "# 图像对 " << left_filename << "/" << right_filename << std::endl;
        
        // 步骤1: 去畸变并校正到归一化坐标
        std::vector<cv::Point2f> left_normalized, right_normalized;
        cv::undistortPoints(
            image_pairs_[i].left_centers,
            left_normalized,
            result_.camera_matrix_left,
            result_.dist_coeffs_left,
            R1,  // 应用校正旋转
            P1   // 应用校正投影矩阵
        );
        
        cv::undistortPoints(
            image_pairs_[i].right_centers,
            right_normalized,
            result_.camera_matrix_right,
            result_.dist_coeffs_right,
            R2,  // 应用校正旋转
            P2   // 应用校正投影矩阵
        );
        
        // 步骤2: 使用校正后的投影矩阵进行三角化
        cv::Mat points4D;
        cv::triangulatePoints(
            P1, P2,
            left_normalized,
            right_normalized,
            points4D
        );
        
        // 步骤3: 转换为3D点（齐次坐标转欧式坐标）
        for (int j = 0; j < points4D.cols; ++j) {
            float w = points4D.at<float>(3, j);
            if (std::abs(w) < 1e-6) {
                // 跳过无效点
                continue;
            }
            
            float x = points4D.at<float>(0, j) / w;
            float y = points4D.at<float>(1, j) / w;
            float z = points4D.at<float>(2, j) / w;
            
            // 输出3D点（单位：毫米）
            ofs << std::fixed << std::setprecision(6) 
                << x << " " << y << " " << z << std::endl;
            total_points++;
        }
        
        ofs << std::endl;
        valid_image_count++;
    }
    
    ofs.close();
    
    std::cout << "3D点云已保存到: " << txt_path << std::endl;
    std::cout << "共输出 " << valid_image_count << " 组图像，" << total_points << " 个3D点" << std::endl;
    
    return true;
}

/**
 * @brief 在单张图像中检测椭圆
 * 
 * @param image 输入图像
 * @param roi 感兴趣区域（可选），如果指定则只在ROI内检测
 * @return 检测到的椭圆列表（坐标为原图坐标系）
 */
std::vector<Ellipse> StereoCalibrator::detectEllipsesInImage(const cv::Mat& image, const cv::Rect& roi) {
    DetectorParams params;
    // 使用默认参数：
    // - gradThresh = 11（梯度阈值）
    // - polarity = 1（检测暗圆）
    // - threads = 8（并行线程数）
    // - minMajorAxis = 3.0（最小长轴）
    // - maxMajorAxis = 无限大（最大长轴）
    
    std::vector<Ellipse> ellipses;
    
    if (roi.width > 0 && roi.height > 0) {
        // 在ROI区域内检测
        cv::Mat roi_image = image(roi);
        ellipses = detectEllipses(roi_image, params);
        
        // 将椭圆坐标转换回原图坐标系
        for (auto& ellipse : ellipses) {
            ellipse.center.x += roi.x;
            ellipse.center.y += roi.y;
        }
    } else {
        // 全图检测
        ellipses = detectEllipses(image, params);
    }
    
    return ellipses;
}

/**
 * @brief 从椭圆列表中提取圆心坐标
 */
std::vector<cv::Point2f> StereoCalibrator::extractCenters(const std::vector<Ellipse>& ellipses) {
    std::vector<cv::Point2f> centers;
    centers.reserve(ellipses.size());
    
    for (const auto& ellipse : ellipses) {
        // 椭圆的 center 已经是亚像素级的 cv::Point2d
        // 转换为 cv::Point2f
        centers.push_back(cv::Point2f(
            static_cast<float>(ellipse.center.x),
            static_cast<float>(ellipse.center.y)
        ));
    }
    
    return centers;
}

/**
 * @brief 生成所有圆点的世界坐标
 * 
 * @param flatten_z 是否将 Z 坐标强制设为 0（平面化）
 *                  - true: 用于初始标定（OpenCV 需要平面点来估计初始内参）
 *                  - false: 使用真实 Z 值进行精确标定
 * 
 * 根据配置选择自动生成或使用外部文件加载的坐标
 */
std::vector<cv::Point3f> StereoCalibrator::generateWorldCoordinates(bool flatten_z) {
    std::vector<cv::Point3f> world_points;
    
    if (board_config_.auto_generate_coords) {
        // 自动生成模式：按行列顺序生成世界坐标（Z 始终为 0）
        world_points.reserve(board_config_.rows * board_config_.cols);
        
        for (int r = 0; r < board_config_.rows; ++r) {
            for (int c = 0; c < board_config_.cols; ++c) {
                world_points.push_back(cv::Point3f(
                    c * board_config_.circle_spacing,
                    r * board_config_.circle_spacing,
                    0.0f
                ));
            }
        }
    } else {
        // 外部文件模式：使用预加载的坐标数据
        world_points.reserve(board_config_.world_coords.size());
        
        for (const auto& pt : board_config_.world_coords) {
            if (flatten_z) {
                // 平面化：用于初始标定
                world_points.push_back(cv::Point3f(pt.coord.x, pt.coord.y, 0.0f));
            } else {
                // 使用真实 Z：用于精确标定
                world_points.push_back(pt.coord);
            }
        }
    }
    
    return world_points;
}

/**
 * @brief 计算重投影误差
 */
void StereoCalibrator::computeReprojectionErrors() {
    std::vector<double> pair_3d_errors;
    
    std::cout << "\n=== 3D重建误差 (基于点云对齐分析) ===" << std::endl;
    
    // 双目外参
    cv::Mat R_stereo = result_.R;
    cv::Mat T_stereo = result_.T;

    for (size_t i = 0; i < image_pairs_.size(); ++i) {
        if (!image_pairs_[i].isValid()) continue;
        
        const auto& world_pts_3f = image_pairs_[i].world_points;
        const auto& left_pts = image_pairs_[i].left_centers;
        const auto& right_pts = image_pairs_[i].right_centers;
        int N = static_cast<int>(world_pts_3f.size());
        
        // 准备双精度世界点
        std::vector<cv::Point3d> world_pts;
        for(const auto& p : world_pts_3f) world_pts.emplace_back(p.x, p.y, p.z);
        
        // 去畸变得到归一化坐标 (Normalization Plane)
        std::vector<cv::Point2f> undist_left, undist_right;
        cv::undistortPoints(left_pts, undist_left, result_.camera_matrix_left, result_.dist_coeffs_left);
        cv::undistortPoints(right_pts, undist_right, result_.camera_matrix_right, result_.dist_coeffs_right);
        
        // 三角化投影矩阵 (归一化投影)
        cv::Mat P_left_norm = (cv::Mat_<double>(3,4) << 1,0,0,0, 0,1,0,0, 0,0,1,0);
        cv::Mat P_right_norm = cv::Mat::zeros(3, 4, CV_64F);
        R_stereo.copyTo(P_right_norm(cv::Rect(0, 0, 3, 3)));
        T_stereo.copyTo(P_right_norm(cv::Rect(3, 0, 1, 3)));
        
        cv::Mat points_4d;
        cv::triangulatePoints(P_left_norm, P_right_norm, undist_left, undist_right, points_4d);
        
        std::vector<cv::Point3d> pts_reconstructed;
        for (int j = 0; j < points_4d.cols; ++j) {
            double w = points_4d.at<float>(3, j);
            if (std::abs(w) > 1e-6) {
                pts_reconstructed.emplace_back(
                    points_4d.at<float>(0, j) / w,
                    points_4d.at<float>(1, j) / w,
                    points_4d.at<float>(2, j) / w);
            } else {
                pts_reconstructed.emplace_back(0, 0, 0);
            }
        }
        
        // ========== Procrustes 对齐 相当于 estimateAffine3D SVD) ==========
        auto get_centroid = [](const std::vector<cv::Point3d>& pts) {
            cv::Point3d c(0,0,0);
            for(const auto& p : pts) c += p;
            return c * (1.0 / pts.size());
        };
        
        cv::Point3d c_src = get_centroid(pts_reconstructed);
        cv::Point3d c_dst = get_centroid(world_pts);
        
        cv::Mat H = cv::Mat::zeros(3, 3, CV_64F);
        for(int j=0; j<N; ++j) {
            cv::Mat src_v = (cv::Mat_<double>(3,1) << pts_reconstructed[j].x - c_src.x, 
                                                     pts_reconstructed[j].y - c_src.y, 
                                                     pts_reconstructed[j].z - c_src.z);
            cv::Mat dst_v = (cv::Mat_<double>(3,1) << world_pts[j].x - c_dst.x, 
                                                     world_pts[j].y - c_dst.y, 
                                                     world_pts[j].z - c_dst.z);
            H += src_v * dst_v.t();
        }
        
        cv::SVD svd(H);
        cv::Mat R_align = svd.vt.t() * svd.u.t();
        if (cv::determinant(R_align) < 0) {
            cv::Mat V = svd.vt.t();
            V.col(2) *= -1;
            R_align = V * svd.u.t();
        }
        cv::Mat t_align = (cv::Mat_<double>(3,1) << c_dst.x, c_dst.y, c_dst.z) - 
                          R_align * (cv::Mat_<double>(3,1) << c_src.x, c_src.y, c_src.z);
        
        // 计算 RMS 误差
        double sum_sq_err = 0;
        for (int j = 0; j < N; ++j) {
            cv::Mat p_src = (cv::Mat_<double>(3,1) << pts_reconstructed[j].x, pts_reconstructed[j].y, pts_reconstructed[j].z);
            cv::Mat p_aligned = R_align * p_src + t_align;
            
            double dx = p_aligned.at<double>(0) - world_pts[j].x;
            double dy = p_aligned.at<double>(1) - world_pts[j].y;
            double dz = p_aligned.at<double>(2) - world_pts[j].z;
            sum_sq_err += (dx*dx + dy*dy + dz*dz);
        }
        
        // 用户要求的公式: norm(predicted, target) / sqrt(N)
        double pair_error = std::sqrt(sum_sq_err) / std::sqrt(static_cast<double>(N));
        pair_3d_errors.push_back(pair_error);
        
        std::string img_name = std::filesystem::path(image_pairs_[i].left_path).stem().string();
        std::cout << "  " << img_name << ": " << std::fixed << std::setprecision(4) 
                   << pair_error << " mm" << std::endl;
    }
    
    // 最终总体误差
    if (!pair_3d_errors.empty()) {
        double sum_sq = 0;
        for (double e : pair_3d_errors) sum_sq += e * e;
        double rms_3d = std::sqrt(sum_sq / pair_3d_errors.size());
        std::cout << "最终双目重建RMS误差: " << std::fixed << std::setprecision(4) << rms_3d << " mm" << std::endl;
    }

    std::cout << "\n=== 重投影误差摘要 (OpenCV RMS) ===" << std::endl;
    std::cout << "左相机 RMS: " << std::fixed << std::setprecision(4) << result_.rms_left << " px" << std::endl;
    std::cout << "右相机 RMS: " << std::fixed << std::setprecision(4) << result_.rms_right << " px" << std::endl;
    std::cout << "双目系统 RMS: " << std::fixed << std::setprecision(4) << result_.rms_stereo << " px" << std::endl;
}

/**
 * @brief 初始化YOLO分割器
 */
bool StereoCalibrator::initSegmentor(const std::string& model_path, float conf_threshold, int target_size) {
    try {
        segmentor_ = std::make_unique<YoloSegmentor>(model_path, conf_threshold, target_size);
        if (segmentor_->isLoaded()) {
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "[StereoCalibrator] 分割器初始化失败: " << e.what() << std::endl;
    }
    segmentor_.reset();
    return false;
}



/**
 * @brief 评估几何一致性（第一性原理误差）
 */
void StereoCalibrator::evaluateGeometricConsistency() {
    std::cout << "\n=== 评估几何一致性 (第一性原理) ===" << std::endl;
    std::cout << "指标说明: RMSE (均方根误差) - 衡量检测点与最佳拟合平面的偏差" << std::endl;
    
    float total_board_error = 0;
    int valid_count = 0;

    for (size_t i = 0; i < image_pairs_.size(); ++i) {
        if (!image_pairs_[i].isValid()) continue;
        
        auto evaluateImage = [&](const std::vector<cv::Point2f>& detected, const std::string& name) -> float {
            // 构建理想坐标
            std::vector<cv::Point2f> ideal;
            for(const auto& p3 : image_pairs_[i].world_points) {
                ideal.push_back(cv::Point2f(p3.x, p3.y));
            }
            
            // 计算最佳单应性 (使用 LMEDS 抗噪)
            cv::Mat H = cv::findHomography(ideal, detected, cv::LMEDS);
            if (H.empty()) return -1.0f;
            
            // 计算重投影点
            std::vector<cv::Point2f> reprojected;
            cv::perspectiveTransform(ideal, reprojected, H);
            
            // 计算反投影点 (图像 -> 标定板)
            cv::Mat H_inv = H.inv();
            std::vector<cv::Point2f> back_projected;
            cv::perspectiveTransform(detected, back_projected, H_inv);
            
            double sum_sq_err_px = 0;
            double sum_sq_err_mm = 0;
            
            for(size_t k=0; k<detected.size(); ++k) {
                double err_px = cv::norm(detected[k] - reprojected[k]);
                sum_sq_err_px += err_px * err_px;
                
                double err_mm = cv::norm(ideal[k] - back_projected[k]);
                sum_sq_err_mm += err_mm * err_mm;
            }
            
            float rmse_px = std::sqrt(sum_sq_err_px / detected.size());
            float rmse_mm = std::sqrt(sum_sq_err_mm / detected.size());
            
            std::cout << "  " << name << ": RMSE(Px)=" << std::fixed << std::setprecision(3) << rmse_px 
                      << ", RMSE(mm)=" << rmse_mm;
            
            if (rmse_mm > 0.5) std::cout << " [警告: 几何一致性差]";
            std::cout << std::endl;
            
            return rmse_mm;
        };
        
        std::string left_name = std::filesystem::path(image_pairs_[i].left_path).stem().string();
        float err_l = evaluateImage(image_pairs_[i].left_centers, "L_" + left_name);
        float err_r = evaluateImage(image_pairs_[i].right_centers, "R_" + left_name); // use same stem usually
        
        if (err_l >= 0 && err_r >= 0) {
            total_board_error += (err_l + err_r) / 2.0f;
            valid_count++;
        }
    }
    
    if (valid_count > 0) {
        std::cout << "平均平面拟合误差: " << (total_board_error / valid_count) << " mm" << std::endl;
    }
}
