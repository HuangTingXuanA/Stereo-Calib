/**
 * @file visualize_all_matches.cpp
 * @brief 可视化所有测试图像的匹配结果
 */

#include "include/BoardConfig.h"
#include "include/ellipse_detector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

// 提取5个特征大圆（使用改进的聚类方法）
bool extractAnchorCircles(const std::vector<Ellipse>& ellipses,
                          std::vector<Ellipse>& anchors,
                          std::vector<int>& indices) {
    if (ellipses.size() < 5) {
        return false;
    }
    
    // 计算所有圆的面积并排序
    std::vector<std::pair<double, int>> sorted_areas;
    for (size_t i = 0; i < ellipses.size(); ++i) {
        double area = CV_PI * ellipses[i].a * ellipses[i].b;
        sorted_areas.push_back({area, static_cast<int>(i)});
    }
    std::sort(sorted_areas.rbegin(), sorted_areas.rend());
    
    // 策略：使用K-means聚类思想，找到面积最大且最一致的5个圆
    // 1. 先检查前10个圆（如果有的话）
    int check_count = std::min(15, static_cast<int>(sorted_areas.size()));
    
    // 2. 使用滑动窗口找到最一致的5个圆
    int best_start = 0;
    double best_score = std::numeric_limits<double>::max();
    
    for (int start = 0; start <= check_count - 5; ++start) {
        // 计算这5个圆的面积统计
        double sum = 0.0;
        for (int i = start; i < start + 5; ++i) {
            sum += sorted_areas[i].first;
        }
        double mean = sum / 5.0;
        
        // 计算变异系数
        double variance = 0.0;
        for (int i = start; i < start + 5; ++i) {
            double diff = sorted_areas[i].first - mean;
            variance += diff * diff;
        }
        variance /= 5.0;
        double cv = std::sqrt(variance) / mean;
        
        // 综合评分：变异系数 + 位置惩罚（越靠后惩罚越大）
        // 这样既考虑一致性，又倾向于选择较大的圆
        double position_penalty = start * 0.05;  // 每后移一位增加5%惩罚
        double score = cv + position_penalty;
        
        if (score < best_score) {
            best_score = score;
            best_start = start;
        }
    }
    
    // 提取选定的5个圆
    anchors.clear();
    indices.clear();
    for (int i = best_start; i < best_start + 5; ++i) {
        int idx = sorted_areas[i].second;
        anchors.push_back(ellipses[idx]);
        indices.push_back(idx);
    }
    
    return true;
}

// 计算图像指纹
cv::Mat computeImageFingerprint(const std::vector<Ellipse>& ellipses) {
    cv::Mat dist_fingerprint = cv::Mat::zeros(5, 5, CV_64F);
    cv::Mat angle_fingerprint = cv::Mat::zeros(5, 5, CV_64F);
    
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            if (i == j) {
                dist_fingerprint.at<double>(i, j) = 0.0;
                angle_fingerprint.at<double>(i, j) = 0.0;
            } else {
                double dist = cv::norm(ellipses[i].center - ellipses[j].center);
                dist_fingerprint.at<double>(i, j) = dist;
                
                double dx = ellipses[j].center.x - ellipses[i].center.x;
                double dy = ellipses[j].center.y - ellipses[i].center.y;
                double angle = std::atan2(dy, dx);
                angle_fingerprint.at<double>(i, j) = angle;
            }
        }
    }
    
    cv::Mat fingerprint = cv::Mat::zeros(10, 5, CV_64F);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            fingerprint.at<double>(i, j) = dist_fingerprint.at<double>(i, j);
            fingerprint.at<double>(i + 5, j) = angle_fingerprint.at<double>(i, j);
        }
    }
    
    return fingerprint;
}

// 使用重投影误差找到最佳匹配
std::vector<int> findBestMatch(const std::vector<Ellipse>& large_circles, const BoardConfig& config) {
    std::vector<int> indices = {0, 1, 2, 3, 4};
    std::vector<int> best_perm = indices;
    double best_error = std::numeric_limits<double>::max();
    
    do {
        std::vector<cv::Point2f> src_points, dst_points;
        for (int i = 0; i < 5; ++i) {
            const auto& anchor = config.anchors[indices[i]];
            src_points.push_back(cv::Point2f(anchor.world.x, anchor.world.y));
            dst_points.push_back(cv::Point2f(large_circles[i].center.x, large_circles[i].center.y));
        }
        
        cv::Mat H = cv::findHomography(src_points, dst_points, 0);
        std::vector<cv::Point2f> transformed;
        cv::perspectiveTransform(src_points, transformed, H);
        
        double total_error = 0.0;
        for (int i = 0; i < 5; ++i) {
            double error = cv::norm(dst_points[i] - transformed[i]);
            total_error += error;
        }
        double avg_error = total_error / 5.0;
        
        if (avg_error < best_error) {
            best_error = avg_error;
            best_perm = indices;
        }
        
    } while (std::next_permutation(indices.begin(), indices.end()));
    
    return best_perm;
}

// 处理单张图像（简化输出）
double processImage(const std::string& image_path, const std::string& output_path, 
                    const BoardConfig& config) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        return -1.0;
    }
    
    DetectorParams params;
    std::vector<Ellipse> all_ellipses = detectEllipses(image, params);
    
    if (all_ellipses.size() < 5) {
        return -1.0;
    }
    
    // 提取5个特征大圆
    std::vector<Ellipse> large_circles;
    std::vector<int> large_indices;
    if (!extractAnchorCircles(all_ellipses, large_circles, large_indices)) {
        return -1.0;
    }
    
    // 找到最佳匹配
    std::vector<int> best_perm = findBestMatch(large_circles, config);
    
    // 计算重投影误差
    std::vector<cv::Point2f> src_points, dst_points;
    for (int i = 0; i < 5; ++i) {
        const auto& anchor = config.anchors[best_perm[i]];
        src_points.push_back(cv::Point2f(anchor.world.x, anchor.world.y));
        dst_points.push_back(cv::Point2f(large_circles[i].center.x, large_circles[i].center.y));
    }
    
    cv::Mat H = cv::findHomography(src_points, dst_points, 0);
    std::vector<cv::Point2f> transformed;
    cv::perspectiveTransform(src_points, transformed, H);
    
    double total_error = 0.0;
    for (int i = 0; i < 5; ++i) {
        total_error += cv::norm(dst_points[i] - transformed[i]);
    }
    double avg_error = total_error / 5.0;
    
    // 推算所有圆的位置
    std::vector<cv::Point2f> all_world_points;
    for (int r = 0; r < config.rows; ++r) {
        for (int c = 0; c < config.cols; ++c) {
            all_world_points.push_back(cv::Point2f(
                c * config.circle_spacing,
                r * config.circle_spacing
            ));
        }
    }
    
    std::vector<cv::Point2f> predicted_circles;
    cv::perspectiveTransform(all_world_points, predicted_circles, H);
    
    // 可视化
    cv::Mat vis;
    cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
    
    // 绘制所有推算的圆（绿色小点）
    for (const auto& pt : predicted_circles) {
        cv::circle(vis, cv::Point(pt.x, pt.y), 3, cv::Scalar(0, 255, 0), -1);
    }
    
    // 绘制检测到的椭圆（蓝色）
    for (const auto& ellipse : all_ellipses) {
        cv::ellipse(vis,
                   cv::Point(ellipse.center.x, ellipse.center.y),
                   cv::Size(ellipse.a, ellipse.b),
                   ellipse.phi * 180.0 / CV_PI,
                   0, 360,
                   cv::Scalar(255, 0, 0), 1);
    }
    
    // 绘制5个大圆（红色）
    for (int i = 0; i < 5; ++i) {
        const auto& ellipse = large_circles[i];
        cv::ellipse(vis,
                   cv::Point(ellipse.center.x, ellipse.center.y),
                   cv::Size(ellipse.a, ellipse.b),
                   ellipse.phi * 180.0 / CV_PI,
                   0, 360,
                   cv::Scalar(0, 0, 255), 2);
        
        std::string label = std::to_string(i) + "->" + std::to_string(best_perm[i]);
        cv::putText(vis, label,
                   cv::Point(ellipse.center.x + 15, ellipse.center.y - 15),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }
    
    // 添加信息文本
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4) << avg_error;
    std::string info = "RMSE: " + ss.str() + "px";
    cv::putText(vis, info, cv::Point(20, 40),
               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
    
    cv::imwrite(output_path, vis);
    
    return avg_error;
}

int main() {
    std::cout << "批量可视化所有测试图像的匹配结果" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // 加载配置
    BoardConfig config;
    config.loadFromFile("calibration_board.yaml");
    
    // 创建输出目录
    fs::create_directories("match_results");
    
    // 处理所有测试图像
    for (int i = 0; i <= 40; ++i) {
        std::string left_path = "images/left/image_" + std::to_string(i) + ".bmp";
        std::string right_path = "images/right/image_" + std::to_string(i) + ".bmp";
        
        std::string left_output = "match_results/left_image_" + std::to_string(i) + ".png";
        std::string right_output = "match_results/right_image_" + std::to_string(i) + ".png";
        
        // 处理左图
        double left_rmse = processImage(left_path, left_output, config);
        if (left_rmse >= 0) {
            std::cout << left_path << ": " << std::fixed << std::setprecision(4) 
                      << left_rmse << "rmse" << std::endl;
        } else {
            std::cout << left_path << ": FAILED" << std::endl;
        }
        
        // 处理右图
        double right_rmse = processImage(right_path, right_output, config);
        if (right_rmse >= 0) {
            std::cout << right_path << ": " << std::fixed << std::setprecision(4) 
                      << right_rmse << "rmse" << std::endl;
        } else {
            std::cout << right_path << ": FAILED" << std::endl;
        }
    }
    
    std::cout << "\n所有图像处理完成！" << std::endl;
    std::cout << "结果保存在 match_results/ 目录中" << std::endl;
    
    return 0;
}
