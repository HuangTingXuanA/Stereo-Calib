/**
 * @file ellipse_detector.cpp
 * @brief 椭圆检测器实现 - 简化版
 * 
 * 基于闭合弧段的椭圆检测算法：
 * 1. ED边缘 -> 像素级弧段
 * 2. 闭合弧段筛选（曼哈顿距离判定）
 * 3. 凸性检查
 * 4. cv::fitEllipse拟合 + 几何约束初筛
 * 5. 基于ROI的Tukey Loss亚像素精修
 */

#include "ellipse_detector.h"
#include "dsf.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 检查弧段是否闭合（曼哈顿距离判定）
 * @param arc 弧段点集
 * @param thresh 曼哈顿距离阈值
 * @return 是否闭合
 */
bool isClosedArc(const std::vector<cv::Point>& arc, int thresh) {
    if (arc.size() < 10) return false;
    
    const cv::Point& start = arc.front();
    const cv::Point& end = arc.back();
    
    // 曼哈顿距离判定：|dx| + |dy| <= thresh
    int manhattanDist = std::abs(start.x - end.x) + std::abs(start.y - end.y);
    return manhattanDist <= thresh;
}

/**
 * @brief 检查弧段是否为凸曲线
 * @param arc 弧段点集
 * @return 是否为凸曲线（所有叉积同号）
 */
bool isConvexArc(const std::vector<cv::Point>& arc, double thresh = 0.50) {
    if (arc.size() < 5) return false;
    
    // 采样检查，避免对每个点都计算
    int step = std::max(1, static_cast<int>(arc.size()) / 20);
    int positiveCount = 0;
    int negativeCount = 0;
    
    for (size_t i = step; i < arc.size() - step; i += step) {
        cv::Point v1 = arc[i] - arc[i - step];
        cv::Point v2 = arc[i + step] - arc[i];
        
        int cross = v1.x * v2.y - v1.y * v2.x;
        
        if (cross > 0) positiveCount++;
        else if (cross < 0) negativeCount++;
    }
    
    // 允许少量噪声
    int total = positiveCount + negativeCount;
    if (total == 0) return false;
    
    double ratio = static_cast<double>(std::max(positiveCount, negativeCount)) / total;
    return ratio > thresh;
}

/**
 * @brief 双线性插值获取亚像素位置的值
 */
template<typename T>
double bilinearInterp(const cv::Mat& img, double x, double y) {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = x0 + 1, y1 = y0 + 1;
    
    if (x0 < 0 || y0 < 0 || x1 >= img.cols || y1 >= img.rows) {
        return 0.0;
    }
    
    double dx = x - x0, dy = y - y0;
    return img.at<T>(y0, x0) * (1-dx) * (1-dy) +
           img.at<T>(y0, x1) * dx * (1-dy) +
           img.at<T>(y1, x0) * (1-dx) * dy +
           img.at<T>(y1, x1) * dx * dy;
}

/**
 * @brief 沿径向方向搜索边界位置，纠正边缘点（自适应极性）
 * 
 * 自动检测椭圆极性（内亮外暗 vs 内暗外亮），选择正确的边界位置：
 * - 内亮外暗：从亮到暗的过渡，阈值接近暗侧（外侧 25%）
 * - 内暗外亮：从暗到亮的过渡，阈值接近暗侧（内侧 75%）
 * 
 * @param gray 灰度图像
 * @param center 椭圆中心
 * @param edgePoint 原始边缘点
 * @param searchRange 搜索范围（像素）
 * @return 纠正后的亚像素坐标
 */
cv::Point2f refineEdgePointGray(const cv::Mat& gray,
                                 const cv::Point2f& center,
                                 const cv::Point& edgePoint,
                                 const EllipseParams& ep) {
    double searchRange = ep.radialSearchRange;
    // 径向方向（从中心指向边缘点）
    cv::Point2f dir(edgePoint.x - center.x, edgePoint.y - center.y);
    double len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
    if (len < 1e-6) return cv::Point2f(edgePoint);
    dir.x /= len;
    dir.y /= len;
    
    // 沿径向采样灰度值
    std::vector<double> samples;
    std::vector<double> positions;
    double minG = 1e9, maxG = -1e9;
    
    for (double t = -searchRange; t <= searchRange; t += ep.radialSearchStep) {
        double x = edgePoint.x + t * dir.x;
        double y = edgePoint.y + t * dir.y;
        
        double g = bilinearInterp<uchar>(gray, x, y);
        if (g > 0) {
            samples.push_back(g);
            positions.push_back(t);
            if (g < minG) { minG = g; }
            if (g > maxG) { maxG = g; }
        }
    }
    
    if (samples.size() < 3 || (maxG - minG) < 10) {
        return cv::Point2f(edgePoint);  // 对比度太低，保持原位置
    }
    
    // 检测极性：比较内侧（负t）和外侧（正t）的平均灰度
    double innerSum = 0, outerSum = 0;
    int innerCnt = 0, outerCnt = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        if (positions[i] < 0) { innerSum += samples[i]; innerCnt++; }
        else { outerSum += samples[i]; outerCnt++; }
    }
    double innerAvg = innerCnt > 0 ? innerSum / innerCnt : 0;
    double outerAvg = outerCnt > 0 ? outerSum / outerCnt : 0;
    
    // 计算对比度（灰度范围）
    double contrast = maxG - minG;
    
    // 根据极性和对比度选择阈值
    // 对比度低时使用更靠外的阈值（因为边缘模糊带更宽）
    double threshold;
    bool innerBright = (innerAvg > outerAvg);
    
    // 基础阈值百分比，根据对比度调整
    double baseRatio;
    if (innerBright) {
        // 内亮外暗：对比度高时17%，对比度低时用更小的值（更靠外）
        baseRatio = (contrast > ep.contrastHigh) ? ep.ratioInnerBrightHigh : 
                    (contrast > ep.contrastMid) ? ep.ratioInnerBrightMid : ep.ratioInnerBrightLow;
        threshold = minG + (maxG - minG) * baseRatio;
    } else {
        // 内暗外亮：对比度高时75%，对比度低时用更大的值（更靠外）
        baseRatio = (contrast > ep.contrastHigh) ? ep.ratioInnerDarkHigh : 
                    (contrast > ep.contrastMid) ? ep.ratioInnerDarkMid : ep.ratioInnerDarkLow;
        threshold = minG + (maxG - minG) * baseRatio;
    }
    
    // 从外向内搜索，找到灰度穿越阈值的位置
    for (size_t i = samples.size() - 1; i > 0; i--) {
        bool cross = (samples[i] < threshold && samples[i-1] >= threshold) ||
                     (samples[i] >= threshold && samples[i-1] < threshold);
        if (cross) {
            double ratio = (threshold - samples[i-1]) / (samples[i] - samples[i-1]);
            double t = positions[i-1] + ratio * (positions[i] - positions[i-1]);
            return cv::Point2f(edgePoint.x + t * dir.x, edgePoint.y + t * dir.y);
        }
    }
    
    return cv::Point2f(edgePoint);
}


/**

 * @brief 计算点到椭圆的几何距离（正交距离）
 * 
 * 使用迭代法找到椭圆上离给定点最近的点。
 * 椭圆参数方程: (a*cos(t), b*sin(t))
 * 
 * @param px, py 点坐标（椭圆局部坐标系，已减去中心并反旋转）
 * @param a, b 椭圆半轴
 * @return 几何距离（有符号：正=点在椭圆外，负=点在椭圆内）
 */
double pointToEllipseDistance(double px, double py, double a, double b) {
    // 处理退化情况
    if (a < 1e-6 || b < 1e-6) return std::sqrt(px * px + py * py);
    
    // 利用对称性，将点映射到第一象限
    double x = std::abs(px);
    double y = std::abs(py);
    
    // 初始估计：参数 t 使得椭圆点 (a*cos(t), b*sin(t)) 接近目标点
    double t = std::atan2(a * y, b * x);
    
    // Newton-Raphson 迭代求解最近点
    for (int iter = 0; iter < 10; iter++) {
        double cosT = std::cos(t);
        double sinT = std::sin(t);
        
        // 椭圆上的点
        double ex = a * cosT;
        double ey = b * sinT;
        
        // 椭圆切向量
        double tx = -a * sinT;
        double ty = b * cosT;
        
        // 从椭圆点到目标点的向量
        double dx = x - ex;
        double dy = y - ey;
        
        // 切向投影（应该为0才是最近点）
        double dot = dx * tx + dy * ty;
        
        // 切向量模长的平方
        double tLen2 = tx * tx + ty * ty;
        
        if (tLen2 < 1e-12) break;
        
        // 更新 t
        double dt = dot / tLen2;
        t += dt;
        
        // 收敛检查
        if (std::abs(dt) < 1e-8) break;
    }
    
    // 计算最终距离
    double cosT = std::cos(t);
    double sinT = std::sin(t);
    double ex = a * cosT;
    double ey = b * sinT;
    
    double dist = std::sqrt((x - ex) * (x - ex) + (y - ey) * (y - ey));
    
    // 判断点在椭圆内还是外
    double ellipseVal = (px * px) / (a * a) + (py * py) / (b * b);
    if (ellipseVal < 1.0) {
        dist = -dist;  // 点在椭圆内部
    }
    
    return dist;
}

/**
 * @brief 使用几何距离 + Tukey Loss + Gauss-Newton 进行椭圆精修
 * 
 * 流程：
 * 1. 对边缘点进行径向梯度极值纠正（统一边缘位置）
 * 2. 使用几何距离进行 LM 优化
 * 
 * @param image 灰度图像
 * @param initEll 初始椭圆
 * @param arc 边缘点集
 * @param maxIter 最大迭代次数
 * @param tukeyC Tukey Loss 参数
 * @return 精修后的椭圆
 */
cv::RotatedRect geometricRefinement(const cv::Mat& image,
                                     const cv::RotatedRect& initEll,
                                     const std::vector<cv::Point>& arc,
                                     const EllipseParams& ep) {
    if (arc.size() < 5) return initEll;
    
    // Step 1: 对边缘点进行径向灰度百分位纠正
    std::vector<cv::Point2f> points;
    points.reserve(arc.size());
    for (const auto& p : arc) {
        cv::Point2f refined = refineEdgePointGray(image, initEll.center, p, ep);
        points.push_back(refined);
    }
    int n = static_cast<int>(points.size());
    
    // Step 2: 椭圆参数初始化
    double params[5] = {
        initEll.center.x,
        initEll.center.y,
        initEll.size.width / 2.0,
        initEll.size.height / 2.0,
        initEll.angle * CV_PI / 180.0
    };
    
    // 确保 a >= b
    if (params[2] < params[3]) {
        std::swap(params[2], params[3]);
        params[4] += CV_PI / 2.0;
    }
    
    // Step 3: Gauss-Newton 迭代
    double prevError = 1e30;
    int maxIter = ep.refineMaxIter;
    double tukeyC = ep.tukeyConstant;
    
    for (int iter = 0; iter < maxIter; iter++) {
        double cx = params[0], cy = params[1];
        double a = params[2], b = params[3];
        double theta = params[4];
        double cosT = std::cos(theta);
        double sinT = std::sin(theta);
        
        if (a < 1 || b < 1) break;
        
        // 构建 Jacobian 矩阵 J 和残差向量 r
        // 使用 Tukey Loss 加权
        cv::Mat J(n, 5, CV_64F, cv::Scalar(0));
        cv::Mat r(n, 1, CV_64F, cv::Scalar(0));
        cv::Mat W(n, n, CV_64F, cv::Scalar(0));  // 权重对角矩阵
        
        double totalError = 0;
        int validCount = 0;
        
        for (int i = 0; i < n; i++) {
            // 变换到椭圆局部坐标系
            double dx = points[i].x - cx;
            double dy = points[i].y - cy;
            double lx = dx * cosT + dy * sinT;
            double ly = -dx * sinT + dy * cosT;
            
            // 计算几何距离
            double dist = pointToEllipseDistance(lx, ly, a, b);
            r.at<double>(i, 0) = dist;
            
            // Tukey 权重
            double absD = std::abs(dist);
            double weight = 0.0;
            if (absD < tukeyC) {
                double ratio = absD / tukeyC;
                double tmp = 1.0 - ratio * ratio;
                weight = tmp * tmp;
                validCount++;
            }
            W.at<double>(i, i) = weight;
            
            totalError += absD * weight;
            
            // 数值求导计算 Jacobian（中心差分）
            double eps = ep.jacobianEps;
            for (int j = 0; j < 5; j++) {
                double orig = params[j];
                
                params[j] = orig + eps;
                double cx2 = params[0], cy2 = params[1];
                double a2 = params[2], b2 = params[3];
                double theta2 = params[4];
                double cosT2 = std::cos(theta2);
                double sinT2 = std::sin(theta2);
                double dx2 = points[i].x - cx2;
                double dy2 = points[i].y - cy2;
                double lx2 = dx2 * cosT2 + dy2 * sinT2;
                double ly2 = -dx2 * sinT2 + dy2 * cosT2;
                double dist_plus = pointToEllipseDistance(lx2, ly2, a2, b2);
                
                params[j] = orig - eps;
                cx2 = params[0]; cy2 = params[1];
                a2 = params[2]; b2 = params[3];
                theta2 = params[4];
                cosT2 = std::cos(theta2);
                sinT2 = std::sin(theta2);
                dx2 = points[i].x - cx2;
                dy2 = points[i].y - cy2;
                lx2 = dx2 * cosT2 + dy2 * sinT2;
                ly2 = -dx2 * sinT2 + dy2 * cosT2;
                double dist_minus = pointToEllipseDistance(lx2, ly2, a2, b2);
                
                params[j] = orig;
                
                J.at<double>(i, j) = (dist_plus - dist_minus) / (2 * eps);
            }
        }
        
        // 收敛检查
        if (validCount < 5 || totalError >= prevError * 0.999) break;
        prevError = totalError;
        
        // 解正规方程: (J^T W J) delta = -J^T W r
        cv::Mat JtW = J.t() * W;
        cv::Mat A = JtW * J;
        cv::Mat b_vec = -JtW * r;
        
        // 添加 LM 阻尼项
        double lambda = ep.lmLambda;
        for (int i = 0; i < 5; i++) {
            A.at<double>(i, i) *= (1.0 + lambda);
        }
        
        cv::Mat delta;
        if (!cv::solve(A, b_vec, delta, cv::DECOMP_SVD)) break;
        
        // 更新参数
        for (int j = 0; j < 5; j++) {
            params[j] += delta.at<double>(j, 0);
        }
        
        // 确保 a >= b
        if (params[2] < params[3]) {
            std::swap(params[2], params[3]);
            params[4] += CV_PI / 2.0;
        }
        
        // 角度归一化到 [0, 180)
        while (params[4] < 0) params[4] += CV_PI;
        while (params[4] >= CV_PI) params[4] -= CV_PI;
    }
    
    // 构建返回椭圆
    cv::RotatedRect result;
    result.center.x = static_cast<float>(params[0]);
    result.center.y = static_cast<float>(params[1]);
    result.size.width = static_cast<float>(params[2] * 2);
    result.size.height = static_cast<float>(params[3] * 2);
    result.angle = static_cast<float>(params[4] * 180.0 / CV_PI);
    
    return result;
}



/**
 * @brief 椭圆检测器内部实现类
 */
class EllipseDetectorImpl {
public:
    EllipseDetectorImpl(const cv::Mat& image, const DetectorParams& params);
    std::vector<Ellipse> detect();

private:
    // 核心检测流程
    void filterClosedArcs();
    void fitAndValidate();
    void refineEllipses();
    void clusterEllipses();
    
    // 调试辅助
    void writeDebugImages();
    
private:
    DetectorParams params_;
    int width_, height_;
    cv::Mat smoothImage_;
    cv::Mat srcImage_;
    
    // 边缘数据
    std::vector<std::vector<cv::Point>> edges_;
    
    // 闭合弧段
    std::vector<std::vector<cv::Point>> closedArcs_;
    
    // 候选椭圆
    std::vector<cv::RotatedRect> candidates_;
    std::vector<int> candidateArcIdx_;
    
    // 最终椭圆
    std::vector<cv::RotatedRect> finalEllipses_;
    std::vector<double> finalScores_;
};

// ============================================================================
// 椭圆检测器实现
// ============================================================================

EllipseDetectorImpl::EllipseDetectorImpl(const cv::Mat& image, const DetectorParams& params)
    : params_(params) {
    
    srcImage_ = image.clone();
    
    // 1. 边缘检测
    EdgeDetector edgeDetector(image, params.edge);
    edges_ = edgeDetector.getSegments();
    smoothImage_ = edgeDetector.getSmoothImage();
    width_ = edgeDetector.getWidth();
    height_ = edgeDetector.getHeight();
}

std::vector<Ellipse> EllipseDetectorImpl::detect() {
    if (edges_.empty()) {
        return {};
    }
    
    // 2. 筛选闭合弧段
    filterClosedArcs();
    
    // 3. 拟合与验证
    fitAndValidate();
    
    // 4. 亚像素精修
    refineEllipses();
    
    // 5. 聚类去重
    clusterEllipses();
    
    // 调试输出
    if (params_.debug) {
        writeDebugImages();
    }
    
    // 转换为输出格式
    std::vector<Ellipse> result;
    result.reserve(finalEllipses_.size());
    
    for (size_t i = 0; i < finalEllipses_.size(); i++) {
        result.emplace_back(finalEllipses_[i], finalScores_[i]);
    }
    
    return result;
}

void EllipseDetectorImpl::filterClosedArcs() {
    for (const auto& edge : edges_) {
        // 闭合检测：参数化的曼哈顿距离判定
        if (!isClosedArc(edge, params_.arc.closedDistThresh)) continue;
        
        // 这一步仅做闭合筛选，用于生成 diag_2
        closedArcs_.push_back(edge);
    }
}

void EllipseDetectorImpl::fitAndValidate() {
    for (size_t i = 0; i < closedArcs_.size(); i++) {
        const auto& arc = closedArcs_[i];
        
        // 最小长度检查
        if (static_cast<int>(arc.size()) < params_.arc.minArcLength) continue;
        
        // 凸性检查（适配噪声，略微放宽阈值）
        if (!isConvexArc(arc, params_.arc.convexRatioThresh)) continue;

        if (arc.size() < 5) continue;
        
        // 椭圆拟合
        cv::RotatedRect ell = cv::fitEllipse(arc);
        
        // 几何约束检查
        double a = ell.size.width / 2.0;
        double b = ell.size.height / 2.0;
        
        // 尺寸约束
        if (std::min(a, b) < params_.ellipse.minMinorAxis) continue;
        if (std::max(a, b) > std::max(width_, height_)) continue;
        
        // 长短轴比约束
        double aspectRatio = std::min(a, b) / std::max(a, b);
        if (aspectRatio < params_.ellipse.minAspectRatio) continue;
        
        // 中心点在图像内
        if (ell.center.x < 0 || ell.center.x >= width_ ||
            ell.center.y < 0 || ell.center.y >= height_) continue;
        
        // 拟合误差检查：计算点到椭圆的平均距离
        double theta = -ell.angle * CV_PI / 180.0;
        double cosT = std::cos(theta);
        double sinT = std::sin(theta);
        double invA2 = 1.0 / (a * a);
        double invB2 = 1.0 / (b * b);
        
        int inlierCount = 0;
        double inlierDist2 = params_.ellipse.inlierDist * params_.ellipse.inlierDist;
        
        for (const auto& p : arc) {
            cv::Point2f tp = cv::Point2f(p) - ell.center;
            double rx = tp.x * cosT - tp.y * sinT;
            double ry = tp.x * sinT + tp.y * cosT;
            double h = (rx * rx) * invA2 + (ry * ry) * invB2;
            
            // 近似几何距离
            double d2 = (tp.x * tp.x + tp.y * tp.y) * std::pow(h - 1.0, 2) * 0.25;
            
            if (d2 < inlierDist2) inlierCount++;
        }
        
        double inlierRatio = static_cast<double>(inlierCount) / arc.size();
        if (inlierRatio < params_.ellipse.minEllipseScore1) continue;
        
        candidates_.push_back(ell);
        candidateArcIdx_.push_back(static_cast<int>(i));
    }
}

void EllipseDetectorImpl::refineEllipses() {
    finalEllipses_.clear();
    finalScores_.clear();
    
    for (size_t i = 0; i < candidates_.size(); i++) {
        const auto& arc = closedArcs_[candidateArcIdx_[i]];
        
        // 几何距离精修（使用正交距离 + 径向灰度百分位搜寻）
        cv::RotatedRect refined = geometricRefinement(srcImage_, candidates_[i], arc, params_.ellipse);
        
        // 精修后再次验证几何约束
        double a = refined.size.width / 2.0;
        double b = refined.size.height / 2.0;
        
        if (std::min(a, b) < params_.ellipse.minMinorAxis) continue;
        
        double aspectRatio = std::min(a, b) / std::max(a, b);
        if (aspectRatio < params_.ellipse.minAspectRatio) continue;
        
        // 计算覆盖率作为质量分数
        double perimeter = CV_PI * (3 * (a + b) - std::sqrt((3 * a + b) * (a + 3 * b)));
        double coverage = static_cast<double>(arc.size()) / perimeter;
        
        finalEllipses_.push_back(refined);
        finalScores_.push_back(std::min(1.0, coverage));
    }
}


void EllipseDetectorImpl::clusterEllipses() {
    if (finalEllipses_.empty()) return;
    
    // NMS: 对相近的椭圆进行非极大值抑制
    std::vector<int> inDegree(finalEllipses_.size(), 0);
    
    for (size_t i = 0; i < finalEllipses_.size(); i++) {
        const auto& ell1 = finalEllipses_[i];
        for (size_t j = i + 1; j < finalEllipses_.size(); j++) {
            const auto& ell2 = finalEllipses_[j];
            
            double dist = std::sqrt(
                std::pow(ell1.center.x - ell2.center.x, 2) +
                std::pow(ell1.center.y - ell2.center.y, 2) +
                std::pow(ell1.size.width - ell2.size.width, 2) +
                std::pow(ell1.size.height - ell2.size.height, 2));
            
            if (dist < params_.ellipse.clusterDist) {
                if (finalScores_[j] < finalScores_[i]) {
                    inDegree[j]++;
                } else {
                    inDegree[i]++;
                }
            }
        }
    }
    
    // 保留入度为0的椭圆
    std::vector<cv::RotatedRect> filtered;
    std::vector<double> filteredScores;
    
    for (size_t i = 0; i < finalEllipses_.size(); i++) {
        if (inDegree[i] == 0) {
            filtered.push_back(finalEllipses_[i]);
            filteredScores.push_back(finalScores_[i]);
        }
    }
    
    finalEllipses_ = std::move(filtered);
    finalScores_ = std::move(filteredScores);
}

void EllipseDetectorImpl::writeDebugImages() {
    namespace fs = std::filesystem;
    std::string diagDir = "debug_diag";
    fs::create_directories(diagDir);
    
    cv::Mat color;
    cv::cvtColor(smoothImage_, color, cv::COLOR_GRAY2BGR);
    
    // 1. diag_1_edges.png: 显示所有边缘，并标记首尾端点 (使用 .at)
    cv::Mat edgeMap = color.clone();
    for (const auto& edge : edges_) {
        cv::Scalar c(rand() % 255, rand() % 255, rand() % 255);
        cv::Vec3b pixelColor((uchar)c[0], (uchar)c[1], (uchar)c[2]);
        
        for (const auto& p : edge) {
            if (p.x >= 0 && p.x < width_ && p.y >= 0 && p.y < height_) {
                edgeMap.at<cv::Vec3b>(p.y, p.x) = pixelColor;
            }
        }
        
        // 标记端点：首点绿色，尾点红色
        if (!edge.empty()) {
            cv::Point pStart = edge.front();
            cv::Point pEnd = edge.back();
            if (pStart.x >= 0 && pStart.x < width_ && pStart.y >= 0 && pStart.y < height_)
                edgeMap.at<cv::Vec3b>(pStart.y, pStart.x) = cv::Vec3b(0, 255, 0);
            if (pEnd.x >= 0 && pEnd.x < width_ && pEnd.y >= 0 && pEnd.y < height_)
                edgeMap.at<cv::Vec3b>(pEnd.y, pEnd.x) = cv::Vec3b(0, 0, 255);
        }
    }
    cv::imwrite(diagDir + "/diag_1_edges.png", edgeMap);
    
    // 2. diag_2_closed_arcs.png: 仅显示符合闭合条件的弧段 (使用 .at)
    cv::Mat closedMap = color.clone();
    for (const auto& arc : closedArcs_) {
        cv::Scalar c(rand() % 255, rand() % 255, rand() % 255);
        cv::Vec3b pixelColor((uchar)c[0], (uchar)c[1], (uchar)c[2]);
        
        for (const auto& p : arc) {
            if (p.x >= 0 && p.x < width_ && p.y >= 0 && p.y < height_) {
                closedMap.at<cv::Vec3b>(p.y, p.x) = pixelColor;
            }
        }
    }
    cv::imwrite(diagDir + "/diag_2_closed_arcs.png", closedMap);
    
    // 3. 候选椭圆
    cv::Mat candidateMap = color.clone();
    for (const auto& ell : candidates_) {
        cv::ellipse(candidateMap, ell, cv::Scalar(255, 0, 0), 1);
    }
    cv::imwrite(diagDir + "/diag_3_candidates.png", candidateMap);
    
    // 4. 最终椭圆
    cv::Mat finalMap = color.clone();
    for (const auto& ell : finalEllipses_) {
        cv::ellipse(finalMap, ell, cv::Scalar(0, 255, 0), 2);
        cv::circle(finalMap, ell.center, 2, cv::Scalar(0, 0, 255), -1);
    }
    cv::imwrite(diagDir + "/diag_4_final.png", finalMap);
    
    // 5. 边缘点灰度值和梯度模长诊断
    // 随机抽取100个边缘点（来自不同的候选椭圆），输出到CSV文件
    {
        // 计算梯度
        cv::Mat gradX, gradY;
        cv::Sobel(srcImage_, gradX, CV_16S, 1, 0, 3);
        cv::Sobel(srcImage_, gradY, CV_16S, 0, 1, 3);
        
        // 收集所有候选椭圆的边缘点
        std::vector<std::tuple<int, cv::Point, int>> allEdgePoints; // (椭圆ID, 点, 弧段索引)
        for (size_t i = 0; i < candidateArcIdx_.size() && i < candidates_.size(); i++) {
            const auto& arc = closedArcs_[candidateArcIdx_[i]];
            for (const auto& p : arc) {
                allEdgePoints.push_back({static_cast<int>(i), p, static_cast<int>(candidateArcIdx_[i])});
            }
        }
        
        // 随机采样100个点
        std::vector<size_t> sampleIdx;
        if (allEdgePoints.size() <= 100) {
            for (size_t i = 0; i < allEdgePoints.size(); i++) sampleIdx.push_back(i);
        } else {
            std::srand(42);  // 固定种子保证可复现
            while (sampleIdx.size() < 100) {
                size_t idx = std::rand() % allEdgePoints.size();
                if (std::find(sampleIdx.begin(), sampleIdx.end(), idx) == sampleIdx.end()) {
                    sampleIdx.push_back(idx);
                }
            }
        }
        
        // 输出CSV文件
        std::ofstream csv(diagDir + "/diag_5_edge_points.csv");
        csv << "ellipse_id,arc_idx,x,y,gray,grad_x,grad_y,grad_mag,dist_to_center,ellipse_cx,ellipse_cy,ellipse_a,ellipse_b,ellipse_angle\n";
        
        for (size_t idx : sampleIdx) {
            auto& [ellId, pt, arcIdx] = allEdgePoints[idx];
            
            if (pt.x < 0 || pt.y < 0 || pt.x >= width_ || pt.y >= height_) continue;
            
            int gray = srcImage_.at<uchar>(pt.y, pt.x);
            short gx = gradX.at<short>(pt.y, pt.x);
            short gy = gradY.at<short>(pt.y, pt.x);
            double gradMag = std::sqrt(gx * gx + gy * gy);
            
            // 获取对应的椭圆参数
            const auto& ell = candidates_[ellId];
            double distToCenter = std::sqrt(
                (pt.x - ell.center.x) * (pt.x - ell.center.x) + 
                (pt.y - ell.center.y) * (pt.y - ell.center.y)
            );
            
            csv << ellId << "," << arcIdx << "," << pt.x << "," << pt.y << ","
                << gray << "," << gx << "," << gy << "," << gradMag << ","
                << distToCenter << "," << ell.center.x << "," << ell.center.y << ","
                << ell.size.width/2 << "," << ell.size.height/2 << "," << ell.angle << "\n";
        }
        csv.close();
        
        // 在图像上可视化这些采样点
        cv::Mat sampleMap = color.clone();
        for (size_t idx : sampleIdx) {
            auto& [ellId, pt, arcIdx] = allEdgePoints[idx];
            if (pt.x < 0 || pt.y < 0 || pt.x >= width_ || pt.y >= height_) continue;
            cv::circle(sampleMap, pt, 3, cv::Scalar(0, 0, 255), -1);  // 红点标记采样点
        }
        cv::imwrite(diagDir + "/diag_5_sampled_edge_points.png", sampleMap);
        
        std::cout << "已输出边缘点诊断数据: " << diagDir << "/diag_5_edge_points.csv" << std::endl;
        std::cout << "采样点数量: " << sampleIdx.size() << std::endl;
    }
}


} // anonymous namespace

// ============================================================================
// 公开接口
// ============================================================================

std::vector<Ellipse> detectEllipses(const cv::Mat& image, const DetectorParams& params) {
    EllipseDetectorImpl detector(image, params);
    return detector.detect();
}
