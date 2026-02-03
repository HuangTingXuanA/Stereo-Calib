// 1. 存储原始图像
// 2. 添加 refineEllipses 方法声明
// 3. 在 detect() 中调用 refineEllipses
// 4. 实现 refineEllipses 逻辑

#include "ellipse_detector.hpp"
#include "dsf.hpp"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// 内部数据结构
// ============================================================================

namespace {

/**
 * @brief 带权边结构
 */
struct WeightedEdge {
    int id;              // 目标圆弧ID
    cv::Vec3f score;     // 评分 [内点数, 内点率, 覆盖率]
    
    WeightedEdge(int _id, const cv::Vec3f& _score) : id(_id), score(_score) {}
    
    bool operator<(const WeightedEdge& e) const {
        return score[0] > e.score[0]; // 按内点数降序
    }
};

/**
 * @brief 集合节点（并查集）
 */
struct SetNode {
    int rootId;                      // 根节点ID
    int broRootId;                   // 兄弟根ID
    std::vector<WeightedEdge> wq;    // 带权边队列
    std::vector<int> setElements;    // 集合元素
    cv::Vec3f score;                 // 评分
    std::vector<int> bro;            // 兄弟节点列表
    cv::RotatedRect ell;             // 拟合椭圆
    
    SetNode() : rootId(0), broRootId(0) {}
    
    SetNode(int _rootId, int _broRootId, const std::vector<int>& _setElements,
            const cv::RotatedRect& _ell, const cv::Vec3f& _score)
        : rootId(_rootId), broRootId(_broRootId), setElements(_setElements),
          score(_score), ell(_ell) {}
};

/**
 * @brief 椭圆检测器内部实现类
 */
class EllipseDetectorImpl {
public:
    EllipseDetectorImpl(const cv::Mat& image, const DetectorParams& params);
    
    std::vector<Ellipse> detect();

private:
    // 核心算法步骤
    void buildWeightedEdge();
    void initUnionFind();
    void arcMatchingByUnionFind();
    void ellipseCluster();
    void refineEllipses(); // 二次细化
    
    // 工具函数
    int findRoot(int k);
    bool canFromWeightedPair(int id1, int id2);
    bool canMatch(int id1, int id2);
    bool canMerge(int id1, int id2);
    bool check(const std::vector<cv::Point>& lv);
    cv::RotatedRect fit(const std::vector<int>& ids);
    cv::Vec3f interiorRate(const std::vector<int>& ids, const cv::RotatedRect& ell);
    
private:
    DetectorParams params_;
    int width_, height_;
    cv::Mat smoothImage_;
    cv::Mat originalImage_; // 存储用于细化的原始图像
    
    // 圆弧数据
    std::vector<std::vector<cv::Point>> arcs_;
    std::vector<std::vector<std::pair<int, int>>> arcSegs_;
    std::vector<int> polarities_;
    
    // 并查集数据
    std::vector<SetNode> setNodes_;
    
    // 椭圆候选
    std::vector<cv::Vec3f> ellipseScore_;
    std::vector<cv::RotatedRect> ellipseList_;
    std::vector<std::vector<int>> ellipseArcId_;
    
    // 聚类后的椭圆
    std::vector<cv::Vec3f> clusteredEllipseScore_;
    std::vector<cv::RotatedRect> clusteredEllipse_;
    std::vector<std::vector<int>> clusteredEllipseArcId_;
};

// ============================================================================
// 梯度验证并行处理类
// ============================================================================

class ParallelComputeMatchScore : public cv::ParallelLoopBody {
public:
    ParallelComputeMatchScore(cv::RotatedRect* ellipseList,
                              std::vector<int>* remainId,
                              std::vector<double>* remainScore,
                              const cv::Mat2f& direction,
                              int sampleNum, int radius,
                              int width, int height,
                              double remainScoreThresh,
                              int polarity,
                              int centerIntensityThresh,
                              int brightCenterThresh,
                              int darkCenterThresh,
                              double cosGradAngle,
                              const cv::Mat& image,
                              int nums, int threads)
        : ellipseList_(ellipseList), remainId_(remainId), remainScore_(remainScore),
          direction_(direction), sampleNum_(sampleNum), radius_(radius),
          width_(width), height_(height), remainScoreThresh_(remainScoreThresh),
          polarity_(polarity),
          nums_(nums), threads_(threads),    
          centerIntensityThresh_(centerIntensityThresh),
          brightCenterThresh_(brightCenterThresh),
          darkCenterThresh_(darkCenterThresh),
          cosGradAngle_(cosGradAngle),
          image_(image) {
        
        sinAlpha_.resize(sampleNum);
        cosAlpha_.resize(sampleNum);
        
        for (int i = 0; i < sampleNum; i++) {
            double rad = i * CV_2PI / sampleNum;
            sinAlpha_[i] = std::sin(rad);
            cosAlpha_[i] = std::cos(rad);
        }
    }
    
    void operator()(const cv::Range& r) const override {
        int range = nums_ / threads_;
        
        int vEnd = r.end * range;
        if (r.end == threads_) {
            vEnd = nums_;
        }
        
        std::vector<int>* id = &remainId_[r.start];
        std::vector<double>* score = &remainScore_[r.start];
        
        for (int v = r.start * range; v < vEnd; v++) {
            cv::RotatedRect& ell = ellipseList_[v];
            
            // --- 中心亮度约束 ---
            if (centerIntensityThresh_ != 0) {
                 int cx = cvRound(ell.center.x);
                 int cy = cvRound(ell.center.y);
                 if (cx >= 0 && cx < width_ && cy >= 0 && cy < height_) {
                     uchar val = image_.at<uchar>(cy, cx);
                     if (polarity_ == 1) { // 亮椭圆 -> 中心应该是亮的
                         if (val <= brightCenterThresh_) continue;
                     } else if (polarity_ == -1) { // 暗椭圆 -> 中心应该是暗的
                         if (val >= darkCenterThresh_) continue;
                     }
                 }
            }
            // -----------------------------------

            float b = ell.size.height * 0.5f;
            float a = ell.size.width * 0.5f;
            float theta = -ell.angle * static_cast<float>(CV_PI) / 180.0f;
            float sinTheta = std::sin(theta);
            float cosTheta = std::cos(theta);
            float invA2 = 1.0f / (a * a);
            float invB2 = 1.0f / (b * b);
            
            std::vector<cv::Vec2f> imgGrad(sampleNum_);
            std::vector<cv::Vec2f> ellGrad(sampleNum_);
            
            for (int j = 0; j < sampleNum_; j++) {
                float cosa = a * static_cast<float>(cosAlpha_[j]);
                float cosb = b * static_cast<float>(sinAlpha_[j]);
                int xi = static_cast<int>(cosTheta * cosa + sinTheta * cosb + ell.center.x);
                int yi = static_cast<int>(-sinTheta * cosa + cosTheta * cosb + ell.center.y);
                
                if (xi < 0 || xi >= width_ || yi < 0 || yi >= height_) {
                    imgGrad[j] = cv::Vec2f(1, 0);
                    ellGrad[j] = cv::Vec2f(1, 0);
                    continue;
                }
                
                cv::Point p(xi, yi);
                imgGrad[j] = direction_(p);
                
                cv::Point tp = cv::Point2f(p) - ell.center;
                float rx = (tp.x * cosTheta - tp.y * sinTheta);
                float ry = (tp.x * sinTheta + tp.y * cosTheta);
                cv::Vec2f rdir(2 * rx * cosTheta * invA2 + 2 * ry * sinTheta * invB2,
                               2 * rx * (-sinTheta) * invA2 + 2 * ry * cosTheta * invB2);
                rdir /= cv::norm(rdir);
                ellGrad[j] = rdir;
            }
            
            int count = 0;
            for (int j = 0; j < sampleNum_; j++) {
                cv::Vec2f cdir = imgGrad[j];
                for (int k = 1; k < radius_; k++) {
                    cdir += imgGrad[(j - k + sampleNum_) % sampleNum_];
                    cdir += imgGrad[(j + k) % sampleNum_];
                }
                cdir /= radius_ * 2 - 1;
                float norm = cv::norm(cdir);
                if (norm > 0) cdir /= norm;
                
                cv::Vec2f rdir = ellGrad[j];
                
                // 图像梯度与椭圆法线（指向外）的点积
                double dot = cdir.dot(rdir);

                bool pass = false;
                if (polarity_ == 0) {
                    // 接受平行和反向平行
                    if (std::abs(dot) > cosGradAngle_) pass = true;
                } else if (polarity_ == 1) { 
                    // 极性 1: 亮椭圆 -> 梯度指向内 -> 与法线反向
                    if (dot <= -cosGradAngle_) pass = true;
                } else if (polarity_ == -1) { 
                    // 极性 -1: 暗椭圆 -> 梯度指向外 -> 与法线同向
                    if (dot >= cosGradAngle_) pass = true;
                }

                if (pass) count++;
            }
            
            double tmp = 1.0 * count / sampleNum_;
            if (tmp > remainScoreThresh_) {
                id->emplace_back(v);
                score->emplace_back(tmp);
            }
        }
    }

private:
    cv::RotatedRect* ellipseList_;
    std::vector<int>* remainId_;
    std::vector<double>* remainScore_;
    cv::Mat2f direction_;
    int sampleNum_, radius_;
    int width_, height_;
    double remainScoreThresh_;
    int polarity_;
    int nums_, threads_;
    std::vector<double> sinAlpha_, cosAlpha_;
    int centerIntensityThresh_, brightCenterThresh_, darkCenterThresh_;
    double cosGradAngle_;
    const cv::Mat& image_;
};

// ============================================================================
// 椭圆检测器实现
// ============================================================================

EllipseDetectorImpl::EllipseDetectorImpl(const cv::Mat& image, const DetectorParams& params)
    : params_(params), originalImage_(image) {
    
    // 1. 边缘检测
    EdgeDetector edgeDetector(image, params.edge);
    auto edges = edgeDetector.getSegments();
    smoothImage_ = edgeDetector.getSmoothImage();
    width_ = edgeDetector.getWidth();
    height_ = edgeDetector.getHeight();
    
    // 2. 圆弧提取
    ArcParams arcParams = params.arc;
    arcParams.threads = params.threads;  // 使用统一线程数
    
    ArcExtractor arcExtractor(edges, arcParams);
    arcExtractor.extract();
    arcExtractor.computePolarity(smoothImage_);
    arcs_ = arcExtractor.getArcs();
    arcSegs_ = arcExtractor.getArcSegs();
    polarities_ = arcExtractor.getPolarities();
}

std::vector<Ellipse> EllipseDetectorImpl::detect() {
    if (arcs_.empty()) {
        return {};
    }
    
    // 3. 构建带权边
    buildWeightedEdge();
    
    // 4. 初始化并查集
    initUnionFind();
    
    // 5. 圆弧匹配
    arcMatchingByUnionFind();
    
    // 6. 椭圆聚类与验证
    ellipseCluster();
    
    // 7. 二次拟合 (椭圆细化)
    if (params_.refine.useRefine) {
        refineEllipses();
    }
    
    // 转换为 Ellipse 结构
    std::vector<Ellipse> result;
    result.reserve(clusteredEllipse_.size());
    
    for (size_t i = 0; i < clusteredEllipse_.size(); i++) {
        double goodness = clusteredEllipseScore_[i][2];
        result.emplace_back(clusteredEllipse_[i], goodness);
    }
    
    return result;
}

void EllipseDetectorImpl::buildWeightedEdge() {
    setNodes_.resize(arcs_.size());
    
    for (size_t i = 0; i < arcs_.size(); i++) {
        if (arcSegs_[i].size() < 2) continue;
        
        for (size_t j = i + 1; j < arcs_.size(); j++) {
            if (arcSegs_[j].size() < 2) continue;
            
            if (canFromWeightedPair(static_cast<int>(i), static_cast<int>(j))) {
                std::vector<int> ids = {static_cast<int>(i), static_cast<int>(j)};
                cv::RotatedRect ell = fit(ids);
                
                std::vector<int> tmp = {static_cast<int>(j)};
                cv::Vec3f score = interiorRate(tmp, ell);
                
                if (score[1] > params_.fit.minEdgeScore) {
                    setNodes_[i].wq.emplace_back(WeightedEdge(static_cast<int>(j), score));
                }
            }
        }
    }
}

void EllipseDetectorImpl::initUnionFind() {
    int ct = 0;
    for (auto& node : setNodes_) {
        node.rootId = ct;
        node.broRootId = ct++;
        if (node.wq.empty()) continue;
        std::sort(node.wq.begin(), node.wq.end());
    }
}

int EllipseDetectorImpl::findRoot(int k) {
    return k == setNodes_[k].rootId ? k : findRoot(setNodes_[k].rootId);
}

void EllipseDetectorImpl::arcMatchingByUnionFind() {
    for (size_t i = 0; i < arcs_.size(); i++) {
        if (setNodes_[i].wq.empty()) continue;
        
        for (size_t j = 0; j < setNodes_[i].wq.size(); j++) {
            int id = setNodes_[i].wq[j].id;
            
            int r1 = findRoot(static_cast<int>(i));
            int r2 = findRoot(id);
            if (r1 == r2) continue;
            
            bool isMerge = false;
            
            for (int bro : setNodes_[i].bro) {
                int r = findRoot(bro);
                if (r2 == r) {
                    isMerge = true;
                    break;
                }
                
                bool isMatch = true;
                for (int se : setNodes_[r].setElements) {
                    if (!canMerge(setNodes_[se].broRootId, id)) {
                        isMatch = false;
                        break;
                    }
                }
                
                if (isMatch) {
                    std::vector<int> ids(setNodes_[r].setElements);
                    ids.emplace_back(id);
                    cv::RotatedRect ell = fit(ids);
                    cv::Vec3f score = interiorRate(ids, ell);
                    
                    if (score[1] > params_.fit.minEllipseScore1 && score[2] > setNodes_[bro].score[2]) {
                        setNodes_[bro].setElements.emplace_back(id);
                        setNodes_[bro].ell = ell;
                        setNodes_[bro].score = score;
                        setNodes_[id].rootId = r;
                        isMerge = true;
                        break;
                    }
                }
            }
            
            if (!isMerge) {
                std::vector<int> ids = {static_cast<int>(i), id};
                cv::RotatedRect ell = fit(ids);
                cv::Vec3f score = interiorRate(ids, ell);
                
                if (score[1] > params_.fit.minEllipseScore1) {
                    if (setNodes_[i].bro.empty()) {
                        setNodes_[i].bro.emplace_back(static_cast<int>(i));
                        setNodes_[id].rootId = static_cast<int>(i);
                        setNodes_[i].ell = ell;
                        setNodes_[i].score = score;
                        setNodes_[i].setElements = {static_cast<int>(i), id};
                    } else {
                        int rootId = static_cast<int>(setNodes_.size());
                        setNodes_[i].bro.emplace_back(rootId);
                        setNodes_[id].rootId = rootId;
                        setNodes_.emplace_back(SetNode(rootId, static_cast<int>(i), ids, ell, score));
                        setNodes_.back().bro.emplace_back(rootId);
                    }
                }
            }
        }
    }
}

void EllipseDetectorImpl::ellipseCluster() {
    // 收集候选椭圆
    for (size_t i = 0; i < setNodes_.size(); i++) {
        if (!setNodes_[i].bro.empty() && setNodes_[i].score[2] > params_.fit.minEllipseScore2) {
            ellipseList_.emplace_back(setNodes_[i].ell);
            ellipseScore_.emplace_back(setNodes_[i].score);
            ellipseArcId_.emplace_back(setNodes_[i].setElements);
        }
        
        // 单圆弧椭圆（如果有足够的点）
        if (i < arcSegs_.size() && arcSegs_[i].size() >= 5) {
            std::vector<int> tmp = {static_cast<int>(i)};
            cv::RotatedRect ell = fit(tmp);
            cv::Vec3f score = interiorRate(tmp, ell);
            
            if (score[1] > params_.fit.minEllipseScore1 && score[2] > params_.fit.minEllipseScore2) {
                ellipseList_.emplace_back(ell);
                ellipseScore_.emplace_back(score);
                ellipseArcId_.emplace_back(tmp);
            }
        }
    }
    
    if (ellipseList_.empty()) return;
    
    // 计算梯度方向
    cv::Mat2f direction(smoothImage_.size(), cv::Vec2f(0, 0));
    cv::Mat1s dx, dy;
    cv::Sobel(smoothImage_, dx, CV_16S, 1, 0, 3);
    cv::Sobel(smoothImage_, dy, CV_16S, 0, 1, 3);
    
    for (int i = 0; i < smoothImage_.rows; i++) {
        for (int j = 0; j < smoothImage_.cols; j++) {
            double len = std::sqrt(dx(i, j) * dx(i, j) + dy(i, j) * dy(i, j));
            if (len != 0) {
                direction(i, j) = cv::Point2f(dx(i, j) / len, dy(i, j) / len);
            } else {
                direction(i, j) = cv::Point2f(0, 0);
            }
        }
    }
    
    // 并行梯度验证
    std::vector<std::vector<int>> remainId(params_.threads);
    std::vector<std::vector<double>> remainScore(params_.threads);
    
    // 计算梯度角度阈值的余弦值
    double cosGradAngle = std::cos(params_.fit.gradAngleThresh * CV_PI / 180.0);
    
    cv::parallel_for_(cv::Range(0, params_.threads),
                      ParallelComputeMatchScore(ellipseList_.data(),
                                                remainId.data(),
                                                remainScore.data(),
                                                direction,
                                                params_.fit.sampleNum,
                                                params_.fit.gradRadius,
                                                width_, height_,
                                                params_.fit.remainScore,
                                                params_.fit.polarity,
                                                params_.fit.centerIntensityThresh,
                                                params_.fit.brightCenterThresh,
                                                params_.fit.darkCenterThresh,
                                                cosGradAngle,
                                                smoothImage_,
                                                static_cast<int>(ellipseList_.size()),
                                                params_.threads));
    
    // 收集验证通过的椭圆
    std::vector<float> ellScore;
    for (size_t i = 0; i < remainId.size(); i++) {
        for (size_t j = 0; j < remainId[i].size(); j++) {
            int id = remainId[i][j];
            double score = remainScore[i][j];
            clusteredEllipse_.emplace_back(ellipseList_[id]);
            clusteredEllipseArcId_.emplace_back(ellipseArcId_[id]);
            clusteredEllipseScore_.emplace_back(ellipseScore_[id]);
            ellScore.emplace_back(static_cast<float>(score));
        }
    }
    
    // 参数聚类：压制重复检测
    std::vector<int> inDegree(clusteredEllipse_.size(), 0);
    for (size_t i = 0; i < clusteredEllipse_.size(); i++) {
        cv::RotatedRect& ell1 = clusteredEllipse_[i];
        for (size_t j = i + 1; j < clusteredEllipse_.size(); j++) {
            cv::RotatedRect& ell2 = clusteredEllipse_[j];
            
            float diff = std::sqrt(
                std::pow(ell1.center.x - ell2.center.x, 2) +
                std::pow(ell1.center.y - ell2.center.y, 2) +
                std::pow(ell1.size.height - ell2.size.height, 2) +
                std::pow(ell1.size.width - ell2.size.width, 2));
            
            if (diff < params_.fit.clusterDist) {
                if (ellScore[j] < ellScore[i]) {
                    inDegree[j]++;
                } else {
                    inDegree[i]++;
                }
            }
        }
    }
    
    // 保留入度为0的椭圆（非重复）
    std::vector<cv::RotatedRect> tmp;
    std::vector<cv::Vec3f> tmpScore;
    for (size_t i = 0; i < clusteredEllipse_.size(); i++) {
        if (inDegree[i] == 0) {
            tmp.emplace_back(clusteredEllipse_[i]);
            tmpScore.emplace_back(clusteredEllipseScore_[i]);
        }
    }
    clusteredEllipse_ = tmp;
    clusteredEllipseScore_ = tmpScore;
}

// ============================================================================
// 优化与亚像素工具
// ============================================================================

namespace {

// 使用 3x3 Sobel 计算梯度幅值和方向
void computeGradientROI(const cv::Mat& img, cv::Mat& mag, cv::Mat& dx, cv::Mat& dy) {
    cv::Sobel(img, dx, CV_32F, 1, 0, 3);
    cv::Sobel(img, dy, CV_32F, 0, 1, 3);
    cv::magnitude(dx, dy, mag);
}

// 抛物线插值寻找亚像素峰值
// 返回相对于中心的偏移量 [-0.5, 0.5]
float parabolicInterpolation(float y_minus, float y_0, float y_plus) {
    float denom = 2.0f * (y_minus - 2.0f * y_0 + y_plus);
    if (std::abs(denom) < 1e-5f) return 0.0f;
    return (y_minus - y_plus) / denom;
}

// 获取浮点坐标处的插值结果
float getInterpValue(const cv::Mat& img, float x, float y) {
    if (x < 0 || x >= img.cols - 1 || y < 0 || y >= img.rows - 1) return 0.0f;
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    float ax = x - x0;
    float ay = y - y0;
    
    float v00 = img.at<float>(y0, x0);
    float v10 = img.at<float>(y0, x0+1);
    float v01 = img.at<float>(y0+1, x0);
    float v11 = img.at<float>(y0+1, x0+1);
    
    return (1-ax)*(1-ay)*v00 + ax*(1-ay)*v10 + (1-ax)*ay*v01 + ax*ay*v11;
}

// 鲁棒几何距离近似 (Sampson 距离)
// f = (x/a)^2 + (y/b)^2 - 1
// Grad(f) = [2x/a^2, 2y/b^2]
// 距离近似 = |f| / |Grad(f)|
double computeSampsonDist(const cv::Point2f& p, const cv::RotatedRect& ell, double* outRes = nullptr) {
    double angleRad = ell.angle * CV_PI / 180.0;
    double cx = ell.center.x;
    double cy = ell.center.y;
    double a = ell.size.width / 2.0;
    double b = ell.size.height / 2.0;
    
    double cosA = std::cos(angleRad);
    double sinA = std::sin(angleRad);
    
    double tx = p.x - cx;
    double ty = p.y - cy;
    // 旋转到标准坐标系
    double rx = tx * cosA + ty * sinA;
    double ry = -tx * sinA + ty * cosA;
    
    double a2 = a * a;
    double b2 = b * b;
    
    double f = (rx * rx) / a2 + (ry * ry) / b2 - 1.0;
    
    double df_dx = 2 * rx / a2; // 标准坐标系下
    double df_dy = 2 * ry / b2;
    
    // 原始空间的梯度模长与旋转空间相同（旋转是等距变换）
    double gradNorm = std::sqrt(df_dx * df_dx + df_dy * df_dy);
    
    // 避免除以零
    if (gradNorm < 1e-6) gradNorm = 1e-6;
    
    if (outRes) *outRes = f / gradNorm;
    
    return std::abs(f) / gradNorm;
}

// 计算 5 个参数的解析雅可比矩阵: cx, cy, a, b, angle
// 参数矢量: [cx, cy, a, b, angle]
void computeJacobianAnalytical(const cv::Point2f& p, double* params, double* J, double& res) {
    double cx = params[0];
    double cy = params[1];
    double a = params[2];
    double b = params[3];
    double theta = params[4] * CV_PI / 180.0;
    
    double cosT = std::cos(theta);
    double sinT = std::sin(theta);
    
    double dx = p.x - cx;
    double dy = p.y - cy;
    
    double rx = dx * cosT + dy * sinT;
    double ry = -dx * sinT + dy * cosT;
    
    double a2 = a * a;
    double b2 = b * b;
    double a3 = a2 * a;
    double b3 = b2 * b;
    double a4 = a2 * a2;
    double b4 = b2 * b2;
    
    double f = (rx * rx) / a2 + (ry * ry) / b2 - 1.0;
    double gx = 2.0 * rx / a2; // 旋转框架下的 df/drx
    double gy = 2.0 * ry / b2; // 旋转框架下的 df/dry
    
    // g = ||Grad(f)||^2。由于旋转是等距变换，梯度模长在旋转前后不变。
    double g = 4.0 * (rx * rx / a4 + ry * ry / b4);
    if (g < 1e-12) g = 1e-12;
    double sqrtG = std::sqrt(g);
    
    res = f / sqrtG;
    
    // f 对参数的偏导
    double df_dcx = -gx * cosT + gy * sinT;
    double df_dcy = -gx * sinT - gy * cosT;
    double df_da = -2.0 * rx * rx / a3;
    double df_db = -2.0 * ry * ry / b3;
    double df_dtheta = (2.0 * rx * ry / a2) - (2.0 * ry * rx / b2); 
    
    // g 对参数的偏导
    double dg_dcx = (-8.0 * rx / a4) * cosT + (8.0 * ry / b4) * sinT;
    double dg_dcy = (-8.0 * rx / a4) * sinT - (8.0 * ry / b4) * cosT;
    double dg_da = -16.0 * rx * rx / (a2 * a3); // -16 * rx^2 / a^5
    double dg_db = -16.0 * ry * ry / (b2 * b3); // -16 * ry^2 / b^5
    double dg_dtheta = (8.0 * rx * ry / a4) - (8.0 * ry * rx / b4);
    
    // d(f/sqrt(g)) = (1/sqrt(g)) df - (f/2 g^(3/2)) dg
    double common = f / (2.0 * g * sqrtG);
    
    J[0] = df_dcx / sqrtG - common * dg_dcx;
    J[1] = df_dcy / sqrtG - common * dg_dcy;
    J[2] = df_da / sqrtG - common * dg_da;
    J[3] = df_db / sqrtG - common * dg_db;
    J[4] = (df_dtheta / sqrtG - common * dg_dtheta) * (CV_PI / 180.0); // 转换为角度单位的偏导
}

} // 命名空间

void EllipseDetectorImpl::refineEllipses() {
    if (clusteredEllipse_.empty()) return;

    const double convergeTol = params_.refine.convergeTol;
    const double minEllipseRadius = params_.fit.minEllipseRadius;
    const double minAspectRatio = params_.fit.minAspectRatio;
    const double maxAspectRatio = params_.fit.maxAspectRatio;

    // 预计算梯度场供校验使用
    cv::Mat2f direction(smoothImage_.size(), cv::Vec2f(0, 0));
    cv::Mat1s dX, dY;
    cv::Sobel(smoothImage_, dX, CV_16S, 1, 0, 3);
    cv::Sobel(smoothImage_, dY, CV_16S, 0, 1, 3);
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            double len = std::sqrt((double)dX(y, x)*dX(y, x) + (double)dY(y, x)*dY(y, x));
            if (len > 1e-4) direction(y, x) = cv::Vec2f((float)(dX(y, x)/len), (float)(dY(y, x)/len));
        }
    }

    std::vector<bool> validFlags(clusteredEllipse_.size(), true);

    for (size_t i = 0; i < clusteredEllipse_.size(); i++) {
        cv::RotatedRect& ell = clusteredEllipse_[i];
        cv::RotatedRect refinedEll = ell;

        // 1. 提取感兴趣区域 (ROI)
        double maxDim = std::max(ell.size.width, ell.size.height);
        int roiSize = static_cast<int>(maxDim * params_.refine.roiRatio);
        
        int r_minX = std::max(0, static_cast<int>(ell.center.x - roiSize / 2));
        int r_minY = std::max(0, static_cast<int>(ell.center.y - roiSize / 2));
        int r_maxX = std::min(width_, static_cast<int>(ell.center.x + roiSize / 2));
        int r_maxY = std::min(height_, static_cast<int>(ell.center.y + roiSize / 2));

        if (r_maxX - r_minX < params_.refine.minROISize || r_maxY - r_minY < params_.refine.minROISize) continue;

        cv::Rect roiRect(r_minX, r_minY, r_maxX - r_minX, r_maxY - r_minY);
        cv::Mat roiImg = originalImage_(roiRect);
        if (roiImg.empty()) continue;

        // 2. 梯度与亚像素提取
        cv::Mat mag, dx, dy;
        computeGradientROI(roiImg, mag, dx, dy);

        std::vector<cv::Point2f> points;
        double searchDist = std::max(8.0, params_.fit.inlierDist * 4.0); // 放宽搜索半径：防止小椭圆因初始定位偏差而丢失边缘点
        
        for (int y = 1; y < mag.rows - 1; y++) {
            for (int x = 1; x < mag.cols - 1; x++) {
                float val = mag.at<float>(y, x);
                if (val < params_.refine.minGradient) continue;
                
                float gX = dx.at<float>(y, x);
                float gY = dy.at<float>(y, x);
                float len = std::sqrt(gX*gX + gY*gY);
                if (len < 1e-4) continue;
                gX /= len; gY /= len;
                
                float vPlus = getInterpValue(mag, x + gX, y + gY);
                float vMinus = getInterpValue(mag, x - gX, y - gY);
                
                if (val >= vPlus && val >= vMinus) {
                    float delta = parabolicInterpolation(vMinus, val, vPlus);
                    float spX_global = x + delta * gX + r_minX;
                    float spY_global = y + delta * gY + r_minY;
                    
                    if (computeSampsonDist(cv::Point2f(spX_global, spY_global), ell) < searchDist) {
                        points.emplace_back(cv::Point2f(spX_global, spY_global));
                    }
                }
            }
        }

        if (static_cast<int>(points.size()) < params_.refine.minEdgePoints) continue;

        // 3. LM 优化
        double p[5] = {ell.center.x, ell.center.y, ell.size.width/2.0, ell.size.height/2.0, ell.angle};
        double lambda = params_.refine.lambdaInit;
        double bestErr = 1e30;
        
        for (int iter = 0; iter < params_.refine.maxIter; iter++) {
            int N = static_cast<int>(points.size());
            cv::Mat J(N, 5, CV_64F);
            cv::Mat r_vec(N, 1, CV_64F);
            std::vector<double> residuals(N);
            
            for (int k = 0; k < N; k++) {
                double res_val;
                double Jrow[5];
                computeJacobianAnalytical(points[k], p, Jrow, res_val);
                residuals[k] = res_val;
                r_vec.at<double>(k, 0) = res_val;
                for (int j = 0; j < 5; j++) J.at<double>(k, j) = Jrow[j];
            }
            
            std::vector<double> absRes;
            absRes.reserve(N);
            for(double v : residuals) absRes.push_back(std::abs(v));
            std::nth_element(absRes.begin(), absRes.begin() + N/2, absRes.end());
            double sigma = 1.4826 * absRes[N/2];
            double c_val = std::max(params_.refine.tukeyMinC, params_.refine.tukeyAlpha * sigma);
            
            cv::Mat W = cv::Mat::zeros(N, N, CV_64F);
            double currentTotalErr = 0;
            for (int k = 0; k < N; k++) {
                double u = std::abs(residuals[k]) / c_val;
                double w = (u <= 1.0) ? std::pow(1.0 - u*u, 2) : 0;
                W.at<double>(k, k) = w;
                currentTotalErr += w * residuals[k] * residuals[k];
            }
            
            cv::Mat Jt = J.t();
            cv::Mat JtW = Jt * W;
            cv::Mat H = JtW * J;
            cv::Mat g_vec = JtW * r_vec;
            
            for (int j = 0; j < 5; j++) H.at<double>(j, j) += lambda * H.at<double>(j, j) + 1e-6;
            
            cv::Mat delta_vec;
            if (!cv::solve(H, -g_vec, delta_vec, cv::DECOMP_CHOLESKY)) break;
            
            double p_new[5];
            for (int j = 0; j < 5; j++) p_new[j] = p[j] + delta_vec.at<double>(j, 0);
            
            // 约束检查
            double minR = std::min(p_new[2], p_new[3]);
            double maxR = std::max(p_new[2], p_new[3]);
            if (minR < minEllipseRadius || maxR > std::max(width_, height_) || 
                (maxR/minR) > maxAspectRatio || (minR/maxR) < minAspectRatio) {
                lambda *= 10;
                continue;
            }

            if (currentTotalErr < bestErr) {
                bestErr = currentTotalErr;
                for (int j = 0; j < 5; j++) p[j] = p_new[j];
                lambda /= 10;
                if (lambda < params_.refine.lambdaMin) lambda = params_.refine.lambdaMin;
            } else {
                lambda *= 10;
            }
            
            if (cv::norm(delta_vec) < convergeTol) break;
        }

        refinedEll = cv::RotatedRect(cv::Point2f(p[0],p[1]), cv::Size2f(p[2]*2, p[3]*2), p[4]);
        
        // 最终合理性检查与二次质量验证 (First Principles: 拟合必须与实际梯度场一致)
        double finMinR = std::min(refinedEll.size.width, refinedEll.size.height) / 2.0;
        double finMaxR = std::max(refinedEll.size.width, refinedEll.size.height) / 2.0;
        
        bool geometricValid = (finMinR >= minEllipseRadius) && (finMinR/finMaxR >= minAspectRatio) && 
                              (finMaxR/finMinR <= maxAspectRatio) &&
                              (refinedEll.center.x >= 0 && refinedEll.center.x < width_) && 
                              (refinedEll.center.y >= 0 && refinedEll.center.y < height_);
        
        if (!geometricValid) {
            validFlags[i] = false;
        } else {
            // 进行梯度方向一致性校验 (解决“衣服上的假椭圆”问题)
            double a_f = refinedEll.size.width / 2.0;
            double b_f = refinedEll.size.height / 2.0;
            double theta_f = -refinedEll.angle * CV_PI / 180.0;
            double cosT_f = std::cos(theta_f);
            double sinT_f = std::sin(theta_f);
            double invA2_f = 1.0 / (a_f * a_f);
            double invB2_f = 1.0 / (b_f * b_f);
            double cosGradAngle = std::cos(params_.fit.gradAngleThresh * CV_PI / 180.0);
            
            int sampleNum = params_.fit.sampleNum;
            int count = 0;
            int validSamples = 0;
            
            for (int j = 0; j < sampleNum; j++) {
                double rad = j * CV_2PI / sampleNum;
                double cosa = a_f * std::cos(rad);
                double cosb = b_f * std::sin(rad);
                int xi = cvRound(cosT_f * cosa + sinT_f * cosb + refinedEll.center.x);
                int yi = cvRound(-sinT_f * cosa + cosT_f * cosb + refinedEll.center.y);
                
                if (xi < 0 || xi >= width_ || yi < 0 || yi >= height_) continue;
                
                validSamples++;
                cv::Vec2f g = direction(cv::Point(xi, yi));
                if (g == cv::Vec2f(0, 0)) continue;
                
                // 计算椭圆在该点的法线
                double dx = xi - refinedEll.center.x;
                double dy = yi - refinedEll.center.y;
                double rx = dx * cosT_f - dy * sinT_f;
                double ry = dx * sinT_f + dy * cosT_f;
                cv::Vec2d rdir(2 * rx * cosT_f * invA2_f + 2 * ry * sinT_f * invB2_f,
                               2 * rx * (-sinT_f) * invA2_f + 2 * ry * cosT_f * invB2_f);
                double norm = cv::norm(rdir);
                if (norm > 0) rdir /= norm;
                
                double dot = g[0] * rdir[0] + g[1] * rdir[1];
                bool pass = false;
                if (params_.fit.polarity == 0) pass = (std::abs(dot) > cosGradAngle);
                else if (params_.fit.polarity == 1) pass = (dot <= -cosGradAngle);
                else if (params_.fit.polarity == -1) pass = (dot >= cosGradAngle);
                
                if (pass) count++;
            }
            
            double score = (validSamples > 0) ? (double)count / validSamples : 0;
            // 细化后的评分必须达到一定阈值（衣服上的噪声纹理通常不具备一致的梯度场）
            if (score < params_.fit.remainScore * 0.9) { // 略微放宽，因为细化后的点可能在像素间隙
                validFlags[i] = false;
            } else {
                clusteredEllipse_[i] = refinedEll;
            }
        }
    }

    std::vector<cv::RotatedRect> validEllipses;
    std::vector<cv::Vec3f> validScores;
    for (size_t i = 0; i < clusteredEllipse_.size(); i++) {
        if (validFlags[i]) {
            validEllipses.emplace_back(clusteredEllipse_[i]);
            if (i < clusteredEllipseScore_.size()) validScores.emplace_back(clusteredEllipseScore_[i]);
        }
    }
    clusteredEllipse_ = std::move(validEllipses);
    clusteredEllipseScore_ = std::move(validScores);
}

// ============================================================================
// 工具函数
// ============================================================================

bool EllipseDetectorImpl::check(const std::vector<cv::Point>& lv) {
    for (size_t i = 1; i < lv.size() - 1; i++) {
        if (lv[i - 1].cross(lv[i]) * lv[i].cross(lv[i + 1]) < 0) {
            return false;
        }
    }
    return true;
}

bool EllipseDetectorImpl::canMatch(int id1, int id2) {
    const std::vector<cv::Point>& e1 = arcs_[id1];
    const std::vector<cv::Point>& e2 = arcs_[id2];
    const std::vector<std::pair<int, int>>& seg1 = arcSegs_[id1];
    const std::vector<std::pair<int, int>>& seg2 = arcSegs_[id2];
    
    int segSize1 = static_cast<int>(seg1.size());
    int segSize2 = static_cast<int>(seg2.size());
    
    if (segSize1 < 2 || segSize2 < 2) return false;
    
    cv::Point l1 = e1[seg1[segSize1 - 2].second - 1] - e1[seg1[segSize1 - 2].first];
    cv::Point l2 = e1[seg1[segSize1 - 1].second - 1] - e1[seg1[segSize1 - 1].first];
    cv::Point l3 = e2[seg2[0].first] - e1[seg1[segSize1 - 1].second - 1];
    cv::Point l4 = e2[seg2[0].second - 1] - e2[seg2[0].first];
    cv::Point l5 = e2[seg2[1].second - 1] - e2[seg2[1].first];
    
    bool flag = true;
    if (!check({l1, l2, l3, l4, l5})) {
        cv::Point l3_1 = e2[seg2[0].first] - e1[seg1[segSize1 - 2].second - 1];
        int lenL1 = seg1[segSize1 - 1].second - seg1[segSize1 - 1].first;
        
        cv::Point l3_2 = e2[seg2[1].first] - e1[seg1[segSize1 - 1].second - 1];
        int lenL4 = seg2[0].second - seg2[0].first;
        
        bool tmpFlag = false;
        
        if (segSize1 > 2) {
            cv::Point l6 = e1[seg1[segSize1 - 3].second - 1] - e1[seg1[segSize1 - 3].first];
            if (check({l6, l1, l3_1, l4, l5}) && lenL1 * 3 < static_cast<int>(e1.size())) {
                tmpFlag = true;
            }
        } else {
            if (check({l1, l3_1, l4, l5}) && lenL1 * 3 < static_cast<int>(e1.size())) {
                tmpFlag = true;
            }
        }
        
        if (segSize2 > 2) {
            cv::Point l7 = e2[seg2[2].second - 1] - e2[seg2[2].first];
            if (check({l1, l2, l3_2, l5, l7}) && lenL4 * 3 < static_cast<int>(e2.size())) {
                tmpFlag = true;
            }
        } else {
            if (check({l1, l2, l3_2, l5}) && lenL4 * 3 < static_cast<int>(e2.size())) {
                tmpFlag = true;
            }
        }
        
        std::vector<cv::Point> lv;
        if (segSize1 > 2) {
            cv::Point l6 = e1[seg1[segSize1 - 3].second - 1] - e1[seg1[segSize1 - 3].first];
            lv.emplace_back(l6);
        }
        lv.emplace_back(l1);
        lv.emplace_back(l5);
        if (segSize2 > 2) {
            cv::Point l7 = e2[seg2[2].second - 1] - e2[seg2[2].first];
            lv.emplace_back(l7);
        }
        if (check(lv)) tmpFlag = true;
        
        flag = tmpFlag;
    }
    
    return flag;
}

bool EllipseDetectorImpl::canFromWeightedPair(int id1, int id2) {
    const std::vector<cv::Point>& e1 = arcs_[id1];
    const std::vector<cv::Point>& e2 = arcs_[id2];
    
    int longerLen = std::max(static_cast<int>(e1.size()), static_cast<int>(e2.size()));
    int shorterLen = std::min(static_cast<int>(e1.size()), static_cast<int>(e2.size()));
    
    // 弧段长度比限制
    if (1.0 * longerLen / shorterLen > params_.fit.thLengthRatio) {
        return false;
    }
    
    // 距离限制
    const cv::Point& p1m = e1[e1.size() / 2];
    const cv::Point& p2m = e2[e2.size() / 2];
    if (cv::norm(p1m - p2m) > params_.fit.thDistance * (longerLen + shorterLen)) {
        return false;
    }
    
    // 融合约束：两个圆弧的中点应在对方的"凸侧"
    cv::Point chordV1 = e1.back() - e1.front();
    if (chordV1.cross(p1m - e1.front()) * chordV1.cross(p2m - e1.front()) > 0) {
        return false;
    }
    
    cv::Point chordV2 = e2.back() - e2.front();
    if (chordV2.cross(p1m - e2.front()) * chordV2.cross(p2m - e2.front()) > 0) {
        return false;
    }
    
    return canMatch(id1, id2) && canMatch(id2, id1);
}

bool EllipseDetectorImpl::canMerge(int id1, int id2) {
    const std::vector<cv::Point>& e1 = arcs_[id1];
    const std::vector<cv::Point>& e2 = arcs_[id2];
    const cv::Point& p1m = e1[e1.size() / 2];
    const cv::Point& p2m = e2[e2.size() / 2];
    
    cv::Point chordV1 = e1.back() - e1.front();
    if (chordV1.cross(p1m - e1.front()) * chordV1.cross(p2m - e1.front()) > 0) {
        return false;
    }
    
    cv::Point chordV2 = e2.back() - e2.front();
    if (chordV2.cross(p1m - e2.front()) * chordV2.cross(p2m - e2.front()) > 0) {
        return false;
    }
    
    return true;
}

cv::RotatedRect EllipseDetectorImpl::fit(const std::vector<int>& ids) {
    std::vector<cv::Point> points;
    points.reserve(12);
    
    for (int i : ids) {
        for (const auto& seg : arcSegs_[i]) {
            points.emplace_back(arcs_[i][seg.first]);
        }
        points.emplace_back(arcs_[i][arcSegs_[i].back().second - 1]);
    }
    
    if (points.size() < 5) return cv::RotatedRect();
    
    cv::Mat ptsMat;
    cv::Mat(points).convertTo(ptsMat, CV_32F);
    cv::RotatedRect ell = cv::fitEllipse(ptsMat);
    
    double minDim = std::min(ell.size.width, ell.size.height);
    double maxDim = std::max(ell.size.width, ell.size.height);
    
    if (minDim < params_.fit.minEllipseRadius * 2.0) return cv::RotatedRect();
    if (maxDim > 0 && (minDim / maxDim) < params_.fit.minAspectRatio) return cv::RotatedRect();
    if (minDim > 0 && (maxDim / minDim) > params_.fit.maxAspectRatio) return cv::RotatedRect();
    
    return ell;
}

cv::Vec3f EllipseDetectorImpl::interiorRate(const std::vector<int>& ids, const cv::RotatedRect& ell) {
    double a = ell.size.width / 2.0;
    double b = ell.size.height / 2.0;
    double minR = std::min(a, b);
    double maxR = std::max(a, b);
    
    if (maxR > std::max(height_, width_) || minR < params_.fit.minEllipseRadius) return cv::Vec3f(0, 0, 0);
    if (maxR > 0 && (minR / maxR) < params_.fit.minAspectRatio) return cv::Vec3f(0, 0, 0);
    if (minR > 0 && (maxR / minR) > params_.fit.maxAspectRatio) return cv::Vec3f(0, 0, 0);
    
    double L = CV_PI * (3 * (a + b) - std::sqrt((3 * a + b) * (a + 3 * b)));
    double theta = ell.angle;
    
    float cosT = std::cos(-theta * CV_PI / 180);
    float sinT = std::sin(-theta * CV_PI / 180);
    
    float invA2 = 1.0f / (a * a);
    float invB2 = 1.0f / (b * b);
    
    double inlierDist2 = params_.fit.inlierDist * params_.fit.inlierDist;
    
    auto countOnEllipse = [&](const std::vector<cv::Point>& _points) -> int {
        int counter = 0;
        for (const auto& p : _points) {
            auto tp = cv::Point2f(p) - ell.center;
            float rx = (tp.x * cosT - tp.y * sinT);
            float ry = (tp.x * sinT + tp.y * cosT);
            float h = (rx * rx) * invA2 + (ry * ry) * invB2;
            float d2 = (tp.x * tp.x + tp.y * tp.y) * (h * h * 0.25f - h * 0.5f + 0.25f);
            if (d2 < inlierDist2) ++counter;
        }
        return counter;
    };
    
    double ct = 0;
    int total = 0;
    for (int i : ids) {
        total += static_cast<int>(arcs_[i].size());
        ct += countOnEllipse(arcs_[i]);
    }
    
    return cv::Vec3f(static_cast<float>(ct), static_cast<float>(ct / total), static_cast<float>(ct / L));
}

} // 匿名命名空间

// ============================================================================
// 公开接口
// ============================================================================

std::vector<Ellipse> detectEllipses(const cv::Mat& image, const DetectorParams& params) {
    EllipseDetectorImpl detector(image, params);
    return detector.detect();
}
