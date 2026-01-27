// 1. Store original image
// 2. Add refineEllipses method declaration
// 3. Call refineEllipses in detect()
// 4. Implement refineEllipses logic

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
    void refineEllipses(); // Secondary refinement
    
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
    cv::Mat originalImage_; // Stored original image for refinement
    
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
            
            // --- Center Intensity Constraint ---
            if (centerIntensityThresh_ != 0) {
                 int cx = cvRound(ell.center.x);
                 int cy = cvRound(ell.center.y);
                 if (cx >= 0 && cx < width_ && cy >= 0 && cy < height_) {
                     uchar val = image_.at<uchar>(cy, cx);
                     if (polarity_ == 1) { // Bright Ellipse -> Center should be Bright
                         if (val <= brightCenterThresh_) continue;
                     } else if (polarity_ == -1) { // Dark Ellipse -> Center should be Dark
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
                
                // Dot product of image gradient and ellipse normal (outward)
                double dot = cdir.dot(rdir);

                bool pass = false;
                if (polarity_ == 0) {
                    // Accept both parallel and anti-parallel
                    if (std::abs(dot) > cosGradAngle_) pass = true;
                } else if (polarity_ == 1) { 
                    // Polarity 1: Bright Ellipse -> Gradient Inward -> Anti-parallel to Normal
                    if (dot <= -cosGradAngle_) pass = true;
                } else if (polarity_ == -1) { 
                    // Polarity -1: Dark Ellipse -> Gradient Outward -> Parallel to Normal
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
    
    // 7. 二次拟合 (Refinement)
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

// 3x3 Sobel for magnitude and direction
void computeGradientROI(const cv::Mat& img, cv::Mat& mag, cv::Mat& dx, cv::Mat& dy) {
    cv::Sobel(img, dx, CV_32F, 1, 0, 3);
    cv::Sobel(img, dy, CV_32F, 0, 1, 3);
    cv::magnitude(dx, dy, mag);
}

// Parabolic interpolation for sub-pixel peak
// Returns offset [-0.5, 0.5] from center
float parabolicInterpolation(float y_minus, float y_0, float y_plus) {
    float denom = 2.0f * (y_minus - 2.0f * y_0 + y_plus);
    if (std::abs(denom) < 1e-5f) return 0.0f;
    return (y_minus - y_plus) / denom;
}

// Get interpolated value at float coordinates
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

// Robust geometric distance approximation (Sampson distance)
// f = (x/a)^2 + (y/b)^2 - 1
// Grad(f) = [2x/a^2, 2y/b^2]
// Dist approx = |f| / |Grad(f)|
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
    // Rotate to canonical
    double rx = tx * cosA + ty * sinA;
    double ry = -tx * sinA + ty * cosA;
    
    double a2 = a * a;
    double b2 = b * b;
    
    double f = (rx * rx) / a2 + (ry * ry) / b2 - 1.0;
    
    double df_dx = 2 * rx / a2; // in canonical
    double df_dy = 2 * ry / b2;
    
    // Norm of gradient in original space is same as in rotated space (rotation is isometric)
    double gradNorm = std::sqrt(df_dx * df_dx + df_dy * df_dy);
    
    // Avoid div by zero
    if (gradNorm < 1e-6) gradNorm = 1e-6;
    
    if (outRes) *outRes = f / gradNorm;
    
    return std::abs(f) / gradNorm;
}

// Helper to compute Jacobian numerically for 5 params: cx, cy, a, b, angle
// Params vector: [cx, cy, a, b, angle]
void computeJacobianNumerical(const cv::Point2f& p, double* params, double* J) {
    double eps[5] = {0.5, 0.5, 0.5, 0.5, 0.5}; // Step sizes
    
    cv::RotatedRect baseEll(cv::Point2f(params[0], params[1]), cv::Size2f(params[2]*2, params[3]*2), params[4]);
    double y0;
    computeSampsonDist(p, baseEll, &y0);
    
    for (int i = 0; i < 5; i++) {
        double oldVal = params[i];
        params[i] += eps[i];
        
        cv::RotatedRect pertEll(cv::Point2f(params[0], params[1]), cv::Size2f(params[2]*2, params[3]*2), params[4]);
        double yPlus;
        computeSampsonDist(p, pertEll, &yPlus);
        
        params[i] = oldVal - eps[i];
        cv::RotatedRect pertEll2(cv::Point2f(params[0], params[1]), cv::Size2f(params[2]*2, params[3]*2), params[4]);
        double yMinus;
        computeSampsonDist(p, pertEll2, &yMinus);
        
        params[i] = oldVal; // Restore
        
        J[i] = (yPlus - yMinus) / (2.0 * eps[i]);
    }
}

} // namespace

void EllipseDetectorImpl::refineEllipses() {
    if (clusteredEllipse_.empty()) return;

    const int maxIter = 10; // Max LM iterations
    const double roiRatio = params_.refine.roiRatio;
    const double tukeyAlpha = params_.refine.tukeyAlpha;
    const int minGrad = params_.refine.minGradient;
    const double lambdaInit = 0.01;

    for (size_t i = 0; i < clusteredEllipse_.size(); i++) {
        cv::RotatedRect& ell = clusteredEllipse_[i];

        // 1. Extract ROI (Backtracking)
        double maxDim = std::max(ell.size.width, ell.size.height);
        int roiSize = static_cast<int>(maxDim * roiRatio);
        
        int minX = static_cast<int>(ell.center.x - roiSize / 2);
        int minY = static_cast<int>(ell.center.y - roiSize / 2);
        int maxX = minX + roiSize;
        int maxY = minY + roiSize;

        // Clip
        int r_minX = std::max(0, minX);
        int r_minY = std::max(0, minY);
        int r_maxX = std::min(width_, maxX);
        int r_maxY = std::min(height_, maxY);

        if (r_maxX - r_minX < 10 || r_maxY - r_minY < 10) continue;

        cv::Rect roiRect(r_minX, r_minY, r_maxX - r_minX, r_maxY - r_minY);
        cv::Mat roiImg = originalImage_(roiRect);
        if (roiImg.empty()) continue;

        // 2. Gradients & Subpixel Extraction
        cv::Mat mag, dx, dy;
        computeGradientROI(roiImg, mag, dx, dy);

        std::vector<cv::Point2f> points;
        
        for (int y = 1; y < mag.rows - 1; y++) {
            for (int x = 1; x < mag.cols - 1; x++) {
                float val = mag.at<float>(y, x);
                if (val < minGrad) continue;
                
                // Non-maximum suppression along gradient could be added, but user asked for "Along gradient take 3 points"
                // Let's implement the specific logic:
                // Normalized gradient direction
                float gX = dx.at<float>(y, x);
                float gY = dy.at<float>(y, x);
                float len = std::sqrt(gX*gX + gY*gY);
                if (len < 1e-4) continue;
                gX /= len; gY /= len;
                
                // Sample 3 points: Current (0), +Dir (+1), -Dir (-1)
                float v0 = val;
                float vPlus = getInterpValue(mag, x + gX, y + gY);
                float vMinus = getInterpValue(mag, x - gX, y - gY);
                
                // Check if it's a local maximum along gradient
                if (v0 >= vPlus && v0 >= vMinus) {
                    float delta = parabolicInterpolation(vMinus, v0, vPlus);
                    // Subpixel coordinate
                    float spX = x + delta * gX;
                    float spY = y + delta * gY;
                    
                    // Convert to global
                    points.emplace_back(cv::Point2f(spX + r_minX, spY + r_minY));
                }
            }
        }

        if (points.size() < 10) continue;

        // 3. LM Optimization
        double p[5] = {ell.center.x, ell.center.y, ell.size.width/2.0, ell.size.height/2.0, ell.angle};
        double lambda = lambdaInit;
        
        // Weights for points
        std::vector<double> W(points.size(), 1.0);
        
        for (int iter = 0; iter < maxIter; iter++) {
            // Build Linear System: (J^T W J + lambda I) dp = - J^T W r   (Here r is signed residual)
            // Or usually: min sum ( w * r^2 ). update = -(J'WJ + lam I)^-1 J'W r
            
            cv::Mat J(static_cast<int>(points.size()), 5, CV_64F);
            cv::Mat r(static_cast<int>(points.size()), 1, CV_64F);
            cv::Mat Wmat = cv::Mat::zeros(static_cast<int>(points.size()), static_cast<int>(points.size()), CV_64F);
            
            double currentError = 0;
            std::vector<double> residuals(points.size());
            
            // Compute Residuals & Jacobian
            for (size_t k = 0; k < points.size(); k++) {
                double res;
                computeSampsonDist(points[k], 
                    cv::RotatedRect(cv::Point2f(p[0],p[1]), cv::Size2f(p[2]*2, p[3]*2), p[4]), 
                    &res);
                
                residuals[k] = res; // Signed geometric distance approx
                r.at<double>(static_cast<int>(k), 0) = res;
                
                // Jacobian row
                double Jrow[5];
                computeJacobianNumerical(points[k], p, Jrow);
                for (int j = 0; j < 5; j++) J.at<double>(static_cast<int>(k), j) = Jrow[j];
                
                // 4. Update Weights (Tukey Bisquare)
                // Need scale estimate 'c'
                // Re-estimate c every iter or once? Usually once per robust step or adaptive.
                // User said: "Use Bisquare weights".
                // We calculate c using Median Absolute Deviation of residuals
            }
            
            // Calc c (MAD) 
            // Better: c = 1.4826 * MAD * tune_factor. 
            // If tukeyAlpha is the 'tune_factor' (e.g. 4.685 for 95% efficiency), then c = tukeyAlpha * sigma_mad.
            // Let's assume params_.tukeyAlpha IS the 'c' in pixels directly if small, OR factor.
            // Typically c ~ 4-5 pixels for robust fitting is generous. 
            // Let's calculate sigma first.
           
            // Copy residuals absolute values
            std::vector<double> absRes;
            absRes.reserve(points.size());
            for(double v : residuals) absRes.push_back(std::abs(v));
            size_t n = absRes.size();
            std::nth_element(absRes.begin(), absRes.begin() + n/2, absRes.end());
            double mad = absRes[n/2];
            double sigma = 1.4826 * mad;
            double c_val = (sigma < 0.1) ? tukeyAlpha : (tukeyAlpha * sigma); // Use user alpha as factor
            if (c_val < 1.0) c_val = 1.0; // Min clamping
            
            // Apply Tukey
            for (size_t k = 0; k < points.size(); k++) {
                double u = std::abs(residuals[k]) / c_val;
                double w = 0;
                if (u <= 1.0) {
                    double tmp = 1.0 - u*u;
                    w = tmp * tmp;
                }
                Wmat.at<double>(static_cast<int>(k),static_cast<int>(k)) = w;
                currentError += w * residuals[k] * residuals[k];
            }
            
            // Solve System
            cv::Mat Jt = J.t();
            cv::Mat JtW = Jt * Wmat;
            cv::Mat JtWJ = JtW * J;
            cv::Mat JtWr = JtW * r;
            
            // Levenberg-Marquardt Damping
            for (int j = 0; j < 5; j++) {
                JtWJ.at<double>(j, j) *= (1.0 + lambda);
            }
            
            cv::Mat delta;
            bool solved = cv::solve(JtWJ, -JtWr, delta, cv::DECOMP_CHOLESKY);
            
            if (!solved) break;
            
            // Update candidate
            double p_new[5];
            for (int j = 0; j < 5; j++) p_new[j] = p[j] + delta.at<double>(j, 0);
            
            // Check if error reduced (simplified LM step check)
            // Just accept step for this simple impl, or check bounds
            // Constraints: a>0, b>0, and not too large
            double maxImgDim = std::max(width_, height_) * 1.5;
            if (p_new[2] < 1.0 || p_new[3] < 1.0 || p_new[2] > maxImgDim || p_new[3] > maxImgDim) {
                lambda *= 10;
                continue;
            }
            
            // Accept
            for (int j = 0; j < 5; j++) p[j] = p_new[j];
            lambda /= 10;
            if (lambda < 1e-6) lambda = 1e-6;
             
             // Convergence check
             if (cv::norm(delta) < 1e-4) break;
        }

        // Update result
        clusteredEllipse_[i] = cv::RotatedRect(cv::Point2f(p[0],p[1]), cv::Size2f(p[2]*2, p[3]*2), p[4]);
    }
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
    
    if (points.size() < 5) {
        return cv::RotatedRect();
    }
    
    cv::RotatedRect ell = cv::fitEllipse(points);
    
    // 偏心率/长短轴比约束 (Eccentricity Constraint)
    double minDim = std::min(ell.size.width, ell.size.height);
    double maxDim = std::max(ell.size.width, ell.size.height);
    if (maxDim > 0 && (minDim / maxDim) < params_.fit.minAspectRatio) {
        return cv::RotatedRect(); // Return invalid rect if too flat
    }
    
    return ell;
}

cv::Vec3f EllipseDetectorImpl::interiorRate(const std::vector<int>& ids, const cv::RotatedRect& ell) {
    int a = static_cast<int>(ell.size.width / 2.0);
    int b = static_cast<int>(ell.size.height / 2.0);
    double r = 1.0 * a / b;
    
    if (std::max(a, b) > std::max(height_, width_) ||
        std::min(a, b) < 2 || r < 0.2 || r > 5) {
        return cv::Vec3f(0, 0, 0);
    }
    
    double L = CV_PI * (3 * (a + b) - std::sqrt((3 * a + b) * (a + 3 * b)));
    double theta = ell.angle;
    
    float cosT = std::cos(-theta * CV_PI / 180);
    float sinT = std::sin(-theta * CV_PI / 180);
    
    float invA2 = 1.0f / (a * a);
    float invB2 = 1.0f / (b * b);
    
    double inlierDist2 = params_.fit.inlierDist * params_.fit.inlierDist;
    
    auto countOnEllipse = [&](const std::vector<cv::Point>& points) -> int {
        int counter = 0;
        for (const auto& p : points) {
            auto tp = cv::Point2f(p) - ell.center;
            float rx = (tp.x * cosT - tp.y * sinT);
            float ry = (tp.x * sinT + tp.y * cosT);
            float h = (rx * rx) * invA2 + (ry * ry) * invB2;
            float d2 = (tp.x * tp.x + tp.y * tp.y) * (h * h * 0.25f - h * 0.5f + 0.25f);
            
            if (d2 < inlierDist2) {
                ++counter;
            }
        }
        return counter;
    };
    
    double ct = 0;
    int total = 0;
    for (int i : ids) {
        total += static_cast<int>(arcs_[i].size());
        ct += countOnEllipse(arcs_[i]);
    }
    
    return cv::Vec3f(static_cast<float>(ct),
                     static_cast<float>(ct / total),
                     static_cast<float>(ct / L));
}

} // anonymous namespace

// ============================================================================
// 公开接口
// ============================================================================

std::vector<Ellipse> detectEllipses(const cv::Mat& image, const DetectorParams& params) {
    EllipseDetectorImpl detector(image, params);
    return detector.detect();
}
