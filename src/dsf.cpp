/**
 * @file dsf.cpp
 * @brief 边缘检测与平滑圆弧提取模块实现
 * 
 * 从 EDSF 项目严格移植的边缘检测算法
 */

#include "dsf.h"
#include <algorithm>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// 算法内部常量 (仅在此文件内有效)
// ============================================================================

#define LEFT  1
#define RIGHT 2
#define UP    3
#define DOWN  4

#define EDGE_VERTICAL   1
#define EDGE_HORIZONTAL 2

// ============================================================================
// 内部数据结构
// ============================================================================

struct Chain {
    int dir;            // 链方向
    int len;            // 像素数量
    int parent;         // 父节点索引
    int children[2];    // 子节点索引
    cv::Point* pixels;  // 像素数组指针
};

struct StackNode {
    int r, c;       // 起始像素坐标
    int parent;     // 父链索引（-1表示无父节点）
    int dir;        // 追踪方向
};

// ============================================================================
// 边缘检测器实现
// ============================================================================

EdgeDetector::EdgeDetector(const cv::Mat& srcImage, const EdgeParams& params)
    : params_(params) {
    
    // 参数合法性检查
    if (params_.gradThresh < 1) params_.gradThresh = 1;
    if (params_.anchorThresh < 0) params_.anchorThresh = 0;
    if (params_.sigma < 1.0) params_.sigma = 1.0;
    
    segmentPoints_.reserve(50);
    
    // 转换为灰度图像
    cv::Mat grayImage;
    if (srcImage.channels() == 3) {
        cv::cvtColor(srcImage, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = srcImage.clone();
    }
    
    // 初步平滑
    if (params_.preBlurSize > 0) {
        cv::GaussianBlur(grayImage, grayImage, cv::Size(params_.preBlurSize, params_.preBlurSize), params_.preBlurSigma);
    }
    
    srcImage_ = grayImage;
    height_ = srcImage_.rows;
    width_ = srcImage_.cols;
    
    segmentNos_ = 0;
    segmentPoints_.emplace_back(std::vector<cv::Point>()); // 预分配空白段
    
    // 初始化图像
    edgeImage_ = cv::Mat(height_, width_, CV_8UC1, cv::Scalar(0));
    smoothImage_ = cv::Mat(height_, width_, CV_8UC1);
    gradImage_ = cv::Mat(height_, width_, CV_16SC1);
    
    srcImg_ = srcImage_.data;
    
    // 高斯平滑
    if (params_.sigma > 0 && params_.smoothBlurSize > 0) {
        if (params_.sigma == 1.0) {
            cv::GaussianBlur(srcImage_, smoothImage_, cv::Size(params_.smoothBlurSize, params_.smoothBlurSize), params_.sigma);
        } else {
            cv::GaussianBlur(srcImage_, smoothImage_, cv::Size(), params_.sigma);
        }
    } else {
        smoothImage_ = srcImage_.clone();
    }
    
    // 分配指针
    smoothImg_ = smoothImage_.data;
    gradImg_ = reinterpret_cast<short*>(gradImage_.data);
    edgeImg_ = edgeImage_.data;
    
    // 分配方向图像
    dirImg_ = new unsigned char[width_ * height_];
    dirImage_ = cv::Mat(height_, width_, CV_8UC1, dirImg_);
    
    // 执行边缘检测流程
    computeGradient();
    computeAnchorPoints();
    joinAnchorPointsUsingSortedAnchors();
}

void EdgeDetector::computeGradient() {
    // 初始化边界梯度
    for (int j = 0; j < width_; j++) {
        gradImg_[j] = params_.gradThresh - 1;
        gradImg_[(height_ - 1) * width_ + j] = params_.gradThresh - 1;
    }
    for (int i = 1; i < height_ - 1; i++) {
        gradImg_[i * width_] = params_.gradThresh - 1;
        gradImg_[(i + 1) * width_ - 1] = params_.gradThresh - 1;
    }
    
    // 计算梯度
    for (int i = 1; i < height_ - 1; i++) {
        for (int j = 1; j < width_ - 1; j++) {
            // Prewitt/Sobel/Scharr 算子
            // A B C
            // D x E
            // F G H
            int com1 = smoothImg_[(i + 1) * width_ + j + 1] - smoothImg_[(i - 1) * width_ + j - 1];
            int com2 = smoothImg_[(i - 1) * width_ + j + 1] - smoothImg_[(i + 1) * width_ + j - 1];
            
            int gx, gy;
            
            switch (params_.op) {
                case GradientOperator::PREWITT:
                    gx = std::abs(com1 + com2 + (smoothImg_[i * width_ + j + 1] - smoothImg_[i * width_ + j - 1]));
                    gy = std::abs(com1 - com2 + (smoothImg_[(i + 1) * width_ + j] - smoothImg_[(i - 1) * width_ + j]));
                    break;
                case GradientOperator::SOBEL:
                    gx = std::abs(com1 + com2 + 2 * (smoothImg_[i * width_ + j + 1] - smoothImg_[i * width_ + j - 1]));
                    gy = std::abs(com1 - com2 + 2 * (smoothImg_[(i + 1) * width_ + j] - smoothImg_[(i - 1) * width_ + j]));
                    break;
                case GradientOperator::SCHARR:
                    gx = std::abs(3 * (com1 + com2) + 10 * (smoothImg_[i * width_ + j + 1] - smoothImg_[i * width_ + j - 1]));
                    gy = std::abs(3 * (com1 - com2) + 10 * (smoothImg_[(i + 1) * width_ + j] - smoothImg_[(i - 1) * width_ + j]));
                    break;
                default:
                    gx = std::abs(com1 + com2 + (smoothImg_[i * width_ + j + 1] - smoothImg_[i * width_ + j - 1]));
                    gy = std::abs(com1 - com2 + (smoothImg_[(i + 1) * width_ + j] - smoothImg_[(i - 1) * width_ + j]));
            }
            
            int sum = static_cast<int>(std::sqrt(static_cast<double>(gx * gx + gy * gy)));
            int index = i * width_ + j;
            gradImg_[index] = static_cast<short>(sum);
            
            if (sum >= params_.gradThresh) {
                if (std::abs(gx) >= std::abs(gy)) {
                    dirImg_[index] = EDGE_VERTICAL;
                } else {
                    dirImg_[index] = EDGE_HORIZONTAL;
                }
            }
        }
    }
}

void EdgeDetector::computeAnchorPoints() {
    for (int i = 2; i < height_ - 2; i++) {
        int start = 2;
        int inc = 1;
        if (i % params_.scanInterval != 0) {
            start = params_.scanInterval;
            inc = params_.scanInterval;
        }
        
        for (int j = start; j < width_ - 2; j += inc) {
            int currentGrad = gradImg_[i * width_ + j];
            if (currentGrad < params_.gradThresh) continue;
            
            if (dirImg_[i * width_ + j] == EDGE_VERTICAL) {
                // 垂直边缘 (寻找水平方向的极大值)
                int leftGrad = gradImg_[i * width_ + j - 1];
                int rightGrad = gradImg_[i * width_ + j + 1];
                
                // 改进逻辑：只要是局部极大值且满足最小差值要求即可
                // 对于模糊边缘，这个差值可能很小
                if (currentGrad > leftGrad && currentGrad > rightGrad) {
                   if ((currentGrad - leftGrad) >= params_.anchorThresh || (currentGrad - rightGrad) >= params_.anchorThresh) {
                        edgeImg_[i * width_ + j] = ANCHOR_PIXEL;
                        anchorPoints_.emplace_back(cv::Point(j, i));
                   }
                }
            } else {
                // 水平边缘 (寻找垂直方向的极大值)
                int upGrad = gradImg_[(i - 1) * width_ + j];
                int downGrad = gradImg_[(i + 1) * width_ + j];
                
                if (currentGrad > upGrad && currentGrad > downGrad) {
                    if ((currentGrad - upGrad) >= params_.anchorThresh || (currentGrad - downGrad) >= params_.anchorThresh) {
                        edgeImg_[i * width_ + j] = ANCHOR_PIXEL;
                        anchorPoints_.emplace_back(cv::Point(j, i));
                    }
                }
            }
        }
    }
    
    anchorNos_ = static_cast<int>(anchorPoints_.size());
}

// ============================================================================
// 搜索辅助函数 (Static Helpers)
// ============================================================================

static int longestChain(Chain* chains, int root) {
    if (root == -1 || chains[root].len == 0) return 0;
    
    int len0 = 0;
    if (chains[root].children[0] != -1) {
        len0 = longestChain(chains, chains[root].children[0]);
    }
    
    int len1 = 0;
    if (chains[root].children[1] != -1) {
        len1 = longestChain(chains, chains[root].children[1]);
    }
    
    int max = 0;
    if (len0 >= len1) {
        max = len0;
        chains[root].children[1] = -1;
    } else {
        max = len1;
        chains[root].children[0] = -1;
    }
    
    return chains[root].len + max;
}

static int retrieveChainNos(Chain* chains, int root, int chainNos[]) {
    int count = 0;
    
    while (root != -1) {
        chainNos[count] = root;
        count++;
        
        if (chains[root].children[0] != -1) {
            root = chains[root].children[0];
        } else {
            root = chains[root].children[1];
        }
    }
    
    return count;
}

void EdgeDetector::sortAnchorsByGradValue() {
    auto sortFunc = [this](const cv::Point& a, const cv::Point& b) {
        return gradImg_[a.y * width_ + a.x] > gradImg_[b.y * width_ + b.x];
    };
    std::sort(anchorPoints_.begin(), anchorPoints_.end(), sortFunc);
}

void EdgeDetector::joinAnchorPointsUsingSortedAnchors() {
    int* chainNos = new int[(width_ + height_) * 8];
    cv::Point* pixels = new cv::Point[width_ * height_];
    StackNode* stack = new StackNode[width_ * height_];
    Chain* chains = new Chain[width_ * height_];
    
    // 按梯度值排序锚点
    const int SIZE = 128 * 256; // 梯度排序用的桶数量
    int* C = new int[SIZE];
    std::memset(C, 0, sizeof(int) * SIZE);
    
    // 统计梯度值
    for (int i = 1; i < height_ - 1; i++) {
        for (int j = 1; j < width_ - 1; j++) {
            if (edgeImg_[i * width_ + j] != ANCHOR_PIXEL) continue;
            int grad = gradImg_[i * width_ + j];
            if (grad >= SIZE) grad = SIZE - 1; // 边界检查
            C[grad]++;
        }
    }
    
    // 计算索引
    for (int i = 1; i < SIZE; i++) C[i] += C[i - 1];
    
    int noAnchors = C[SIZE - 1];
    int* A = new int[noAnchors];
    std::memset(A, 0, sizeof(int) * noAnchors);
    
    for (int i = 1; i < height_ - 1; i++) {
        for (int j = 1; j < width_ - 1; j++) {
            if (edgeImg_[i * width_ + j] != ANCHOR_PIXEL) continue;
            int grad = gradImg_[i * width_ + j];
            if (grad >= SIZE) grad = SIZE - 1; // 边界检查
            int index = --C[grad];
            A[index] = i * width_ + j;
        }
    }
    
    delete[] C;
    
    // 从梯度值最大的锚点开始连接
    for (int k = noAnchors - 1; k >= 0; k--) {
        int pixelOffset = A[k];
        int i = pixelOffset / width_;
        int j = pixelOffset % width_;
        
        if (edgeImg_[i * width_ + j] != ANCHOR_PIXEL) continue;
        
        chains[0].len = 0;
        chains[0].parent = -1;
        chains[0].dir = 0;
        chains[0].children[0] = chains[0].children[1] = -1;
        chains[0].pixels = nullptr;
        
        int noChains = 1;
        int len = 0;
        int duplicatePixelCount = 0;
        int top = -1;
        
        if (dirImg_[i * width_ + j] == EDGE_VERTICAL) {
            stack[++top].r = i;
            stack[top].c = j;
            stack[top].dir = DOWN;
            stack[top].parent = 0;
            
            stack[++top].r = i;
            stack[top].c = j;
            stack[top].dir = UP;
            stack[top].parent = 0;
        } else {
            stack[++top].r = i;
            stack[top].c = j;
            stack[top].dir = RIGHT;
            stack[top].parent = 0;
            
            stack[++top].r = i;
            stack[top].c = j;
            stack[top].dir = LEFT;
            stack[top].parent = 0;
        }
        
        // 边缘追踪
    StartOfWhile:
        while (top >= 0) {
            int r = stack[top].r;
            int c = stack[top].c;
            int dir = stack[top].dir;
            int parent = stack[top].parent;
            top--;
            
            if (edgeImg_[r * width_ + c] != EDGE_PIXEL) duplicatePixelCount++;
            
            chains[noChains].dir = dir;
            chains[noChains].parent = parent;
            chains[noChains].children[0] = chains[noChains].children[1] = -1;
            
            int chainLen = 0;
            chains[noChains].pixels = &pixels[len];
            
            pixels[len].y = r;
            pixels[len].x = c;
            len++;
            chainLen++;
            
            if (dir == LEFT) {
                while (dirImg_[r * width_ + c] == EDGE_HORIZONTAL) {
                    edgeImg_[r * width_ + c] = EDGE_PIXEL;
                    
                    // 清理上下像素
                    if (edgeImg_[(r - 1) * width_ + c] == ANCHOR_PIXEL) edgeImg_[(r - 1) * width_ + c] = 0;
                    if (edgeImg_[(r + 1) * width_ + c] == ANCHOR_PIXEL) edgeImg_[(r + 1) * width_ + c] = 0;
                    
                    // 查找邻居边缘像素
                    if (edgeImg_[r * width_ + c - 1] >= ANCHOR_PIXEL) { c--; }
                    else if (edgeImg_[(r - 1) * width_ + c - 1] >= ANCHOR_PIXEL) { r--; c--; }
                    else if (edgeImg_[(r + 1) * width_ + c - 1] >= ANCHOR_PIXEL) { r++; c--; }
                    else {
                        // 跟随最大梯度
                        int A = gradImg_[(r - 1) * width_ + c - 1];
                        int B = gradImg_[r * width_ + c - 1];
                        int C = gradImg_[(r + 1) * width_ + c - 1];
                        
                        if (A > B) {
                            if (A > C) r--;
                            else r++;
                        } else if (C > B) r++;
                        c--;
                    }
                    
                    if (edgeImg_[r * width_ + c] == EDGE_PIXEL || gradImg_[r * width_ + c] < params_.gradThresh) {
                        if (chainLen > 0) {
                            chains[noChains].len = chainLen;
                            chains[parent].children[0] = noChains;
                            noChains++;
                        }
                        goto StartOfWhile;
                    }
                    
                    pixels[len].y = r;
                    pixels[len].x = c;
                    len++;
                    chainLen++;
                }
                
                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = DOWN;
                stack[top].parent = noChains;
                
                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = UP;
                stack[top].parent = noChains;
                
                len--;
                chainLen--;
                
                chains[noChains].len = chainLen;
                chains[parent].children[0] = noChains;
                noChains++;
                
            } else if (dir == RIGHT) {
                while (dirImg_[r * width_ + c] == EDGE_HORIZONTAL) {
                    edgeImg_[r * width_ + c] = EDGE_PIXEL;
                    
                    if (edgeImg_[(r + 1) * width_ + c] == ANCHOR_PIXEL) edgeImg_[(r + 1) * width_ + c] = 0;
                    if (edgeImg_[(r - 1) * width_ + c] == ANCHOR_PIXEL) edgeImg_[(r - 1) * width_ + c] = 0;
                    
                    if (edgeImg_[r * width_ + c + 1] >= ANCHOR_PIXEL) { c++; }
                    else if (edgeImg_[(r + 1) * width_ + c + 1] >= ANCHOR_PIXEL) { r++; c++; }
                    else if (edgeImg_[(r - 1) * width_ + c + 1] >= ANCHOR_PIXEL) { r--; c++; }
                    else {
                        int A = gradImg_[(r - 1) * width_ + c + 1];
                        int B = gradImg_[r * width_ + c + 1];
                        int C = gradImg_[(r + 1) * width_ + c + 1];
                        
                        if (A > B) {
                            if (A > C) r--;
                            else r++;
                        } else if (C > B) r++;
                        c++;
                    }
                    
                    if (edgeImg_[r * width_ + c] == EDGE_PIXEL || gradImg_[r * width_ + c] < params_.gradThresh) {
                        if (chainLen > 0) {
                            chains[noChains].len = chainLen;
                            chains[parent].children[1] = noChains;
                            noChains++;
                        }
                        goto StartOfWhile;
                    }
                    
                    pixels[len].y = r;
                    pixels[len].x = c;
                    len++;
                    chainLen++;
                }
                
                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = DOWN;
                stack[top].parent = noChains;
                
                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = UP;
                stack[top].parent = noChains;
                
                len--;
                chainLen--;
                
                chains[noChains].len = chainLen;
                chains[parent].children[1] = noChains;
                noChains++;
                
            } else if (dir == UP) {
                while (dirImg_[r * width_ + c] == EDGE_VERTICAL) {
                    edgeImg_[r * width_ + c] = EDGE_PIXEL;
                    
                    if (edgeImg_[r * width_ + c - 1] == ANCHOR_PIXEL) edgeImg_[r * width_ + c - 1] = 0;
                    if (edgeImg_[r * width_ + c + 1] == ANCHOR_PIXEL) edgeImg_[r * width_ + c + 1] = 0;
                    
                    if (edgeImg_[(r - 1) * width_ + c] >= ANCHOR_PIXEL) { r--; }
                    else if (edgeImg_[(r - 1) * width_ + c - 1] >= ANCHOR_PIXEL) { r--; c--; }
                    else if (edgeImg_[(r - 1) * width_ + c + 1] >= ANCHOR_PIXEL) { r--; c++; }
                    else {
                        int A = gradImg_[(r - 1) * width_ + c - 1];
                        int B = gradImg_[(r - 1) * width_ + c];
                        int C = gradImg_[(r - 1) * width_ + c + 1];
                        
                        if (A > B) {
                            if (A > C) c--;
                            else c++;
                        } else if (C > B) c++;
                        r--;
                    }
                    
                    if (edgeImg_[r * width_ + c] == EDGE_PIXEL || gradImg_[r * width_ + c] < params_.gradThresh) {
                        if (chainLen > 0) {
                            chains[noChains].len = chainLen;
                            chains[parent].children[0] = noChains;
                            noChains++;
                        }
                        goto StartOfWhile;
                    }
                    
                    pixels[len].y = r;
                    pixels[len].x = c;
                    len++;
                    chainLen++;
                }
                
                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = RIGHT;
                stack[top].parent = noChains;
                
                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = LEFT;
                stack[top].parent = noChains;
                
                len--;
                chainLen--;
                
                chains[noChains].len = chainLen;
                chains[parent].children[0] = noChains;
                noChains++;
                
            } else { // dir == DOWN
                while (dirImg_[r * width_ + c] == EDGE_VERTICAL) {
                    edgeImg_[r * width_ + c] = EDGE_PIXEL;
                    
                    if (edgeImg_[r * width_ + c + 1] == ANCHOR_PIXEL) edgeImg_[r * width_ + c + 1] = 0;
                    if (edgeImg_[r * width_ + c - 1] == ANCHOR_PIXEL) edgeImg_[r * width_ + c - 1] = 0;
                    
                    if (edgeImg_[(r + 1) * width_ + c] >= ANCHOR_PIXEL) { r++; }
                    else if (edgeImg_[(r + 1) * width_ + c + 1] >= ANCHOR_PIXEL) { r++; c++; }
                    else if (edgeImg_[(r + 1) * width_ + c - 1] >= ANCHOR_PIXEL) { r++; c--; }
                    else {
                        int A = gradImg_[(r + 1) * width_ + c - 1];
                        int B = gradImg_[(r + 1) * width_ + c];
                        int C = gradImg_[(r + 1) * width_ + c + 1];
                        
                        if (A > B) {
                            if (A > C) c--;
                            else c++;
                        } else if (C > B) c++;
                        r++;
                    }
                    
                    if (edgeImg_[r * width_ + c] == EDGE_PIXEL || gradImg_[r * width_ + c] < params_.gradThresh) {
                        if (chainLen > 0) {
                            chains[noChains].len = chainLen;
                            chains[parent].children[1] = noChains;
                            noChains++;
                        }
                        goto StartOfWhile;
                    }
                    
                    pixels[len].y = r;
                    pixels[len].x = c;
                    len++;
                    chainLen++;
                }
                
                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = RIGHT;
                stack[top].parent = noChains;
                
                stack[++top].r = r;
                stack[top].c = c;
                stack[top].dir = LEFT;
                stack[top].parent = noChains;
                
                len--;
                chainLen--;
                
                chains[noChains].len = chainLen;
                chains[parent].children[1] = noChains;
                noChains++;
            }
        }
        
        // 检查路径长度
        if (len - duplicatePixelCount < params_.minPathLen) {
            for (int m = 0; m < len; m++) {
                edgeImg_[pixels[m].y * width_ + pixels[m].x] = 0;
            }
        } else {
            int noSegmentPixels = 0;
            
            int totalLen = longestChain(chains, chains[0].children[1]);
            
            if (totalLen > 0) {
                int count = retrieveChainNos(chains, chains[0].children[1], chainNos);
                
                for (int m = count - 1; m >= 0; m--) {
                    int chainNo = chainNos[m];
                    
                    // 清理重复像素
                    int fr = chains[chainNo].pixels[chains[chainNo].len - 1].y;
                    int fc = chains[chainNo].pixels[chains[chainNo].len - 1].x;
                    
                    int idx = noSegmentPixels - 2;
                    while (idx >= 0) {
                        int dr = std::abs(fr - segmentPoints_[segmentNos_][idx].y);
                        int dc = std::abs(fc - segmentPoints_[segmentNos_][idx].x);
                        
                        if (dr <= 1 && dc <= 1) {
                            segmentPoints_[segmentNos_].pop_back();
                            noSegmentPixels--;
                            idx--;
                        } else break;
                    }
                    
                    if (chains[chainNo].len > 1 && noSegmentPixels > 0) {
                        fr = chains[chainNo].pixels[chains[chainNo].len - 2].y;
                        fc = chains[chainNo].pixels[chains[chainNo].len - 2].x;
                        
                        int dr = std::abs(fr - segmentPoints_[segmentNos_][noSegmentPixels - 1].y);
                        int dc = std::abs(fc - segmentPoints_[segmentNos_][noSegmentPixels - 1].x);
                        
                        if (dr <= 1 && dc <= 1) chains[chainNo].len--;
                    }
                    
                    for (int l = chains[chainNo].len - 1; l >= 0; l--) {
                        segmentPoints_[segmentNos_].emplace_back(chains[chainNo].pixels[l]);
                        noSegmentPixels++;
                    }
                    
                    chains[chainNo].len = 0;
                }
            }
            
            totalLen = longestChain(chains, chains[0].children[0]);
            if (totalLen > 1) {
                int count = retrieveChainNos(chains, chains[0].children[0], chainNos);
                
                int lastChainNo = chainNos[0];
                chains[lastChainNo].pixels++;
                chains[lastChainNo].len--;
                
                for (int m = 0; m < count; m++) {
                    int chainNo = chainNos[m];
                    
                    int fr = chains[chainNo].pixels[0].y;
                    int fc = chains[chainNo].pixels[0].x;
                    
                    int idx = noSegmentPixels - 2;
                    while (idx >= 0) {
                        int dr = std::abs(fr - segmentPoints_[segmentNos_][idx].y);
                        int dc = std::abs(fc - segmentPoints_[segmentNos_][idx].x);
                        
                        if (dr <= 1 && dc <= 1) {
                            segmentPoints_[segmentNos_].pop_back();
                            noSegmentPixels--;
                            idx--;
                        } else break;
                    }
                    
                    int startIndex = 0;
                    int cLen = chains[chainNo].len;
                    if (cLen > 1 && noSegmentPixels > 0) {
                        fr = chains[chainNo].pixels[1].y;
                        fc = chains[chainNo].pixels[1].x;
                        
                        int dr = std::abs(fr - segmentPoints_[segmentNos_][noSegmentPixels - 1].y);
                        int dc = std::abs(fc - segmentPoints_[segmentNos_][noSegmentPixels - 1].x);
                        
                        if (dr <= 1 && dc <= 1) startIndex = 1;
                    }
                    
                    for (int l = startIndex; l < chains[chainNo].len; l++) {
                        segmentPoints_[segmentNos_].emplace_back(chains[chainNo].pixels[l]);
                        noSegmentPixels++;
                    }
                    
                    chains[chainNo].len = 0;
                }
            }
            
            // 清理第一个像素
            if (noSegmentPixels > 1) {
                int fr = segmentPoints_[segmentNos_][1].y;
                int fc = segmentPoints_[segmentNos_][1].x;
                
                int dr = std::abs(fr - segmentPoints_[segmentNos_][noSegmentPixels - 1].y);
                int dc = std::abs(fc - segmentPoints_[segmentNos_][noSegmentPixels - 1].x);
                
                if (dr <= 1 && dc <= 1) {
                    segmentPoints_[segmentNos_].erase(segmentPoints_[segmentNos_].begin());
                    noSegmentPixels--;
                }
            }
            
            segmentNos_++;
            segmentPoints_.emplace_back(std::vector<cv::Point>());
            
            // 复制其余长链
            for (int m = 2; m < noChains; m++) {
                if (chains[m].len < 2) continue;
                
                int totalLen2 = longestChain(chains, m);
                
                if (totalLen2 >= params_.minAuxSegmentLen) {
                    int count = retrieveChainNos(chains, m, chainNos);
                    
                    noSegmentPixels = 0;
                    for (int n = 0; n < count; n++) {
                        int chainNo = chainNos[n];
                        
                        int fr = chains[chainNo].pixels[0].y;
                        int fc = chains[chainNo].pixels[0].x;
                        
                        int idx = noSegmentPixels - 2;
                        while (idx >= 0) {
                            int dr = std::abs(fr - segmentPoints_[segmentNos_][idx].y);
                            int dc = std::abs(fc - segmentPoints_[segmentNos_][idx].x);
                            
                            if (dr <= 1 && dc <= 1) {
                                segmentPoints_[segmentNos_].pop_back();
                                noSegmentPixels--;
                                idx--;
                            } else break;
                        }
                        
                        int startIndex = 0;
                        int cLen = chains[chainNo].len;
                        if (cLen > 1 && noSegmentPixels > 0) {
                            fr = chains[chainNo].pixels[1].y;
                            fc = chains[chainNo].pixels[1].x;
                            
                            int dr = std::abs(fr - segmentPoints_[segmentNos_][noSegmentPixels - 1].y);
                            int dc = std::abs(fc - segmentPoints_[segmentNos_][noSegmentPixels - 1].x);
                            
                            if (dr <= 1 && dc <= 1) startIndex = 1;
                        }
                        
                        for (int l = startIndex; l < chains[chainNo].len; l++) {
                            segmentPoints_[segmentNos_].emplace_back(chains[chainNo].pixels[l]);
                            noSegmentPixels++;
                        }
                        
                        chains[chainNo].len = 0;
                    }
                    
                    segmentPoints_.emplace_back(std::vector<cv::Point>());
                    segmentNos_++;
                }
            }
        }
    }
    
    // 清理
    delete[] A;
    delete[] chains;
    delete[] stack;
    delete[] chainNos;
    delete[] pixels;
}
