/**
 * @file dsf.cpp
 * @brief 边缘检测与平滑圆弧提取模块实现
 * 
 * 从 EDSF 项目严格移植的边缘检测算法
 */

#include "dsf.hpp"
#include <algorithm>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// 边缘检测器实现
// ============================================================================

EdgeDetector::EdgeDetector(const cv::Mat& srcImage, const EdgeParams& params)
    : params_(params) {
    
    // 参数合法性检查
    if (params_.gradThresh < 1) params_.gradThresh = 1;
    if (params_.anchorThresh < 0) params_.anchorThresh = 0;
    if (params_.sigma < 0) params_.sigma = 0;
    
    segmentPoints_.reserve(50);
    
    // 转换为灰度图像
    if (srcImage.channels() == 3) {
        cv::cvtColor(srcImage, srcImage_, cv::COLOR_BGR2GRAY);
    } else {
        srcImage_ = srcImage.clone();
    }
    height_ = srcImage_.rows;
    width_ = srcImage_.cols;
    
    segmentNos_ = 0;
    segmentPoints_.emplace_back(std::vector<cv::Point>()); // 预分配空白段
    
    // 初始化图像
    edgeImage_ = cv::Mat(height_, width_, CV_8UC1, cv::Scalar(0));
    smoothImage_ = cv::Mat(height_, width_, CV_8UC1);
    gradImage_ = cv::Mat(height_, width_, CV_16SC1);
    
    srcImg_ = srcImage_.data;
    
    // 高斯平滑处理
    if (params_.sigma > 0) {
        if (params_.sigma == 1.0) {
            cv::GaussianBlur(srcImage_, smoothImage_, cv::Size(5, 5), params_.sigma);
        } else {
            cv::GaussianBlur(srcImage_, smoothImage_, cv::Size(), params_.sigma);
        }
    } else {
        smoothImage_ = srcImage_.clone(); // 禁用平滑
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
    
    // 清理临时数据
    delete[] dirImg_;
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
            
            int sum = gx + gy;
            int index = i * width_ + j;
            gradImg_[index] = static_cast<short>(sum);
            
            if (sum >= params_.gradThresh) {
                if (gx >= gy) {
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
            if (gradImg_[i * width_ + j] < params_.gradThresh) continue;
            
            if (dirImg_[i * width_ + j] == EDGE_VERTICAL) {
                // 垂直边缘
                int diff1 = gradImg_[i * width_ + j] - gradImg_[i * width_ + j - 1];
                int diff2 = gradImg_[i * width_ + j] - gradImg_[i * width_ + j + 1];
                if (diff1 >= params_.anchorThresh && diff2 >= params_.anchorThresh) {
                    edgeImg_[i * width_ + j] = ANCHOR_PIXEL;
                    anchorPoints_.emplace_back(cv::Point(j, i));
                }
            } else {
                // 水平边缘
                int diff1 = gradImg_[i * width_ + j] - gradImg_[(i - 1) * width_ + j];
                int diff2 = gradImg_[i * width_ + j] - gradImg_[(i + 1) * width_ + j];
                if (diff1 >= params_.anchorThresh && diff2 >= params_.anchorThresh) {
                    edgeImg_[i * width_ + j] = ANCHOR_PIXEL;
                    anchorPoints_.emplace_back(cv::Point(j, i));
                }
            }
        }
    }
    
    anchorNos_ = static_cast<int>(anchorPoints_.size());
}

void EdgeDetector::sortAnchorsByGradValue() {
    auto sortFunc = [this](const cv::Point& a, const cv::Point& b) {
        return gradImg_[a.y * width_ + a.x] > gradImg_[b.y * width_ + b.x];
    };
    std::sort(anchorPoints_.begin(), anchorPoints_.end(), sortFunc);
}

int EdgeDetector::longestChain(Chain* chains, int root) {
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

int EdgeDetector::retrieveChainNos(Chain* chains, int root, int chainNos[]) {
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

void EdgeDetector::joinAnchorPointsUsingSortedAnchors() {
    int* chainNos = new int[(width_ + height_) * 8];
    cv::Point* pixels = new cv::Point[width_ * height_];
    StackNode* stack = new StackNode[width_ * height_];
    Chain* chains = new Chain[width_ * height_];
    
    // 按梯度值排序锚点
    const int SIZE = 128 * 256;
    int* C = new int[SIZE];
    std::memset(C, 0, sizeof(int) * SIZE);
    
    // 统计梯度值
    for (int i = 1; i < height_ - 1; i++) {
        for (int j = 1; j < width_ - 1; j++) {
            if (edgeImg_[i * width_ + j] != ANCHOR_PIXEL) continue;
            int grad = gradImg_[i * width_ + j];
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
                
                if (totalLen2 >= 10) {
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
    
    // 移除最后一个空段
    segmentPoints_.pop_back();
    
    // 清理
    delete[] A;
    delete[] chains;
    delete[] stack;
    delete[] chainNos;
    delete[] pixels;
}

// ============================================================================
// RDP 并行处理实现
// ============================================================================

ParallelRDP::ParallelRDP(std::vector<cv::Point>* edgeLists,
                         std::vector<std::pair<int, int>>* segLists,
                         double epsilon, int num, int threads)
    : edgeLists_(edgeLists), segLists_(segLists),
      epsilon_(epsilon), threads_(threads), num_(num) {
    range_ = num / threads;
}

double ParallelRDP::perpendicularDistance2(const cv::Point& pt,
                                            const cv::Point& lineStart,
                                            const cv::Point& lineEnd) {
    double dx = lineEnd.x - lineStart.x;
    double dy = lineEnd.y - lineStart.y;
    
    double mag = std::sqrt(dx * dx + dy * dy);
    if (mag > 0.0) {
        dx /= mag;
        dy /= mag;
    }
    
    double pvx = pt.x - lineStart.x;
    double pvy = pt.y - lineStart.y;
    
    double pvdot = dx * pvx + dy * pvy;
    
    double dsx = pvdot * dx;
    double dsy = pvdot * dy;
    
    double ax = pvx - dsx;
    double ay = pvy - dsy;
    
    return ax * ax + ay * ay;
}

void ParallelRDP::rdp(const std::vector<cv::Point>& edge,
                      int l, int r, double epsilon, int id) const {
    if (r - l < 2) return;
    
    double dMax = 0;
    int idx = 0;
    
    for (int i = l + 1; i < r; i++) {
        double d = perpendicularDistance2(edge[i], edge[l], edge[r - 1]);
        if (d > dMax) {
            idx = i;
            dMax = d;
        }
    }
    
    if (dMax > epsilon) {
        rdp(edge, l, idx + 1, epsilon, id);
        rdp(edge, idx, r, epsilon, id);
    } else {
        segLists_[id].emplace_back(std::make_pair(l, r));
    }
}

void ParallelRDP::operator()(const cv::Range& r) const {
    for (int v = r.start; v < num_; v += threads_) {
        if (edgeLists_[v].size() > 2) {
            segLists_[v].reserve(10);
            rdp(edgeLists_[v], 0, static_cast<int>(edgeLists_[v].size()),
                epsilon_ * epsilon_, v);
        }
    }
}

// ============================================================================
// 圆弧提取器实现
// ============================================================================

ArcExtractor::ArcExtractor(const std::vector<std::vector<cv::Point>>& edges,
                           const ArcParams& params)
    : params_(params), edges_(edges) {
}

void ArcExtractor::extract() {
    runRDP();
    splitEdge();
    
    // 对圆弧进行第二次 RDP
    arcSegs_.resize(arcs_.size());
    std::sort(arcs_.begin(), arcs_.end(),
              [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                  return a.size() > b.size();
              });
    
    // 初始化极性数组
    polarities_.resize(arcs_.size(), 0);

    cv::parallel_for_(cv::Range(0, params_.threads),
                      ParallelRDP(arcs_.data(), arcSegs_.data(),
                                  params_.epsilon, static_cast<int>(arcs_.size()),
                                  params_.threads));
}

void ArcExtractor::computePolarity(const cv::Mat& gradImage) {
    if (gradImage.empty()) {
        std::fill(polarities_.begin(), polarities_.end(), 0);
        return;
    }

    int width = gradImage.cols;
    int height = gradImage.rows;

    // 计算梯度方向
    cv::Mat dx, dy;
    cv::Mat smoothImage;
    // 需要平滑后的图像来计算梯度，或者直接使用输入的gradImage如果它包含了梯度方向信息
    // 注意：EdgeResult 中的 gradImage 存储的是梯度幅值 (short 类型)。
    // 这里我们假设传入的是原始平滑图像或者需要重新计算梯度。
    // 但是 computePolarity 接口只接受 gradImage。
    // 实际上，我们需要基于 smoothImage 计算 dx, dy 来获得梯度方向。
    // 这里修改实现，假设传入的是平滑后的图像 smoothImage，而非梯度幅值图。
    // 如果传入的是 EdgeResult.gradImage (幅值)，则无法计算方向。
    // 但是 detectEdges 返回的 EdgeResult 包含 smoothImage。
    // 因此调用者应该传入 smoothImage。
    
    // 为了稳健性，我们在函数内部计算 Sobel 梯度
    cv::Sobel(gradImage, dx, CV_32F, 1, 0, 3);
    cv::Sobel(gradImage, dy, CV_32F, 0, 1, 3);

    for (size_t i = 0; i < arcs_.size(); ++i) {
        const auto& arc = arcs_[i];
        if (arc.size() < 5) { // Increase min points for polarity check
            polarities_[i] = 0;
            continue;
        }

        // Calculate Chord Center (Midpoint of start and end)
        cv::Point start = arc.front();
        cv::Point end = arc.back();
        cv::Point2f chordMid = (cv::Point2f(start) + cv::Point2f(end)) * 0.5f;

        double totalProj = 0.0;
        int count = 0;

        for (size_t j = 1; j < arc.size() - 1; ++j) {
            cv::Point p = arc[j];
            if (p.x < 0 || p.x >= width || p.y < 0 || p.y >= height) continue;

            // Gradient Vector (vg)
            float gx = dx.at<float>(p);
            float gy = dy.at<float>(p);
            double gradMag = std::sqrt(gx * gx + gy * gy);
            
            if (gradMag < 1e-5) continue;
            
            double vg_x = gx / gradMag;
            double vg_y = gy / gradMag;
            
            // Inward Vector (v_in): From Point to Chord Center
            // For a convex arc on an ellipse, the chord is strictly inside.
            double v_in_x = chordMid.x - p.x;
            double v_in_y = chordMid.y - p.y;
            double v_in_mag = std::sqrt(v_in_x * v_in_x + v_in_y * v_in_y);
            
            if (v_in_mag < 1e-5) continue;
            
            v_in_x /= v_in_mag;
            v_in_y /= v_in_mag;
            
            // Dot product
            double dot = vg_x * v_in_x + vg_y * v_in_y;
            totalProj += dot;
            count++;
        }

        if (count > 0) {
            double avgProj = totalProj / count;
            // Use slightly stricter threshold
            if (avgProj > 0.05) polarities_[i] = 1;       // Gradient points Inward (Bright Ellipse / Anti-parallel) -> Polarity 1
            else if (avgProj < -0.05) polarities_[i] = -1; // Gradient points Outward (Dark Ellipse / Parallel) -> Polarity -1
            else polarities_[i] = 0;
        } else {
            polarities_[i] = 0;
        }
    }
}

void ArcExtractor::runRDP() {
    segList_.resize(edges_.size());
    
    // 按长度排序边缘
    std::vector<std::vector<cv::Point>> sortedEdges = edges_;
    std::sort(sortedEdges.begin(), sortedEdges.end(),
              [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                  return a.size() > b.size();
              });
    
    edges_ = sortedEdges;
    
    cv::parallel_for_(cv::Range(0, params_.threads),
                      ParallelRDP(edges_.data(), segList_.data(),
                                  params_.epsilon, static_cast<int>(edges_.size()),
                                  params_.threads));
}

void ArcExtractor::splitEdge() {
    double thAngle = std::cos(params_.sharpAngle * M_PI / 180.0);
    
    for (size_t i = 0; i < segList_.size(); i++) {
        if (edges_[i].size() < static_cast<size_t>(params_.minArcLength)) break;
        if (segList_[i].size() < 2) continue;
        
        int preIndex = 0;
        int preSegId = 0;
        auto& seg = segList_[i];
        
        for (size_t j = 1; j < seg.size(); j++) {
            bool isCornerPoint = false, isInflectionPoint = false;
            
            cv::Point l2_v = edges_[i][seg[j - 1].second - 1] - edges_[i][seg[j - 1].first];
            cv::Point l3_v = edges_[i][seg[j].second - 1] - edges_[i][seg[j].first];
            
            double norm1 = cv::norm(l2_v);
            double norm2 = cv::norm(l3_v);
            if (norm1 > 0 && norm2 > 0) {
                double angle = l2_v.dot(l3_v) / (norm1 * norm2);
                if (angle <= thAngle) {
                    isCornerPoint = true;
                }
            }
            
            if (j >= 2) {
                cv::Point l1_v = edges_[i][seg[j - 2].second - 1] - edges_[i][seg[j - 2].first];
                if (l1_v.cross(l2_v) * l2_v.cross(l3_v) < 0) {
                    isInflectionPoint = true;
                }
            }
            
            if (isCornerPoint || isInflectionPoint) {
                int len = seg[j].first - preIndex;
                if (len > params_.minArcLength && static_cast<int>(j) - preSegId > 1) {
                    arcs_.emplace_back(std::vector<cv::Point>(
                        edges_[i].begin() + preIndex,
                        edges_[i].begin() + seg[j].first));
                    
                    std::vector<cv::Point>& arc = arcs_.back();
                    auto v1 = arc[arc.size() / 2] - arc.front();
                    auto v2 = arc.back() - arc[arc.size() / 2];
                    if (v1.cross(v2) > 0) {
                        std::reverse(arc.begin(), arc.end());
                    }
                }
                preIndex = seg[j].first;
                preSegId = static_cast<int>(j);
            }
        }
        
        int len = seg.back().second - preIndex;
        if (len >= params_.minArcLength && static_cast<int>(seg.size()) - preSegId > 1) {
            arcs_.emplace_back(std::vector<cv::Point>(
                edges_[i].begin() + preIndex,
                edges_[i].begin() + seg.back().second));
            
            std::vector<cv::Point>& arc = arcs_.back();
            auto v1 = arc[arc.size() / 2] - arc.front();
            auto v2 = arc.back() - arc[arc.size() / 2];
            if (v1.cross(v2) > 0) {
                std::reverse(arc.begin(), arc.end());
            }
        }
    }
}

// ============================================================================
// 便捷函数接口
// ============================================================================

EdgeResult detectEdges(const cv::Mat& image, const EdgeParams& params) {
    EdgeDetector detector(image, params);
    
    EdgeResult result;
    result.edges = detector.getSegments();
    result.smoothImage = detector.getSmoothImage();
    result.gradImage = detector.getGradImage();
    result.dirImage = detector.getDirImage();
    result.width = detector.getWidth();
    result.height = detector.getHeight();
    
    return result;
}

ArcResult extractArcs(const EdgeResult& edgeResult,
                      const ArcParams& params) {
    ArcExtractor extractor(edgeResult.edges, params);
    extractor.extract();
    extractor.computePolarity(edgeResult.smoothImage); // 使用平滑图像计算极性
    
    ArcResult result;
    result.arcs = extractor.getArcs();
    result.arcSegs = extractor.getArcSegs();
    result.polarities = extractor.getPolarities();
    
    return result;
}
