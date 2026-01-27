/**
 * @file BoardConfig.cpp
 * @brief 标定板配置实现
 */

#include "BoardConfig.h"
#include <iostream>
#include <fstream>

/**
 * @brief 从YAML文件加载配置
 */
bool BoardConfig::loadFromFile(const std::string& filepath) {
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "错误: 无法打开配置文件 " << filepath << std::endl;
        return false;
    }
    
    // 读取标定板参数
    cv::FileNode board_node = fs["board"];
    if (board_node.empty()) {
        std::cerr << "错误: 配置文件缺少 'board' 节点" << std::endl;
        return false;
    }
    
    rows = (int)board_node["rows"];
    cols = (int)board_node["cols"];
    circle_spacing = (double)board_node["circle_spacing"];
    
    // 读取特征锚点配置
    cv::FileNode anchor_node = fs["anchor_circle"]; // 注意用户提到在root下或者board下? 用户之前的diff是在root下的。
    // Wait, the user's snippet shows anchor_circle at root level based on indentation in the snippet?
    // Let's check the snippet again.
    // The snippet:
    // anchor_circle:
    //   - [2, 5]
    // This looks like root level if we look at previous diffs, BUT in the snippet provided by the user:
    // # 标定板特征圆位置
    // ...
    // anchor_circle:
    
    // In YAML 1.0/OpenCV, if it's not indented under "board", it is root.
    // The user's snippet didn't show the "board:" parent explicitly indented wrapping it.
    // However, my previous attempt tried to put it under board and failed indentation.
    // The user's request shows it seemingly at the same level as "segmentor".
    // "segmentor:" is usually at root.
    // So I will assume it is at ROOT.
    
    if (anchor_node.empty()) {
         // Try under board node just in case
         anchor_node = board_node["anchor_circle"];
    }

    if (!anchor_node.empty()) {
        anchors.clear();
        // Check if it is a sequence
        if (anchor_node.type() == cv::FileNode::SEQ) {
             int id = 0;
             for (auto it = anchor_node.begin(); it != anchor_node.end(); ++it, ++id) {
                 cv::FileNode pt_node = *it;
                 if (pt_node.size() == 2) {
                     int r = (int)pt_node[0];
                     int c = (int)pt_node[1];
                     anchors[id] = cv::Point2i(c, r);
                 }
             }
        } else if (anchor_node.type() == cv::FileNode::MAP) {
             // Fallback for map
            for (auto it = anchor_node.begin(); it != anchor_node.end(); ++it) {
                int id = std::stoi((*it).name());
                cv::FileNode pt_node = *it;
                if (pt_node.size() == 2) {
                    int r = (int)pt_node[0];
                    int c = (int)pt_node[1];
                    anchors[id] = cv::Point2i(c, r);
                }
            }
        }
    } else {
        std::cout << "警告: 未找到 'anchor_circle' 配置" << std::endl;
    }
    
    // 读取分割器配置（可选）
    cv::FileNode seg_node = fs["segmentor"];
    if (!seg_node.empty()) {
        if (!seg_node["model_path"].empty()) {
            segmentor.model_path = (std::string)seg_node["model_path"];
        }
        if (!seg_node["inference_size"].empty()) {
            segmentor.inference_size = (int)seg_node["inference_size"];
        }
        if (!seg_node["confidence"].empty()) {
            segmentor.confidence = (float)seg_node["confidence"];
        }
    }
    
    fs.release();
    
    // 验证配置
    if (!isValid()) {
        std::cerr << "错误: 配置文件内容无效" << std::endl;
        return false;
    }
    
    std::cout << "成功加载标定板配置:" << std::endl;
    std::cout << "  尺寸: " << rows << " x " << cols << std::endl;
    std::cout << "  圆心间距: " << circle_spacing << " mm" << std::endl;
    if (!segmentor.model_path.empty()) {
        std::cout << "  分割模型: " << segmentor.model_path << std::endl;
        std::cout << "  推理尺寸: " << segmentor.inference_size << std::endl;
    }
    
    return true;
}

/**
 * @brief 保存配置到YAML文件
 */
bool BoardConfig::saveToFile(const std::string& filepath) const {
    cv::FileStorage fs(filepath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "错误: 无法创建配置文件 " << filepath << std::endl;
        return false;
    }
    
    // 写入标定板参数
    fs << "board" << "{";
    fs << "rows" << rows;
    fs << "cols" << cols;
    fs << "circle_spacing" << circle_spacing;
    fs << "}";
    
    // 写入分割器配置
    if (!segmentor.model_path.empty()) {
        fs << "segmentor" << "{";
        fs << "model_path" << segmentor.model_path;
        fs << "inference_size" << segmentor.inference_size;
        fs << "confidence" << segmentor.confidence;
        fs << "}";
    }
    
    fs.release();
    
    std::cout << "配置已保存到: " << filepath << std::endl;
    return true;
}

/**
 * @brief 验证配置有效性
 */
bool BoardConfig::isValid() const {
    if (rows <= 0 || cols <= 0) {
        std::cerr << "错误: 行数和列数必须大于0" << std::endl;
        return false;
    }
    
    if (circle_spacing <= 0) {
        std::cerr << "错误: 圆心间距必须大于0" << std::endl;
        return false;
    }
    
    return true;
}
