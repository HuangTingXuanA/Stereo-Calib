/**
 * @file BoardConfig.cpp
 * @brief 标定板配置实现
 */

#include "BoardConfig.h"
#include <iostream>
#include <fstream>
#include <sstream>

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
    
    // 读取世界坐标生成模式配置
    cv::FileNode auto_gen_node = board_node["auto_generate_coords"];
    if (!auto_gen_node.empty()) {
        auto_generate_coords = (int)auto_gen_node != 0;
    } else {
        auto_generate_coords = true;  // 默认自动生成
    }
    
    cv::FileNode coords_file_node = board_node["coords_file"];
    if (!coords_file_node.empty()) {
        coords_file = (std::string)coords_file_node;
    }
    
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
    } else if (auto_generate_coords) {
        // 仅在自动生成模式下，anchor_circle 是必需的
        std::cout << "警告: 自动生成模式下未找到 'anchor_circle' 配置" << std::endl;
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
    
    // 如果不自动生成坐标，则加载外部坐标文件
    if (!auto_generate_coords) {
        if (coords_file.empty()) {
            std::cerr << "错误: 未指定 3D 坐标文件路径 (coords_file)" << std::endl;
            return false;
        }
        
        // 处理相对路径：相对于配置文件所在目录
        std::string coords_path = coords_file;
        if (coords_file[0] != '/') {
            size_t last_slash = filepath.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                coords_path = filepath.substr(0, last_slash + 1) + coords_file;
            }
        }
        
        if (!loadCoordsFile(coords_path)) {
            std::cerr << "错误: 加载 3D 坐标文件失败: " << coords_path << std::endl;
            return false;
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
    if (auto_generate_coords) {
        std::cout << "  坐标模式: 自动生成 (间距=" << circle_spacing << " mm)" << std::endl;
    } else {
        std::cout << "  坐标模式: 外部文件 (" << coords_file << ")" << std::endl;
        std::cout << "  加载坐标点数: " << world_coords.size() << std::endl;
        std::cout << "  特征锚点标签: ";
        for (size_t i = 0; i < anchor_labels.size(); ++i) {
            std::cout << anchor_labels[i];
            if (i < anchor_labels.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
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
    
    // 自动生成模式需要有效的间距和锚点配置
    if (auto_generate_coords) {
        if (circle_spacing <= 0) {
            std::cerr << "错误: 自动生成模式下圆心间距必须大于0" << std::endl;
            return false;
        }
        if (anchors.size() != 5) {
            std::cerr << "错误: 自动生成模式下需要5个特征锚点，当前: " << anchors.size() << std::endl;
            return false;
        }
    }
    
    // 外部文件模式需要有效的坐标数据
    if (!auto_generate_coords) {
        int expected = rows * cols;
        if (static_cast<int>(world_coords.size()) != expected) {
            std::cerr << "错误: 3D坐标点数(" << world_coords.size() 
                      << ")与标定板尺寸(" << expected << ")不匹配" << std::endl;
            return false;
        }
        if (anchor_labels.size() != 5) {
            std::cerr << "错误: 特征锚点标签数量必须为5，当前: " << anchor_labels.size() << std::endl;
            return false;
        }
    }
    
    return true;
}

/**
 * @brief 从外部文件加载3D坐标
 * 
 * 文件格式:
 *   第一行: 5个特征锚点的标签，空格分隔
 *   后续行: 标签 X Y Z
 */
bool BoardConfig::loadCoordsFile(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        std::cerr << "错误: 无法打开坐标文件 " << filepath << std::endl;
        return false;
    }
    
    world_coords.clear();
    anchor_labels.clear();
    
    std::string line;
    bool first_line = true;
    
    while (std::getline(ifs, line)) {
        // 跳过空行
        if (line.empty()) continue;
        
        // 去除行尾的\r（Windows换行符兼容）
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        
        if (first_line) {
            // 第一行: 读取5个特征锚点标签
            std::string label;
            while (iss >> label) {
                anchor_labels.push_back(label);
            }
            first_line = false;
        } else {
            // 后续行: 标签 X Y Z
            LabeledPoint3D pt;
            if (iss >> pt.label >> pt.coord.x >> pt.coord.y >> pt.coord.z) {
                world_coords.push_back(pt);
            }
        }
    }
    
    ifs.close();
    
    std::cout << "[坐标加载] 文件: " << filepath << std::endl;
    std::cout << "  特征锚点标签: " << anchor_labels.size() << " 个" << std::endl;
    std::cout << "  坐标点数: " << world_coords.size() << " 个" << std::endl;
    
    return true;
}
