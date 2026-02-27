/**
 * @file main.cpp
 * @brief 双目相机标定主程序
 * 
 * 使用YOLO分割进行标定板检测，结合椭圆拟合提取圆心
 */

#include "StereoCalibrator.h"
#include <iostream>
#include <string>

void printUsage(const char* program_name) {
    std::cout << "\n用法:" << std::endl;
    std::cout << "  " << program_name << " <配置文件> <图像目录> [-d]" << std::endl;
    std::cout << "\n参数说明:" << std::endl;
    std::cout << "  配置文件    - YAML格式的标定板配置文件（包含分割模型路径）" << std::endl;
    std::cout << "  图像目录    - 包含/left和/right子文件夹的根目录" << std::endl;
    std::cout << "  -d          - (可选) 开启调试模式，不管成功与否都将输出图像到debug_img文件夹" << std::endl;
    std::cout << "\n示例:" << std::endl;
    std::cout << "  " << program_name << " board.yaml ./images" << std::endl;
    std::cout << "  " << program_name << " board.yaml ./images -d" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "    双目相机标定系统 (YOLO分割版)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 检查参数
    if (argc < 3 || argc > 4) {
        printUsage(argv[0]);
        return -1;
    }
    
    std::string config_file = argv[1];
    std::string image_dir = argv[2];
    bool debug_mode = false;
    
    if (argc == 4) {
        std::string flag = argv[3];
        if (flag == "-d") {
            debug_mode = true;
        } else {
            std::cout << "未知参数: " << flag << std::endl;
            printUsage(argv[0]);
            return -1;
        }
    }
    
    std::cout << "\n配置文件: " << config_file << std::endl;
    std::cout << "图像目录: " << image_dir << std::endl;
    std::cout << "调试模式: " << (debug_mode ? "已开启 (-d)" : "未开启") << std::endl;
    
    try {
        // 1. 创建标定器（自动加载配置和分割模型）
        StereoCalibrator calibrator(config_file);
        
        // 设置调试模式
        calibrator.setDebugMode(debug_mode);
        
        // 检查分割器状态
        if (!calibrator.hasSegmentor()) {
            std::cerr << "警告: 分割器未就绪，将使用全图检测（可能较慢且不准确）" << std::endl;
        }
        
        // 2. 加载图像
        if (!calibrator.loadImages(image_dir)) {
            std::cerr << "\n标定失败: 图像加载失败" << std::endl;
            return -1;
        }
        
        // 3. 检测圆心
        if (!calibrator.detectCircles()) {
            std::cerr << "\n标定失败: 圆心检测失败" << std::endl;
            return -1;
        }
        
        // 4. 执行标定
        if (!calibrator.calibrate()) {
            std::cerr << "\n标定失败: 标定计算失败" << std::endl;
            return -1;
        }
        
        // 5. 保存结果
        std::string output_yaml = "calibration.yaml";
        std::string output_points = "points_3d.txt";
        
        if (!calibrator.saveResults(output_yaml)) {
            std::cerr << "警告: 标定结果保存失败" << std::endl;
        }
        
        if (!calibrator.export3DPoints(output_points)) {
            std::cerr << "警告: 3D点云输出失败" << std::endl;
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "    标定完成！" << std::endl;
        std::cout << "========================================" << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cerr << "\nOpenCV异常: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "\n异常: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
