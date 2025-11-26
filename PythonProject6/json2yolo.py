"""
COCO格式转YOLO格式转换工具
支持分割（segmentation）和边界框（bbox）标注的转换

功能：
1. 支持RLE格式的segmentation（通过pycocotools解码）
2. 支持polygon格式的segmentation
3. 支持bbox标注
4. 输出YOLO格式的txt文件

使用方法：
    python json2yolo.py --coco <COCO_JSON路径> --out <YOLO输出文件夹>
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

try:
    from pycocotools import mask as mask_utils
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("警告: pycocotools未安装，RLE格式的segmentation将无法解码。")
    print("请运行: pip install pycocotools")


def decode_rle_to_polygon(rle: Dict[str, Any], img_width: int, img_height: int) -> List[List[float]]:
    """
    将RLE格式的segmentation解码为polygon格式
    
    Args:
        rle: RLE格式的segmentation字典，包含counts和size
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        多边形点列表，格式为 [[x1, y1, x2, y2, ...], ...]
    """
    if not PYCOCOTOOLS_AVAILABLE:
        raise ImportError("pycocotools未安装，无法解码RLE格式")
    
    # 解码RLE为mask
    binary_mask = mask_utils.decode(rle)
    
    # 从mask提取轮廓
    import cv2
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if len(contour) < 3:  # 至少需要3个点才能形成多边形
            continue
        
        # 将轮廓点转换为列表格式 [x1, y1, x2, y2, ...]
        polygon = []
        for point in contour:
            polygon.extend([float(point[0][0]), float(point[0][1])])
        
        if len(polygon) >= 6:  # 至少3个点（6个坐标值）
            polygons.append(polygon)
    
    return polygons


def convert_coco_to_yolo(
    coco_json_path: str,
    yolo_output_folder: str,
    use_segmentation: bool = True,
    use_bbox: bool = True
) -> None:
    """
    将COCO格式的标注文件转换为YOLO格式
    
    Args:
        coco_json_path: COCO格式JSON文件的路径
        yolo_output_folder: YOLO格式标注文件的输出文件夹
        use_segmentation: 是否转换segmentation标注（默认True）
        use_bbox: 是否转换bbox标注（默认True）
    """
    # 创建输出目录
    yolo_output_folder = Path(yolo_output_folder)
    yolo_output_folder.mkdir(parents=True, exist_ok=True)
    
    # 读取COCO JSON文件
    print(f"正在读取COCO文件: {coco_json_path}")
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    
    # 建立image_id -> 图像信息映射
    image_info = {}
    for img in coco["images"]:
        image_info[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"]
        }
    
    # 建立category_id -> YOLO类别ID映射（从0开始）
    categories = sorted(coco["categories"], key=lambda x: x["id"])
    category_map = {cat["id"]: idx for idx, cat in enumerate(categories)}
    
    # 打印类别信息
    print(f"\n类别映射:")
    for cat_id, yolo_id in category_map.items():
        cat_name = next(cat["name"] for cat in categories if cat["id"] == cat_id)
        print(f"  COCO ID {cat_id} ({cat_name}) -> YOLO ID {yolo_id}")
    
    # 按图像ID组织标注
    annotations_by_image: Dict[int, List[Dict]] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # 转换每个图像的标注
    total_images = len(image_info)
    processed_images = 0
    total_annotations = 0
    
    print(f"\n开始转换 {total_images} 张图像的标注...")
    
    for img_id, img_data in image_info.items():
        img_w = img_data["width"]
        img_h = img_data["height"]
        file_name = img_data["file_name"]
        
        # 获取该图像的所有标注
        annotations = annotations_by_image.get(img_id, [])
        
        yolo_lines = []
        
        for ann in annotations:
            yolo_class = category_map[ann["category_id"]]
            
            # 处理segmentation
            if use_segmentation and "segmentation" in ann:
                seg = ann["segmentation"]
                
                # 判断segmentation格式
                if isinstance(seg, dict):
                    # RLE格式
                    if "counts" in seg and "size" in seg:
                        try:
                            polygons = decode_rle_to_polygon(seg, img_w, img_h)
                            for polygon in polygons:
                                # 归一化坐标
                                normalized_seg = []
                                for i in range(0, len(polygon), 2):
                                    px = polygon[i] / img_w
                                    py = polygon[i + 1] / img_h
                                    # 确保坐标在[0, 1]范围内
                                    px = max(0.0, min(1.0, px))
                                    py = max(0.0, min(1.0, py))
                                    normalized_seg.append(f"{px:.6f} {py:.6f}")
                                
                                if len(normalized_seg) >= 3:  # 至少3个点
                                    seg_line = f"{yolo_class} " + " ".join(normalized_seg)
                                    yolo_lines.append(seg_line)
                        except Exception as e:
                            print(f"警告: 图像 {file_name} 的RLE解码失败: {e}")
                            # 如果RLE解码失败，尝试使用bbox
                            if use_bbox and "bbox" in ann:
                                x, y, w, h = ann["bbox"]
                                xc = (x + w / 2) / img_w
                                yc = (y + h / 2) / img_h
                                w_norm = w / img_w
                                h_norm = h / img_h
                                bbox_line = f"{yolo_class} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}"
                                yolo_lines.append(bbox_line)
                
                elif isinstance(seg, list):
                    # Polygon格式
                    for polygon in seg:
                        if not isinstance(polygon, list) or len(polygon) < 6:
                            continue
                        
                        # 归一化坐标
                        normalized_seg = []
                        for i in range(0, len(polygon), 2):
                            px = polygon[i] / img_w
                            py = polygon[i + 1] / img_h
                            # 确保坐标在[0, 1]范围内
                            px = max(0.0, min(1.0, px))
                            py = max(0.0, min(1.0, py))
                            normalized_seg.append(f"{px:.6f} {py:.6f}")
                        
                        if len(normalized_seg) >= 3:  # 至少3个点
                            seg_line = f"{yolo_class} " + " ".join(normalized_seg)
                            yolo_lines.append(seg_line)
            
            # 处理bbox（如果没有segmentation或用户要求）
            elif use_bbox and "bbox" in ann:
                x, y, w, h = ann["bbox"]
                # 转换为YOLO格式：中心点坐标和宽高（归一化）
                xc = (x + w / 2) / img_w
                yc = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # 确保坐标在[0, 1]范围内
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))
                
                bbox_line = f"{yolo_class} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}"
                yolo_lines.append(bbox_line)
        
        # 写入YOLO格式的txt文件
        if yolo_lines:
            txt_name = Path(file_name).stem + ".txt"
            txt_path = yolo_output_folder / txt_name
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
            
            total_annotations += len(yolo_lines)
            processed_images += 1
    
    # 保存类别名称文件（可选）
    names_file = yolo_output_folder.parent / "classes.txt"
    with open(names_file, "w", encoding="utf-8") as f:
        for cat in categories:
            f.write(f"{cat['name']}\n")
    
    print(f"\n✓ 转换完成！")
    print(f"  - 处理图像数: {processed_images}/{total_images}")
    print(f"  - 总标注数: {total_annotations}")
    print(f"  - 输出目录: {yolo_output_folder}")
    print(f"  - 类别文件: {names_file}")


def main():
    parser = argparse.ArgumentParser(
        description="将COCO格式的标注文件转换为YOLO格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python json2yolo.py --coco annotations/train_annotations.coco.json --out labels/train
  python json2yolo.py --coco annotations/train_annotations.coco.json --out labels/train --no-segmentation
  python json2yolo.py --coco annotations/train_annotations.coco.json --out labels/train --no-bbox
        """
    )
    
    # parser.add_argument(
    #     "--coco",
    #     type=str,
    #     required=True,
    #     help="COCO格式JSON文件的路径"
    # )
    
    # parser.add_argument(
    #     "--out",
    #     type=str,
    #     required=True,
    #     help="YOLO格式标注文件的输出文件夹"
    # )
    
    # parser.add_argument(
    #     "--no-segmentation",
    #     action="store_true",
    #     help="不转换segmentation标注，只转换bbox"
    # )
    
    # parser.add_argument(
    #     "--no-bbox",
    #     action="store_true",
    #     help="不转换bbox标注，只转换segmentation"
    # )
    
    args = parser.parse_args()
    

    # 检查COCO文件是否存在
    # if not os.path.exists(args.coco):
    #     print(f"错误: COCO文件不存在: {args.coco}")
    #     return
    
    # 执行转换
    convert_coco_to_yolo(
        coco_json_path=r"C:\Users\S2069\PycharmProjects\PythonProject6\_annotations.coco.json",
        yolo_output_folder=r"C:\Users\S2069\PycharmProjects\PythonProject6\labels\train",
        # use_segmentation=not args.no_segmentation,
        # use_bbox=not args.no_bbox
    )


if __name__ == "__main__":
    main()


