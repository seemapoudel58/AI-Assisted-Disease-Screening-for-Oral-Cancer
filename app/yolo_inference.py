import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


class YOLOInferenceEngine:
    
    def __init__(self, model_path: str, class_names: List[str] = None, device: str = 'auto'):
        
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = self._load_model()
        self.class_names = class_names or self._get_class_names()
        self.class_mapping = {i: name for i, name in enumerate(self.class_names)}

    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        return device
    
    def _load_model(self) -> YOLO:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            model = YOLO(self.model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _get_class_names(self) -> List[str]:
        try:
            if hasattr(self.model, 'names') and self.model.names:
                return list(self.model.names.values())
            else:
                return ['Healthy', 'Benign', 'OPMD', 'OCA']
        except:
            return ['Healthy', 'Benign', 'OPMD', 'OCA']
    
    def predict_single_image(self, 
                           image_path: str, 
                           conf_threshold: float = 0.5,
                           iou_threshold: float = 0.45,
                           save_result: bool = False,
                           output_dir: str = None) -> Dict:
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image_rgb.shape[:2]
        
        start_time = time.time()
        results = self.model(
            image_path, 
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )
        inference_time = time.time() - start_time
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_id': cls,
                        'class_name': self.class_names[cls],
                        'area': (x2 - x1) * (y2 - y1)
                    })
        
        result_dict = {
            'image_path': image_path,
            'image_info': {
                'width': original_width,
                'height': original_height,
                'channels': image_rgb.shape[2]
            },
            'detections': detections,
            'num_detections': len(detections),
            'inference_time': inference_time,
            'model_info': {
                'model_path': self.model_path,
                'class_names': self.class_names,
                'device': str(self.device)
            },
            'original_image': image_rgb
        }
        
        if save_result:
            self._save_single_result(result_dict, output_dir)
        
        return result_dict
    
    def predict_batch(self, 
                     image_paths: List[str], 
                     conf_threshold: float = 0.5,
                     iou_threshold: float = 0.45,
                     save_results: bool = False,
                     output_dir: str = None,
                     batch_size: int = 8) -> List[Dict]:
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                result = self.predict_single_image(
                    image_path, 
                    conf_threshold, 
                    iou_threshold, 
                    save_results, 
                    output_dir
                )
                results.append(result)
                
            except Exception as e:
                print(f" Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'detections': [],
                    'num_detections': 0
                })
        
        return results
    
    def predict_directory(self, 
                         input_dir: str, 
                         conf_threshold: float = 0.5,
                         iou_threshold: float = 0.45,
                         save_results: bool = True,
                         output_dir: str = None,
                         image_extensions: List[str] = None) -> List[Dict]:
        
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(Path(input_dir).glob(f"*{ext}"))
            image_paths.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"  No images found in {input_dir}")
            return []
        
        print(f" Found {len(image_paths)} images in {input_dir}")
        
        if output_dir is None:
            output_dir = os.path.join(input_dir, 'yolo_results')
        
        return self.predict_batch(
            image_paths, 
            conf_threshold, 
            iou_threshold, 
            save_results, 
            output_dir
        )
    
    def _save_single_result(self, result: Dict, output_dir: str = None):
        if output_dir is None:
            output_dir = 'yolo_results'
        
        os.makedirs(output_dir, exist_ok=True)
        
        image = result['original_image'].copy()
        detections = result['detections']
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[class_id % len(colors)]
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {conf:.3f}"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        summary_text = f"Detections: {len(detections)} | Time: {result['inference_time']:.3f}s"
        cv2.putText(image, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        image_name = os.path.basename(result['image_path'])
        name, ext = os.path.splitext(image_name)
        output_path = os.path.join(output_dir, f"{name}_result{ext}")
        
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f" Result saved: {output_path}")
    
    def save_results_summary(self, results: List[Dict], output_path: str):
       
        summary = {
            'total_images': len(results),
            'successful_predictions': len([r for r in results if 'error' not in r]),
            'failed_predictions': len([r for r in results if 'error' in r]),
            'total_detections': sum(r.get('num_detections', 0) for r in results),
            'average_inference_time': np.mean([r.get('inference_time', 0) for r in results if 'inference_time' in r]),
            'class_statistics': self._calculate_class_statistics(results),
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f" Results summary saved: {output_path}")
        return summary
    
    def _calculate_class_statistics(self, results: List[Dict]) -> Dict:
        class_counts = {name: 0 for name in self.class_names}
        class_confidences = {name: [] for name in self.class_names}
        
        for result in results:
            if 'detections' in result:
                for detection in result['detections']:
                    class_name = detection['class_name']
                    class_counts[class_name] += 1
                    class_confidences[class_name].append(detection['confidence'])
        
        class_stats = {}
        for class_name in self.class_names:
            confidences = class_confidences[class_name]
            class_stats[class_name] = {
                'count': class_counts[class_name],
                'average_confidence': np.mean(confidences) if confidences else 0.0,
                'max_confidence': np.max(confidences) if confidences else 0.0,
                'min_confidence': np.min(confidences) if confidences else 0.0
            }
        
        return class_stats
    
    def create_visualization_report(self, results: List[Dict], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        successful_results = [r for r in results if 'error' not in r and 'detections' in r]
        
        if not successful_results:
            print(" No successful results to visualize")
            return
        
        self._plot_class_distribution(successful_results, output_dir)
        
        self._plot_confidence_distribution(successful_results, output_dir)
        
        self._create_sample_visualizations(successful_results, output_dir)

        print(f" Visualization report created in: {output_dir}")
    
    def _plot_class_distribution(self, results: List[Dict], output_dir: str):
        class_counts = {name: 0 for name in self.class_names}
        
        for result in results:
            for detection in result['detections']:
                class_counts[detection['class_name']] += 1
        
        plt.figure(figsize=(10, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.bar(classes, counts, color=['red', 'green', 'blue', 'orange'][:len(classes)])
        plt.title('Class Distribution in Detections')
        plt.xlabel('Class')
        plt.ylabel('Number of Detections')
        plt.xticks(rotation=45)
        
        for i, count in enumerate(counts):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, results: List[Dict], output_dir: str):
        all_confidences = []
        class_confidences = {name: [] for name in self.class_names}
        
        for result in results:
            for detection in result['detections']:
                conf = detection['confidence']
                all_confidences.append(conf)
                class_confidences[detection['class_name']].append(conf)
        
        if not all_confidences:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.hist(all_confidences, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_title('Overall Confidence Distribution')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(all_confidences), color='red', linestyle='--', label=f'Mean: {np.mean(all_confidences):.3f}')
        ax1.legend()
        
        for i, (class_name, confidences) in enumerate(class_confidences.items()):
            if confidences:
                ax2.hist(confidences, bins=10, alpha=0.6, label=class_name)
        
        ax2.set_title('Confidence Distribution by Class')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sample_visualizations(self, results: List[Dict], output_dir: str, max_samples: int = 9):
       
        samples_with_detections = [r for r in results if r['num_detections'] > 0]
        
        if not samples_with_detections:
            print(" No samples with detections found")
            return
        
        selected_samples = samples_with_detections[:max_samples]
        
        n_samples = len(selected_samples)
        n_cols = min(3, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_samples == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, result in enumerate(selected_samples):
            ax = axes[i]
            
            image = result['original_image'].copy()
            detections = result['detections']
            
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                conf = detection['confidence']
                class_name = detection['class_name']
                class_id = detection['class_id']
                
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                color = colors[class_id % len(colors)]
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name}: {conf:.3f}"
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            ax.imshow(image)
            ax.set_title(f"{os.path.basename(result['image_path'])}\n{len(detections)} detections")
            ax.axis('off')
        
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_detections.png'), dpi=300, bbox_inches='tight')
        plt.close()


