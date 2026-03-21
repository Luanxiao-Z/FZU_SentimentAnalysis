# ==========================================
# 模型操作类：加载、预测逻辑
# ==========================================

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
from .config import MODEL_CONFIG, EMOTION_NAMES, COARSE_MAP, NUM_LABELS


class EmotionModelHandler:
    """细粒度情感分析模型处理器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化模型处理器
        
        Args:
            model_path: 模型路径，默认使用 config 中的配置
        """
        self.model_path = model_path or MODEL_CONFIG['model_path']
        self.tokenizer = None
        self.model = None
        self.device = None
        self.config = None
    
    def load_model(self):
        """加载模型和配置"""
        # 加载配置文件
        config_path = Path(self.model_path) / 'emotion_config.json'
        import json
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 转换键为整数类型
        self.config['emotion_names'] = {int(k): v for k, v in self.config['emotion_names'].items()}
        self.config['coarse_map'] = {int(k): v for k, v in self.config['coarse_map'].items()}
        
        # 加载分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        
        # 加载模型
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        return self
    
    def predict(self, text: str) -> dict:
        """
        预测单条文本的情感
        
        Args:
            text: 输入文本
            
        Returns:
            预测结果字典
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        if not text or not text.strip():
            raise ValueError("输入文本不能为空或仅包含空白字符。")
        
        # 编码
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=MODEL_CONFIG['truncation'],
            max_length=MODEL_CONFIG['max_length'],
            padding=MODEL_CONFIG['padding']
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 处理结果
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        pred_id = torch.argmax(probabilities).item()
        confidence = probabilities[pred_id].item()
        
        # 所有情感的概率
        all_probs = {
            EMOTION_NAMES[i]: probabilities[i].item() 
            for i in range(int(NUM_LABELS))
        }
        
        # 粗粒度判断
        coarse = COARSE_MAP[pred_id]
        
        # Top3 情感
        top3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'fine_id': pred_id,
            'fine_name': EMOTION_NAMES[pred_id],
            'coarse': coarse,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'top3': top3
        }
    
    def predict_batch(self, texts: list) -> list:
        """
        批量预测文本情感
        
        Args:
            texts: 文本列表
            
        Returns:
            预测结果列表
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results