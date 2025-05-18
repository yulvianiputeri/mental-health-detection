import pandas as pd
import numpy as np
import re
import nltk
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import warnings

warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')

class AdvancedMentalHealthDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),
            stop_words=None
        )
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Enhanced keyword configuration
        self.mental_health_keywords = {
            'depresi': {
                'core': ['sedih', 'depresi', 'putus asa', 'bunuh diri', 'hampa', 'gabut', 'worthless'],
                'secondary': ['kesepian', 'tidak berharga', 'lelah hidup', 'menyesal'],
                'weight': 1.8
            },
            'kecemasan': {
                'core': ['cemas', 'khawatir', 'panik', 'takut', 'deg-degan', 'overthinking'],
                'secondary': ['jantung berdebar', 'keringat dingin', 'sesak napas', 'drama'],
                'weight': 1.5
            },
            'stress': {
                'core': ['burnout', 'stress', 'tertekan', 'deadline', 'cenat-cenut', 'tumpuk'],
                'secondary': ['kewalahan', 'tekanan', 'tagihan', 'berantakan'],
                'weight': 1.5  # Increased weight
            },
            'normal': {
                'core': ['senang', 'bahagia', 'bersyukur', 'chill', 'asyik', 'syukur'],
                'secondary': ['baik', 'oke', 'legA', 'fun'],
                'weight': 1.0
            }
        }
        
        self.emoji_patterns = {
            'positive': ['😊', '😃', '❤️', '👍', '🎉'],
            'negative': ['😢', '😭', '😔', '💔'],
            'neutral': ['😐', '🤔']
        }

    def extract_advanced_features(self, text):
        features = {}
        text_lower = text.lower()
        
        # Basic features
        features.update({
            'text_length': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
        })
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        features.update({
            'sentiment_positive': sentiment['pos'],
            'sentiment_negative': sentiment['neg'],
            'sentiment_neutral': sentiment['neu'],
            'sentiment_compound': sentiment['compound'],
        })
        
        # Emoji analysis
        features.update({
            'positive_emoji': sum(1 for e in self.emoji_patterns['positive'] if e in text),
            'negative_emoji': sum(1 for e in self.emoji_patterns['negative'] if e in text),
        })
        
        # Keyword scoring
        for condition, keywords in self.mental_health_keywords.items():
            core_count = sum(1 for word in keywords['core'] if word in text_lower)
            secondary_count = sum(1 for word in keywords['secondary'] if word in text_lower)
            features[f'{condition}_score'] = (core_count * 2 + secondary_count) * keywords['weight']
        
        # Burnout special handling
        if 'burnout' in text_lower:
            features['stress_score'] *= 1.2
            
        return features

    def preprocess_text(self, text):
        original = text
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[^a-zA-Z0-9\s\!\?\.\,]', '', text)
        return text, original

    def get_feature_importance(self):
        """Mendapatkan fitur terpenting dari model"""
        if not hasattr(self, 'model') or not hasattr(self, 'feature_columns'):
            return None
        
        # Pastikan model sudah dilatih
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        # Buat DataFrame dari feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        })
        
        # Urutkan berdasarkan importance (descending)
        return importance_df.sort_values('importance', ascending=False)

    def train(self, texts, labels, validate=True):
        features = []
        processed_texts = []
        
        for text in texts:
            processed, original = self.preprocess_text(text)
            processed_texts.append(processed)
            features.append(self.extract_advanced_features(original))
            
        features_df = pd.DataFrame(features)
        tfidf_features = self.vectorizer.fit_transform(processed_texts)
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        combined_features = pd.concat([features_df, tfidf_df], axis=1)
        
        encoded_labels = self.label_encoder.fit_transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, encoded_labels, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Validation
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Training Accuracy: {train_score:.2%}")
        print(f"Testing Accuracy: {test_score:.2%}")
        
        if validate:
            cv_scores = cross_val_score(self.model, combined_features, encoded_labels, cv=5)
            print(f"Cross-Validation: {cv_scores.mean():.2%} ± {cv_scores.std()*2:.2%}")
            y_pred = self.model.predict(X_test)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        self.feature_columns = combined_features.columns
        return train_score, test_score

    def predict(self, text):
        critical_keywords = ['burnout', 'kelelahan mental', 'overwork']
        
        # Critical keyword handling
        if any(kw in text.lower() for kw in critical_keywords):
            return {
                'condition': 'stress',
                'confidence': 0.85,
                'risk_level': 'High',
                'probabilities': {'stress': 0.85, 'normal': 0.05, 'depresi': 0.05, 'kecemasan': 0.05},
                'sentiment': self.sentiment_analyzer.polarity_scores(text),
                'confidence_margin': 0.80
            }
        
        # Normal prediction flow
        processed, original = self.preprocess_text(text)
        features = self.extract_advanced_features(original)
        features_df = pd.DataFrame([features])
        
        tfidf_features = self.vectorizer.transform([processed])
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        combined = pd.concat([features_df, tfidf_df], axis=1).reindex(columns=self.feature_columns, fill_value=0)
        
        prediction = self.model.predict(combined)[0]
        probabilities = self.model.predict_proba(combined)[0]
        
        return {
            'condition': self.label_encoder.inverse_transform([prediction])[0],
            'confidence': float(np.max(probabilities)),
            'confidence_margin': float(np.max(probabilities) - np.sort(probabilities)[-2]),
            'probabilities': {k: float(v) for k, v in zip(self.label_encoder.classes_, probabilities)},
            'sentiment': self.sentiment_analyzer.polarity_scores(text),
            'risk_level': self._calculate_risk_level(
                self.label_encoder.inverse_transform([prediction])[0],
                np.max(probabilities)
            )
        }

    def _calculate_risk_level(self, condition, confidence):
        risk_scores = {'depresi': 2.0, 'kecemasan': 1.5, 'stress': 1.5, 'normal': 0}
        adjusted_risk = risk_scores.get(condition, 0) * confidence
        
        if condition == 'stress' and confidence >= 0.75:
            return "High"
        elif adjusted_risk >= 1.8:
            return "High"
        elif adjusted_risk >= 1.0:
            return "Medium"
        else:
            return "Low"

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'keywords': self.mental_health_keywords
            }, f)

    def load_model(self, filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.vectorizer = data['vectorizer']
                self.label_encoder = data['label_encoder']
                self.feature_columns = data['feature_columns']
                self.mental_health_keywords = data.get('keywords', {})
        except Exception as e:
            print(f"Error loading model: {str(e)}")

        # Tambahkan method ini ke class AdvancedMentalHealthDetector di mental_health_detector.py
    def analyze_chat_history(self, messages, timestamps=None):
        """Menganalisis riwayat chat untuk tren kesehatan mental"""
        if timestamps is None:
            timestamps = [datetime.now() - timedelta(hours=i) for i in range(len(messages))]
        
        if len(messages) != len(timestamps):
            raise ValueError("Jumlah pesan dan timestamps harus sama")
        
        results = {
            'individual_results': [],
            'summary': {}
        }
        
        # Analisis setiap pesan
        for i, (message, timestamp) in enumerate(zip(messages, timestamps)):
            if not message.strip():
                continue
            
            prediction = self.predict(message)
            
            # Tambahkan preview pesan
            preview = message[:30] + "..." if len(message) > 30 else message
            prediction['message_preview'] = preview
            
            results['individual_results'].append(prediction)
        
        # Buat ringkasan
        if results['individual_results']:
            condition_counts = {}
            confidence_sum = 0
            high_risk_count = 0
            
            for result in results['individual_results']:
                condition = result['condition']
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
                confidence_sum += result['confidence']
                if result['risk_level'] == 'High':
                    high_risk_count += 1
            
            # Tentukan kondisi dominan
            dominant_condition = max(condition_counts.items(), key=lambda x: x[1])[0]
            
            # Buat statistik ringkasan
            results['summary'] = {
                'total_messages': len(results['individual_results']),
                'dominant_condition': dominant_condition,
                'high_risk_messages': high_risk_count,
                'average_confidence': confidence_sum / len(results['individual_results']) if results['individual_results'] else 0,
                'condition_distribution': condition_counts
            }
        
        return results

if __name__ == "__main__":
    detector = AdvancedMentalHealthDetector()
    print(detector.predict("Saya merasa burnout berat akhir-akhir ini"))