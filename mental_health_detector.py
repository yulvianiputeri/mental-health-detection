# mental_health_detector.py

import pandas as pd
import numpy as np
import re
import nltk
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download data NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')

class AdvancedMentalHealthDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),
            stop_words=None  # Menggunakan None karena dataset dalam bahasa Indonesia
        )
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Kamus kata kunci yang ditingkatkan (dalam bahasa Indonesia)
        self.mental_health_keywords = {
            'depresi': {
                'core': ['sedih', 'depresi', 'putus asa', 'bunuh diri', 'mati', 'hampa', 'kosong', 'gagal'],
                'secondary': ['kesepian', 'tidak berharga', 'tidak berguna', 'lelah hidup', 'menyesal', 'membenci diri'],
                'weight': 2.0
            },
            'kecemasan': {
                'core': ['cemas', 'khawatir', 'panik', 'takut', 'gelisah', 'deg-degan'],
                'secondary': ['jantung berdebar', 'keringat dingin', 'overthinking', 'sesak napas', 'panic attack'],
                'weight': 1.8
            },
            'stress': {
                'core': ['stress', 'tertekan', 'beban', 'overwhelmed', 'pusing', 'deadline'],
                'secondary': ['burned out', 'burnout', 'lelah mental', 'kewalahan', 'capek', 'tekanan'],
                'weight': 1.5
            },
            'normal': {
                'core': ['senang', 'bahagia', 'bersyukur', 'optimis', 'semangat', 'syukur'],
                'secondary': ['produktif', 'tenang', 'damai', 'grateful', 'excited', 'content'],
                'weight': 1.0
            }
        }
        
        # Pola emoji
        self.emoji_patterns = {
            'positive': ['😊', '😃', '🙂', '😄', '❤️', '👍', '🎉', '✨', '🥰', '😍'],
            'negative': ['😢', '😭', '😔', '😟', '😰', '😱', '💔', '😞', '😡', '😤'],
            'neutral': ['😐', '🤔', '😑', '👌', '🤷']
        }
        
    def extract_advanced_features(self, text):
        """Ekstrak fitur lanjutan termasuk sentimen, emoji, dan panjang"""
        features = {}
        
        # Fitur teks dasar
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # Fitur sentimen
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        features['sentiment_positive'] = sentiment_scores['pos']
        features['sentiment_negative'] = sentiment_scores['neg']
        features['sentiment_neutral'] = sentiment_scores['neu']
        features['sentiment_compound'] = sentiment_scores['compound']
        
        # Fitur emoji
        positive_emojis = sum(1 for emoji in self.emoji_patterns['positive'] if emoji in text)
        negative_emojis = sum(1 for emoji in self.emoji_patterns['negative'] if emoji in text)
        features['positive_emoji_count'] = positive_emojis
        features['negative_emoji_count'] = negative_emojis
        
        # Fitur kata kunci dengan bobot
        for condition, keywords_dict in self.mental_health_keywords.items():
            core_count = sum(1 for keyword in keywords_dict['core'] if keyword.lower() in text.lower())
            secondary_count = sum(1 for keyword in keywords_dict['secondary'] if keyword.lower() in text.lower())
            weighted_score = (core_count * 2 + secondary_count) * keywords_dict['weight']
            features[f'{condition}_keyword_score'] = weighted_score
        
        return features
    
    def preprocess_text(self, text):
        """Preprocessing teks yang ditingkatkan"""
        # Simpan original untuk deteksi emoji
        original_text = text
        
        # Preprocessing dasar
        text = text.lower()
        
        # Hapus URL
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Hapus mention dan hashtag tapi tetap simpan teksnya
        text = re.sub(r'@(\w+)', r'\1', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Tetap pertahankan beberapa tanda baca untuk sentimen
        text = re.sub(r'[^a-zA-Z0-9\s\!\?\.\,]', '', text)
        
        return text, original_text
    
    def train(self, texts, labels, validate=True):
        """Pelatihan yang ditingkatkan dengan validasi"""
        print("🚀 Memulai proses pelatihan yang ditingkatkan...")
        
        # Ekstrak semua fitur
        all_features = []
        processed_texts = []
        
        for text in texts:
            processed_text, original = self.preprocess_text(text)
            processed_texts.append(processed_text)
            
            # Ekstrak fitur lanjutan
            advanced_features = self.extract_advanced_features(original)
            all_features.append(advanced_features)
        
        # Konversi ke DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Fitur TF-IDF
        tfidf_features = self.vectorizer.fit_transform(processed_texts)
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Gabungkan semua fitur
        combined_features = pd.concat([features_df, tfidf_df], axis=1)
        
        # Encode label
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, encoded_labels, test_size=0.2, random_state=42
        )
        
        # Latih model
        self.model.fit(X_train, y_train)
        
        # Evaluasi
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"✅ Akurasi training: {train_score:.2%}")
        print(f"✅ Akurasi testing: {test_score:.2%}")
        
        if validate:
            # Cross-validation
            cv_scores = cross_val_score(self.model, combined_features, encoded_labels, cv=5)
            print(f"✅ Skor cross-validation: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
            
            # Laporan klasifikasi detail
            y_pred = self.model.predict(X_test)
            print("\n📊 Laporan Klasifikasi:")
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        self.feature_columns = combined_features.columns
        return train_score, test_score
    
    def predict(self, text, return_probabilities=True):
        """Prediksi yang ditingkatkan dengan skor kepercayaan"""
        # Preprocessing
        processed_text, original = self.preprocess_text(text)
        
        # Ekstrak fitur
        advanced_features = self.extract_advanced_features(original)
        features_df = pd.DataFrame([advanced_features])
        
        # Fitur TF-IDF
        tfidf_features = self.vectorizer.transform([processed_text])
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Gabungkan fitur
        combined_features = pd.concat([features_df, tfidf_df], axis=1)
        
        # Selaraskan kolom
        combined_features = combined_features.reindex(columns=self.feature_columns, fill_value=0)
        
        # Prediksi
        prediction = self.model.predict(combined_features)[0]
        
        if return_probabilities:
            probabilities = self.model.predict_proba(combined_features)[0]
            
            # Buat dictionary probabilitas
            prob_dict = {}
            for i, label in enumerate(self.label_encoder.classes_):
                prob_dict[label] = float(probabilities[i])
            
            # Decode prediksi
            condition = self.label_encoder.inverse_transform([prediction])[0]
            
            # Hitung kepercayaan
            confidence = max(probabilities)
            second_best = sorted(probabilities, reverse=True)[1]
            confidence_margin = confidence - second_best
            
            return {
                'condition': condition,
                'confidence': float(confidence),
                'confidence_margin': float(confidence_margin),
                'probabilities': prob_dict,
                'sentiment': self.sentiment_analyzer.polarity_scores(text),
                'risk_level': self._calculate_risk_level(condition, confidence)
            }
        
        return self.label_encoder.inverse_transform([prediction])[0]
    
    def _calculate_risk_level(self, condition, confidence):
        """Hitung level risiko berdasarkan kondisi dan kepercayaan"""
        risk_scores = {
            'depresi': 3,
            'kecemasan': 2,
            'stress': 1,
            'normal': 0
        }
        
        base_risk = risk_scores.get(condition, 0)
        adjusted_risk = base_risk * confidence
        
        if adjusted_risk >= 2.4:
            return "High"
        elif adjusted_risk >= 1.2:
            return "Medium"
        else:
            return "Low"
    
    def analyze_chat_history(self, messages, timestamps=None):
        """Analisis riwayat chat dengan pola temporal"""
        results = []
        
        for i, message in enumerate(messages):
            result = self.predict(message)
            result['message_index'] = i
            result['message_preview'] = message[:100] + '...' if len(message) > 100 else message
            
            if timestamps:
                result['timestamp'] = timestamps[i]
            
            results.append(result)
        
        # Analisis ringkasan
        conditions = [r['condition'] for r in results]
        condition_counts = pd.Series(conditions).value_counts().to_dict()
        
        # Analisis temporal jika timestamp disediakan
        temporal_analysis = None
        if timestamps:
            temporal_analysis = self._analyze_temporal_patterns(results, timestamps)
        
        # Penilaian risiko
        high_risk_count = sum(1 for r in results if r['risk_level'] == "High")
        
        return {
            'individual_results': results,
            'summary': {
                'total_messages': len(messages),
                'condition_distribution': condition_counts,
                'dominant_condition': max(condition_counts, key=condition_counts.get),
                'average_confidence': np.mean([r['confidence'] for r in results]),
                'high_risk_messages': high_risk_count,
                'temporal_analysis': temporal_analysis
            }
        }
    
    def _analyze_temporal_patterns(self, results, timestamps):
        """Analisis pola temporal dalam kondisi kesehatan mental"""
        # Grup berdasarkan jam dalam hari, hari dalam minggu, dll.
        df = pd.DataFrame(results)
        df['timestamp'] = pd.to_datetime(timestamps)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        try:
            hourly_pattern = df.groupby('hour')['condition'].value_counts().unstack(fill_value=0)
            daily_pattern = df.groupby('day_of_week')['condition'].value_counts().unstack(fill_value=0)
            
            # Periksa apakah kolom ada sebelum mengakses
            peak_depression_hour = None
            peak_anxiety_hour = None
            
            if 'depresi' in hourly_pattern.columns:
                peak_depression_hour = hourly_pattern['depresi'].idxmax()
            if 'kecemasan' in hourly_pattern.columns:
                peak_anxiety_hour = hourly_pattern['kecemasan'].idxmax()
            
            return {
                'hourly_pattern': hourly_pattern.to_dict() if not hourly_pattern.empty else {},
                'daily_pattern': daily_pattern.to_dict() if not daily_pattern.empty else {},
                'peak_depression_hour': peak_depression_hour,
                'peak_anxiety_hour': peak_anxiety_hour
            }
        except Exception as e:
            print(f"Warning: Error in temporal analysis: {e}")
            return None
    
    def get_feature_importance(self):
        """Dapatkan pentingnya fitur dari model"""
        if hasattr(self.model, 'feature_importances_'):
            try:
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return importance_df.head(20)
            except Exception as e:
                print(f"Warning: Error getting feature importance: {e}")
                return None
        return None
    
    def save_model(self, filepath):
        """Simpan model dengan semua komponen"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.vectorizer,
                    'label_encoder': self.label_encoder,
                    'feature_columns': self.feature_columns,
                    'keywords': self.mental_health_keywords
                }, f)
            print(f"✅ Model berhasil disimpan ke {filepath}")
        except Exception as e:
            print(f"❌ Error menyimpan model: {e}")
    
    def load_model(self, filepath):
        """Muat model dengan semua komponen"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.vectorizer = data['vectorizer']
                self.label_encoder = data['label_encoder']
                self.feature_columns = data['feature_columns']
                self.mental_health_keywords = data['keywords']
            print(f"✅ Model berhasil dimuat dari {filepath}")
        except FileNotFoundError:
            print(f"❌ File model tidak ditemukan: {filepath}")
            print("🔧 Silakan jalankan train_model.py terlebih dahulu untuk membuat model")
        except Exception as e:
            print(f"❌ Error memuat model: {e}")
    
    def batch_predict(self, texts):
        """Prediksi batch untuk multiple teks"""
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                print(f"Warning: Error predicting text '{text[:50]}...': {e}")
                # Return default result in case of error
                results.append({
                    'condition': 'normal',
                    'confidence': 0.0,
                    'risk_level': 'Low',
                    'sentiment': {'compound': 0.0}
                })
        return results
    
    def get_model_info(self):
        """Dapatkan informasi tentang model"""
        if hasattr(self, 'model') and hasattr(self, 'label_encoder'):
            return {
                'classes': list(self.label_encoder.classes_),
                'n_estimators': getattr(self.model, 'n_estimators', 'N/A'),
                'learning_rate': getattr(self.model, 'learning_rate', 'N/A'),
                'max_depth': getattr(self.model, 'max_depth', 'N/A'),
                'n_features': len(self.feature_columns) if hasattr(self, 'feature_columns') else 0
            }
        return None
    
    def explain_prediction(self, text):
        """Berikan penjelasan tentang prediksi"""
        result = self.predict(text)
        
        # Analisis kata kunci yang ditemukan
        found_keywords = {}
        for condition, keywords_dict in self.mental_health_keywords.items():
            found_core = [kw for kw in keywords_dict['core'] if kw.lower() in text.lower()]
            found_secondary = [kw for kw in keywords_dict['secondary'] if kw.lower() in text.lower()]
            if found_core or found_secondary:
                found_keywords[condition] = {
                    'core': found_core,
                    'secondary': found_secondary
                }
        
        # Analisis emoji
        found_emojis = {
            'positive': [emoji for emoji in self.emoji_patterns['positive'] if emoji in text],
            'negative': [emoji for emoji in self.emoji_patterns['negative'] if emoji in text]
        }
        
        explanation = {
            'prediction': result,
            'keywords_found': found_keywords,
            'emojis_found': found_emojis,
            'text_stats': {
                'length': len(text),
                'word_count': len(text.split()),
                'exclamation_marks': text.count('!'),
                'question_marks': text.count('?')
            }
        }
        
        return explanation