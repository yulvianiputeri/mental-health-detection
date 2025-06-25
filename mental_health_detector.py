import pandas as pd
import numpy as np
import re
import nltk
from datetime import datetime, timedelta
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
        
        self.mental_health_keywords = {
            'depresi': {
                'core': [
                    'sedih', 'depresi', 'putus asa', 'bunuh diri', 'hampa', 'gabut', 'worthless',
                    'ingin mati', 'tidak berharga', 'benci diri', 'bosan hidup', 'lelah hidup', 
                    'menyerah', 'sia-sia', 'gagal', 'beban', 'menyesal', 'muak', 'nggak ada gunanya',
                    'ga ada gunanya', 'nyerah', 'pengen mati', 'pengen ngilang', 'pengen lenyap',
                    'cape hidup', 'capek hidup', 'benci hidup'
                ],
                'secondary': [
                    'kesepian', 'tidak berharga', 'lelah hidup', 'menyesal', 'malas', 'males', 
                    'apatis', 'nangis', 'menangis', 'sendiri', 'sendirian', 'gelap'
                ],
                'weight': 2.0  
            },
            'kecemasan': {
                'core': [
                    'cemas', 'khawatir', 'panik', 'takut', 'deg-degan', 'overthinking',
                    'anxiety', 'anxious', 'gelisah', 'resah', 'was-was', 'gak tenang', 
                    'ga tenang', 'tidak tenang', 'deg degan', 'gugup', 'paranoid', 'fobia',
                    'trauma', 'ketar-ketir', 'ketar ketir', 'tremor', 'gemetar'
                ],
                'secondary': [
                    'jantung berdebar', 'keringat dingin', 'sesak napas', 'drama', 'baper',
                    'kepikiran', 'mikirin', 'mikir terus', 'pikiran', 'bingung', 'ragu'
                ],
                'weight': 1.8
            },
            'stress': {
                'core': [
                    'burnout', 'stress', 'tertekan', 'deadline', 'cenat-cenut', 'tumpuk',
                    'overwhelmed', 'tekanan', 'beban', 'pusing', 'frustasi', 'frustrasi',
                    'banyak kerjaan', 'banyak tugas', 'numpuk', 'menumpuk', 'kewalahan',
                    'capek mental', 'cape mental', 'lelah mental', 'exhausted', 'overworked',
                    'jenuh', 'muak', 'bosan', 'bete', 'bad mood'
                ],
                'secondary': [
                    'kewalahan', 'tekanan', 'tagihan', 'berantakan', 'sibuk', 'deadline',
                    'terburu-buru', 'batas waktu', 'tenggat', 'waktu mepet', 'dikejar'
                ],
                'weight': 1.8  
            },
            'normal': {
                'core': [
                    'senang', 'bahagia', 'bersyukur', 'chill', 'asyik', 'syukur',
                    'happy', 'tenang', 'damai', 'semangat', 'positif', 'baik',
                    'sehat', 'stabil', 'produktif', 'sukses', 'berhasil', 'puas'
                ],
                'secondary': [
                    'baik', 'oke', 'lega', 'fun', 'santai', 'relax', 'enak', 'nyaman',
                    'menyenangkan', 'enjoy', 'fine', 'ok', 'okay'
                ],
                'weight': 0.8  
            }
        }
        
        self.emoji_patterns = {
            'positive': ['😊', '😃', '😄', '😁', '😆', '😍', '🥰', '😘', '😗', '☺️', '😚', '😙', '🙂', '😀', '❤️', '👍', '🎉', '✨', '😌', '🤗', '😇', '🙏'],
            'negative': ['😢', '😭', '😔', '😞', '😟', '😕', '☹️', '😣', '😖', '😫', '😩', '🥺', '😓', '😥', '😰', '😨', '😱', '😪', '😿', '💔', '👎', '😠', '😡', '🤬'],
            'neutral': ['😐', '🤔', '🙄', '😶', '😑', '😒', '🤨', '😏', '😬', '😯', '😦', '😧']
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
        
        # Sentiment analysis - catatan: VADER tidak optimal untuk bahasa Indonesia
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        print(f"Sentimen untuk '{text}': {sentiment}")  # Menampilkan hasil sentimen untuk teks
        features.update({
            'sentiment_positive': sentiment['pos'],
            'sentiment_negative': sentiment['neg'],
            'sentiment_neutral': sentiment['neu'],
            'sentiment_compound': sentiment['compound'],
        })
        
        # Emoji analysis dengan bobot
        features.update({
            'positive_emoji': sum(1 for e in self.emoji_patterns['positive'] if e in text) * 1.5,
            'negative_emoji': sum(1 for e in self.emoji_patterns['negative'] if e in text) * 2.0,
            'neutral_emoji': sum(1 for e in self.emoji_patterns['neutral'] if e in text),
        })
        
        # Keyword scoring dengan pengecekan pattern yang ditingkatkan
        for condition, keywords in self.mental_health_keywords.items():
            core_matches = []
            for word in keywords['core']:
                if word in text_lower or re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                    core_matches.append(word)
            
            secondary_matches = []
            for word in keywords['secondary']:
                if word in text_lower or re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                    secondary_matches.append(word)
            
            core_count = len(core_matches)
            secondary_count = len(secondary_matches)
            
            # Hitung skor berdasarkan keyword matches
            features[f'{condition}_score'] = (core_count * 2 + secondary_count) * keywords['weight']
            
            # Tambahkan fitur khusus untuk kata kunci yang ditemukan
            features[f'{condition}_core_keywords'] = core_count
            features[f'{condition}_secondary_keywords'] = secondary_count
        
        # Special case handling
        # Jika ada kata kunci kritis, tingkatkan skor kondisinya
        critical_keywords = {
            'depresi': ['bunuh diri', 'mati', 'ingin mati', 'pengen mati', 'ngilang', 'lenyap'],
            'kecemasan': ['panik', 'serangan panik', 'sesak napas', 'jantung berdebar'],
            'stress': ['burnout', 'kelelahan mental', 'overwork', 'overwhelmed']
        }
        
        for condition, keywords in critical_keywords.items():
            if any(kw in text_lower for kw in keywords):
                features[f'{condition}_score'] *= 1.5
        
        # Fitur linguistik tambahan untuk bahasa Indonesia
        intensifiers = ['banget', 'sangat', 'amat', 'sekali', 'terlalu', 'sungguh', 'teramat', 'begitu']
        features['intensifier_count'] = sum(1 for word in intensifiers if word in text_lower)
        
        # Fitur repetisi (misalnya "sedih sedih sedih")
        words = text_lower.split()
        repetition = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
        features['word_repetition'] = repetition
        
        return features

    def preprocess_text(self, text):
        original = text
        text = text.lower()
        
        # Simpan emoji dan special characters yang penting
        emojis = re.findall(r'[^\w\s,\.!?]', text)
        emoji_pattern = ''.join(emojis)
        
        # Hapus URL dan mentions
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        
        # Jangan hapus hashtag, karena bisa jadi konten penting
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Simpan tanda baca penting, hapus yang lain
        text = re.sub(r'[^a-zA-Z0-9\s\!\?\.\,]', ' ', text)
        
        # Tambahkan emoji kembali (jika ada)
        text = text + " " + emoji_pattern
        
        # Normalisasi slang bahasa Indonesia
        slang_map = {
            'gue': 'saya', 'gw': 'saya', 'w': 'saya', 'aku': 'saya',
            'lo': 'kamu', 'lu': 'kamu', 'loe': 'kamu', 'u': 'kamu',
            'ga': 'tidak', 'gak': 'tidak', 'g': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
            'gamau': 'tidak mau', 'gk': 'tidak', 'gapernah': 'tidak pernah',
            'tp': 'tapi', 'tpi': 'tapi', 'cb': 'coba', 'bs': 'bisa',
            'udah': 'sudah', 'udh': 'sudah', 'dah': 'sudah',
            'bgt': 'banget', 'bngt': 'banget',
            'pengen': 'ingin', 'pgn': 'ingin',
            'cape': 'capek', 'capek': 'lelah',
            'gabut': 'tidak ada kerjaan'
        }
        
        # Terapkan normalisasi slang
        words = text.split()
        normalized_words = [slang_map.get(word, word) for word in words]
        text = ' '.join(normalized_words)
        
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
        
        # Sebelum encoding, cek distribusi kelas
        label_counts = pd.Series(labels).value_counts()
        print("Distribusi kelas sebelum training:")
        print(label_counts)
        
        # Periksa apakah ada ketidakseimbangan kelas yang signifikan
        if label_counts.max() / label_counts.min() > 3:
            print("\nPeringatan: Ketidakseimbangan kelas terdeteksi!")
            print("Ini bisa menyebabkan bias ke kelas mayoritas (biasanya 'normal').")
            print("Sebaiknya tambahkan lebih banyak sampel untuk kelas minoritas.")
        
        encoded_labels = self.label_encoder.fit_transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
        )
        
        # Setelah cross-validation, latih model pada seluruh dataset
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
            
            # Confusion matrix untuk analisis lebih mendalam
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(
                cm, 
                index=self.label_encoder.classes_,
                columns=self.label_encoder.classes_
            )
            print(cm_df)
        
        self.feature_columns = combined_features.columns
        return train_score, test_score

    def predict(self, text):
        # Analisis sentimen
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        print(f"Sentimen untuk '{text}': {sentiment}")  # Debugging: Menampilkan hasil sentimen
        
        # Special case handling
        critical_keywords = {
            'depresi': ['bunuh diri', 'mati', 'ingin mati', 'pengen mati'],
            'kecemasan': ['panik', 'serangan panik', 'sesak napas'],
            'stress': ['burnout', 'kelelahan mental', 'overwork']
        }
        
        # Cek kata kunci kritis
        text_lower = text.lower()
        for condition, keywords in critical_keywords.items():
            if any(kw in text_lower for kw in keywords):
                confidence = 0.85
                probas = {'depresi': 0.05, 'kecemasan': 0.05, 'stress': 0.05, 'normal': 0.05}
                probas[condition] = confidence
                
                return {
                    'condition': condition,
                    'confidence': confidence,
                    'risk_level': 'High',
                    'probabilities': probas,
                    'sentiment': sentiment,  # Mengembalikan nilai sentimen
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
        
        # Debugging: hitung nilai fitur kondisi
        debug_scores = {
            'depresi_score': features.get('depresi_score', 0),
            'kecemasan_score': features.get('kecemasan_score', 0),
            'stress_score': features.get('stress_score', 0),
            'normal_score': features.get('normal_score', 0)
        }
        
        # Debugging: Print keywords yang terdeteksi
        debug_keywords = {}
        for condition in ['depresi', 'kecemasan', 'stress', 'normal']:
            core_kw = features.get(f'{condition}_core_keywords', 0)
            sec_kw = features.get(f'{condition}_secondary_keywords', 0)
            debug_keywords[condition] = f"{core_kw} core, {sec_kw} secondary"
        
        # Jika skor keyword sangat tinggi, prioritaskan kondisi tersebut
        max_score_condition = max(debug_scores, key=debug_scores.get)
        max_score = debug_scores[max_score_condition]
        
        # Override model jika ada sinyal kuat dari kata kunci
        if max_score > 5.0 and max_score_condition != 'normal':
            # Ada sinyal kuat dari kata kunci
            condition = max_score_condition
            confidence = min(0.85, max_score / 10.0)  # Scale confidence
            
            # Buat distribusi probabilitas yang realistis
            probas = {'depresi': 0.05, 'kecemasan': 0.05, 'stress': 0.05, 'normal': 0.05}
            remaining = 1.0 - confidence
            for cond in probas:
                if cond != condition:
                    probas[cond] = remaining / 3.0
            probas[condition] = confidence
            
            return {
                'condition': condition,
                'confidence': confidence,
                'risk_level': self._calculate_risk_level(condition, confidence),
                'probabilities': probas,
                'sentiment': self.sentiment_analyzer.polarity_scores(text),
                'confidence_margin': confidence - (remaining / 3.0),
                'debug_info': {
                    'keyword_scores': debug_scores,
                    'detected_keywords': debug_keywords
                }
            }
            
        # Combinasikan semua fitur, dan pastikan semua kolom yang diperlukan ada
        combined = pd.concat([features_df, tfidf_df], axis=1)
        
        # Periksa apakah model sudah dimuat
        if not hasattr(self, 'feature_columns'):
            raise ValueError("Model belum dilatih atau dimuat!")
        
        # Pastikan format fitur sesuai dengan yang diharapkan model
        combined = combined.reindex(columns=self.feature_columns, fill_value=0)
        
        # Prediksi dengan model
        try:
            prediction = self.model.predict(combined)[0]
            probabilities = self.model.predict_proba(combined)[0]
            
            # Konversi indeks ke label
            condition = self.label_encoder.inverse_transform([prediction])[0]
            proba_dict = {k: float(v) for k, v in zip(self.label_encoder.classes_, probabilities)}
            
            return {
                'condition': condition,
                'confidence': float(np.max(probabilities)),
                'confidence_margin': float(np.max(probabilities) - np.sort(probabilities)[-2]),
                'probabilities': proba_dict,
                'sentiment': self.sentiment_analyzer.polarity_scores(text),
                'risk_level': self._calculate_risk_level(
                    condition,
                    np.max(probabilities)
                ),
                'debug_info': {
                    'keyword_scores': debug_scores,
                    'detected_keywords': debug_keywords
                }
            }
        except Exception as e:
            # Fallback ke rule-based jika model gagal
            print(f"Error dalam prediksi model: {str(e)}")
            
            # Rule-based fallback
            condition = max(debug_scores, key=debug_scores.get)
            if debug_scores[condition] < 1.0:
                condition = 'normal'  # Default ke normal jika tidak ada sinyal kuat
                
            confidence = min(0.7, debug_scores[condition] / 10.0 + 0.4)
            
            return {
                'condition': condition,
                'confidence': confidence,
                'confidence_margin': 0.3,
                'probabilities': {
                    'depresi': 0.1 if condition != 'depresi' else confidence,
                    'kecemasan': 0.1 if condition != 'kecemasan' else confidence,
                    'stress': 0.1 if condition != 'stress' else confidence,
                    'normal': 0.7 if condition != 'normal' else confidence
                },
                'sentiment': self.sentiment_analyzer.polarity_scores(text),
                'risk_level': self._calculate_risk_level(condition, confidence),
                'debug_info': {
                    'keyword_scores': debug_scores,
                    'detected_keywords': debug_keywords,
                    'error': str(e)
                }
            }

    def _calculate_risk_level(self, condition, confidence):
        risk_scores = {'depresi': 2.0, 'kecemasan': 1.5, 'stress': 1.5, 'normal': 0}
        adjusted_risk = risk_scores.get(condition, 0) * confidence
        
        if condition == 'depresi' and confidence >= 0.6:
            return "High"
        elif condition == 'stress' and confidence >= 0.75:
            return "High"
        elif adjusted_risk >= 1.5:
            return "High"
        elif adjusted_risk >= 0.8:
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
                self.mental_health_keywords = data.get('keywords', self.mental_health_keywords)
                print("✅ Model berhasil dimuat!")
        except Exception as e:
            print(f"❌ Error saat memuat model: {str(e)}")
            raise

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
            
            # Tambahkan informasi debug jika ada
            if 'debug_info' in prediction:
                prediction['debug_info']['message'] = message
            
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