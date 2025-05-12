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

# Download NLTK data
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
            stop_words='english'
        )
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Enhanced keyword dictionary
        self.mental_health_keywords = {
            'depression': {
                'core': ['sedih', 'depresi', 'putus asa', 'bunuh diri', 'mati'],
                'secondary': ['kesepian', 'hampa', 'gagal', 'tidak berguna', 'lelah hidup'],
                'weight': 2.0
            },
            'anxiety': {
                'core': ['cemas', 'khawatir', 'panik', 'takut', 'gelisah'],
                'secondary': ['jantung berdebar', 'keringat dingin', 'overthinking'],
                'weight': 1.8
            },
            'stress': {
                'core': ['stress', 'tertekan', 'beban', 'overwhelmed'],
                'secondary': ['deadline', 'burned out', 'pusing', 'lelah mental'],
                'weight': 1.5
            },
            'normal': {
                'core': ['senang', 'bahagia', 'bersyukur', 'optimis'],
                'secondary': ['produktif', 'semangat', 'tenang', 'damai'],
                'weight': 1.0
            }
        }
        
        # Emoji patterns
        self.emoji_patterns = {
            'positive': ['😊', '😃', '🙂', '😄', '❤️', '👍', '🎉', '✨'],
            'negative': ['😢', '😭', '😔', '😟', '😰', '😱', '💔', '😞'],
            'neutral': ['😐', '🤔', '😑', '👌', '🤷']
        }
        
    def extract_advanced_features(self, text):
        """Extract advanced features including sentiment, emoji, and length"""
        features = {}
        
        # Basic text features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # Sentiment features
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        features['sentiment_positive'] = sentiment_scores['pos']
        features['sentiment_negative'] = sentiment_scores['neg']
        features['sentiment_neutral'] = sentiment_scores['neu']
        features['sentiment_compound'] = sentiment_scores['compound']
        
        # Emoji features
        positive_emojis = sum(1 for emoji in self.emoji_patterns['positive'] if emoji in text)
        negative_emojis = sum(1 for emoji in self.emoji_patterns['negative'] if emoji in text)
        features['positive_emoji_count'] = positive_emojis
        features['negative_emoji_count'] = negative_emojis
        
        # Keyword features with weights
        for condition, keywords_dict in self.mental_health_keywords.items():
            core_count = sum(1 for keyword in keywords_dict['core'] if keyword.lower() in text.lower())
            secondary_count = sum(1 for keyword in keywords_dict['secondary'] if keyword.lower() in text.lower())
            weighted_score = (core_count * 2 + secondary_count) * keywords_dict['weight']
            features[f'{condition}_keyword_score'] = weighted_score
        
        return features
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        # Keep original for emoji detection
        original_text = text
        
        # Basic preprocessing
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions and hashtags but keep the text
        text = re.sub(r'@(\w+)', r'\1', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Keep some punctuation for sentiment
        text = re.sub(r'[^a-zA-Z0-9\s\!\?\.\,]', '', text)
        
        return text, original_text
    
    def train(self, texts, labels, validate=True):
        """Enhanced training with validation"""
        print("🚀 Starting enhanced training process...")
        
        # Extract all features
        all_features = []
        processed_texts = []
        
        for text in texts:
            processed_text, original = self.preprocess_text(text)
            processed_texts.append(processed_text)
            
            # Extract advanced features
            advanced_features = self.extract_advanced_features(original)
            all_features.append(advanced_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # TF-IDF features
        tfidf_features = self.vectorizer.fit_transform(processed_texts)
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Combine all features
        combined_features = pd.concat([features_df, tfidf_df], axis=1)
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, encoded_labels, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"✅ Training accuracy: {train_score:.2%}")
        print(f"✅ Testing accuracy: {test_score:.2%}")
        
        if validate:
            # Cross-validation
            cv_scores = cross_val_score(self.model, combined_features, encoded_labels, cv=5)
            print(f"✅ Cross-validation score: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
            
            # Detailed classification report
            y_pred = self.model.predict(X_test)
            print("\n📊 Classification Report:")
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        self.feature_columns = combined_features.columns
        return train_score, test_score
    
    def predict(self, text, return_probabilities=True):
        """Enhanced prediction with confidence scores"""
        # Preprocess
        processed_text, original = self.preprocess_text(text)
        
        # Extract features
        advanced_features = self.extract_advanced_features(original)
        features_df = pd.DataFrame([advanced_features])
        
        # TF-IDF features
        tfidf_features = self.vectorizer.transform([processed_text])
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        # Combine features
        combined_features = pd.concat([features_df, tfidf_df], axis=1)
        
        # Align columns
        combined_features = combined_features.reindex(columns=self.feature_columns, fill_value=0)
        
        # Predict
        prediction = self.model.predict(combined_features)[0]
        
        if return_probabilities:
            probabilities = self.model.predict_proba(combined_features)[0]
            
            # Create probability dictionary
            prob_dict = {}
            for i, label in enumerate(self.label_encoder.classes_):
                prob_dict[label] = float(probabilities[i])
            
            # Decode prediction
            condition = self.label_encoder.inverse_transform([prediction])[0]
            
            # Calculate confidence
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
        """Calculate risk level based on condition and confidence"""
        risk_scores = {
            'depression': 3,
            'anxiety': 2,
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
        """Analyze chat history with temporal patterns"""
        results = []
        
        for i, message in enumerate(messages):
            result = self.predict(message)
            result['message_index'] = i
            result['message_preview'] = message[:100] + '...' if len(message) > 100 else message
            
            if timestamps:
                result['timestamp'] = timestamps[i]
            
            results.append(result)
        
        # Summary analysis
        conditions = [r['condition'] for r in results]
        condition_counts = pd.Series(conditions).value_counts().to_dict()
        
        # Temporal analysis if timestamps provided
        temporal_analysis = None
        if timestamps:
            temporal_analysis = self._analyze_temporal_patterns(results, timestamps)
        
        # Risk assessment
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
        """Analyze temporal patterns in mental health conditions"""
        # Group by hour of day, day of week, etc.
        df = pd.DataFrame(results)
        df['timestamp'] = pd.to_datetime(timestamps)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        hourly_pattern = df.groupby('hour')['condition'].value_counts().unstack(fill_value=0)
        daily_pattern = df.groupby('day_of_week')['condition'].value_counts().unstack(fill_value=0)
        
        return {
            'hourly_pattern': hourly_pattern.to_dict(),
            'daily_pattern': daily_pattern.to_dict(),
            'peak_depression_hour': hourly_pattern['depression'].idxmax() if 'depression' in hourly_pattern else None,
            'peak_anxiety_hour': hourly_pattern['anxiety'].idxmax() if 'anxiety' in hourly_pattern else None
        }
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(20)
        return None
    
    def save_model(self, filepath):
        """Save model with all components"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'keywords': self.mental_health_keywords
            }, f)
    
    def load_model(self, filepath):
        """Load model with all components"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.label_encoder = data['label_encoder']
            self.feature_columns = data['feature_columns']
            self.mental_health_keywords = data['keywords']