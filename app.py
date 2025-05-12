# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from mental_health_detector import AdvancedMentalHealthDetector
import random

# Page config
st.set_page_config(
    page_title="Advanced Mental Health Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high { color: #ff4444; font-weight: bold; }
    .risk-medium { color: #ffaa00; font-weight: bold; }
    .risk-low { color: #00C851; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_detector():
    detector = AdvancedMentalHealthDetector()
    detector.load_model('mental_health_model_advanced.pkl')
    return detector

# Enhanced recommendations
def get_enhanced_recommendations(condition, risk_level):
    recommendations = {
        'depression': {
            'high': [
                '🚨 **URGENT**: Please contact a mental health professional immediately',
                '📞 Crisis Hotline: 988 (Suicide & Crisis Lifeline)',
                '🏥 Visit nearest emergency room if having suicidal thoughts',
                '👥 Don\'t stay alone - reach out to trusted friends/family',
                '💊 Consider medication consultation with psychiatrist'
            ],
            'medium': [
                '🩺 Schedule appointment with therapist/counselor',
                '🏃‍♀️ Start regular exercise routine (even 10 min walks)',
                '😴 Prioritize sleep hygiene (7-9 hours)',
                '📱 Use mental health apps (Headspace, Calm)',
                '📓 Start journaling to track mood patterns'
            ],
            'low': [
                '🌱 Practice self-care activities',
                '☀️ Get sunlight exposure daily',
                '🥗 Maintain healthy diet',
                '🧘‍♀️ Try meditation or yoga',
                '🎨 Engage in creative activities'
            ]
        },
        'anxiety': {
            'high': [
                '🚨 Seek immediate professional help',
                '🫁 Practice emergency breathing techniques',
                '💊 Discuss anti-anxiety medication with doctor',
                '🚫 Avoid caffeine and stimulants',
                '📱 Download panic attack apps'
            ],
            'medium': [
                '🧘‍♂️ Learn relaxation techniques',
                '📝 Keep anxiety diary',
                '🏃‍♂️ Regular aerobic exercise',
                '🎯 Challenge negative thoughts',
                '👥 Join anxiety support groups'
            ],
            'low': [
                '☕ Limit caffeine intake',
                '📅 Maintain routine',
                '🌿 Try herbal teas',
                '🎵 Listen to calming music',
                '📚 Read self-help books'
            ]
        },
        'stress': {
            'high': [
                '🚨 Take immediate stress leave if possible',
                '🩺 Consult doctor about stress symptoms',
                '🧘‍♀️ Daily stress-reduction practices mandatory',
                '❌ Learn to say NO to additional responsibilities',
                '💤 Prioritize rest and recovery'
            ],
            'medium': [
                '⏰ Implement time management strategies',
                '🎯 Set realistic goals and boundaries',
                '🏖️ Plan regular breaks and vacations',
                '💪 Delegate tasks when possible',
                '🧘‍♂️ Practice mindfulness meditation'
            ],
            'low': [
                '📱 Use productivity apps',
                '🎨 Engage in hobbies',
                '🌳 Spend time in nature',
                '👥 Maintain social connections',
                '🏃‍♀️ Regular physical activity'
            ]
        },
        'normal': {
            'low': [
                '✅ Continue current healthy habits',
                '📊 Regular mental health check-ins',
                '💪 Build resilience skills',
                '🎯 Set personal growth goals',
                '🤝 Help others in need'
            ]
        }
    }
    
    return recommendations.get(condition, {}).get(risk_level.lower(), [])

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Advanced Mental Health Detection System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # User profile (mock)
        st.subheader("👤 User Profile")
        user_name = st.text_input("Name", "Anonymous")
        user_age = st.number_input("Age", 18, 100, 25)
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.show_analytics = True
        if st.button("📝 Export Report", use_container_width=True):
            st.session_state.export_report = True
        if st.button("🆘 Crisis Resources", use_container_width=True):
            st.session_state.show_crisis = True
            
        st.markdown("---")
        
        # Info
        st.info(
            "This AI system uses advanced NLP and machine learning to detect mental health patterns. "
            "It analyzes text sentiment, keywords, and linguistic patterns to provide insights."
        )
        
        st.warning(
            "⚠️ This is NOT a replacement for professional mental health diagnosis. "
            "Please consult healthcare providers for medical advice."
        )
    
    # Main content area
    tabs = st.tabs(["🔍 Analysis", "📊 Dashboard", "📈 History", "💡 Resources"])
    
    # Analysis Tab
    with tabs[0]:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.header("💬 Text Analysis")
            
            # Input method selection
            input_method = st.radio(
                "Choose input method:",
                ["Single Message", "Chat History", "Voice Note (Beta)"],
                horizontal=True
            )
            
            if input_method == "Single Message":
                user_input = st.text_area(
                    "Share your thoughts and feelings:",
                    height=150,
                    placeholder="I'm feeling... (You can use emojis too! 😊😔)"
                )
                
                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
                with col_btn1:
                    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
                with col_btn2:
                    clear_btn = st.button("🗑️ Clear", use_container_width=True)
                
                if analyze_btn and user_input:
                    with st.spinner("🧠 AI analyzing your text..."):
                        # Load model
                        detector = load_detector()
                        
                        # Get prediction
                        result = detector.predict(user_input)
                        
                        # Add to history
                        st.session_state.history.append({
                            'timestamp': datetime.now(),
                            'text': user_input,
                            'result': result
                        })
                        
                        # Display results
                        st.success("✅ Analysis complete!")
                        
                        # Condition card
                        condition_colors = {
                            'depression': '#ff4444',
                            'anxiety': '#ff8800',
                            'stress': '#ffbb33',
                            'normal': '#00C851'
                        }
                        
                        st.markdown(f"""
                        <div style="background-color: {condition_colors.get(result['condition'], '#gray')}; 
                                    color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                            <h2 style="margin: 0;">Detected Condition: {result['condition'].upper()}</h2>
                            <p style="margin: 0.5rem 0;">Confidence: {result['confidence']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk level
                        risk_class = f"risk-{result['risk_level'].lower()}"
                        st.markdown(f"<p class='{risk_class}'>Risk Level: {result['risk_level']}</p>", 
                                   unsafe_allow_html=True)
                        
                        # Detailed metrics
                        st.subheader("📊 Detailed Analysis")
                        met_col1, met_col2, met_col3 = st.columns(3)
                        
                        with met_col1:
                            st.metric("Sentiment Score", f"{result['sentiment']['compound']:.3f}")
                        with met_col2:
                            st.metric("Confidence Margin", f"{result['confidence_margin']:.2%}")
                        with met_col3:
                            st.metric("Word Count", len(user_input.split()))
                        
                        # Probability distribution
                        st.subheader("🎯 Condition Probabilities")
                        prob_df = pd.DataFrame(list(result['probabilities'].items()), 
                                              columns=['Condition', 'Probability'])
                        fig = px.bar(prob_df, x='Condition', y='Probability', 
                                    color='Condition', 
                                    color_discrete_map=condition_colors,
                                    title="AI Confidence Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                
                if clear_btn:
                    st.session_state.clear()
                    st.rerun()
            
            elif input_method == "Chat History":
                st.write("📱 Paste your chat history (one message per line):")
                chat_input = st.text_area(
                    "Chat messages:",
                    height=200,
                    placeholder="Message 1\nMessage 2\nMessage 3\n..."
                )
                
                if st.button("📊 Analyze History", type="primary"):
                    if chat_input:
                        messages = [msg.strip() for msg in chat_input.split('\n') if msg.strip()]
                        
                        with st.spinner("🔄 Analyzing chat history..."):
                            detector = load_detector()
                            
                            # Create synthetic timestamps
                            timestamps = [datetime.now() - timedelta(hours=i) for i in range(len(messages))]
                            
                            results = detector.analyze_chat_history(messages, timestamps)
                            
                            st.success("✅ Chat history analysis complete!")
                            
                            # Summary metrics
                            summary = results['summary']
                            st.subheader("📈 Summary Statistics")
                            
                            sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                            with sum_col1:
                                st.metric("Total Messages", summary['total_messages'])
                            with sum_col2:
                                st.metric("Dominant Condition", summary['dominant_condition'].upper())
                            with sum_col3:
                                st.metric("High Risk Messages", summary['high_risk_messages'])
                            with sum_col4:
                                st.metric("Avg Confidence", f"{summary['average_confidence']:.1%}")
                            
                            # Condition distribution pie chart
                            dist_df = pd.DataFrame(list(summary['condition_distribution'].items()),
                                                  columns=['Condition', 'Count'])
                            fig_pie = px.pie(dist_df, values='Count', names='Condition',
                                           title="Condition Distribution in Chat History")
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Individual message analysis
                            st.subheader("🔍 Individual Message Analysis")
                            for idx, result in enumerate(results['individual_results']):
                                with st.expander(f"Message {idx+1}: {result['message_preview']}"):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.write(f"**Condition:** {result['condition']}")
                                    with col2:
                                        st.write(f"**Confidence:** {result['confidence']:.1%}")
                                    with col3:
                                        st.write(f"**Risk:** {result['risk_level']}")
            
            else:  # Voice Note
                st.info("🎤 Voice analysis coming soon! This feature will allow you to speak your thoughts.")
                st.write("For now, please use text input.")
        
        with col2:
            st.header("💡 Recommendations")
            
            if 'history' in st.session_state and st.session_state.history:
                latest = st.session_state.history[-1]
                condition = latest['result']['condition']
                risk_level = latest['result']['risk_level']
                
                recommendations = get_enhanced_recommendations(condition, risk_level)
                
                # Risk-based styling
                if risk_level == "High":
                    st.error("⚠️ HIGH RISK DETECTED - Immediate Action Recommended")
                elif risk_level == "Medium":
                    st.warning("⚠️ Moderate Risk - Professional Support Advised")
                else:
                    st.info("ℹ️ Low Risk - Preventive Measures Recommended")
                
                # Display recommendations
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
                
                # Emergency resources
                if risk_level in ["High", "Medium"]:
                    st.markdown("---")
                    st.subheader("🆘 Emergency Resources")
                    st.error("""
                    **Crisis Hotlines:**
                    - 988 Suicide & Crisis Lifeline
                    - Crisis Text Line: Text HOME to 741741
                    - Emergency: 911
                    """)
                    
                    with st.expander("Find Local Resources"):
                        st.write("🏥 [Find Mental Health Services Near You](https://findtreatment.samhsa.gov/)")
                        st.write("👥 [Support Groups](https://www.nami.org/support-education)")
                        st.write("📱 [Mental Health Apps](https://www.psychiatry.org/patients-families/mental-health-apps)")
            else:
                st.info("👈 Enter text and click analyze to get personalized recommendations")
                
                # General tips
                st.subheader("🌟 General Mental Health Tips")
                tips = [
                    "🧘‍♀️ Practice mindfulness daily",
                    "🏃‍♂️ Regular exercise boosts mood",
                    "😴 Prioritize quality sleep",
                    "🥗 Maintain balanced nutrition",
                    "👥 Stay socially connected",
                    "📝 Keep a mood journal",
                    "🎯 Set realistic goals",
                    "🌳 Spend time in nature"
                ]
                for tip in tips:
                    st.write(tip)
    
    # Dashboard Tab
    with tabs[1]:
        st.header("📊 Mental Health Dashboard")
        
        if st.session_state.history:
            # Convert history to DataFrame
            history_df = pd.DataFrame([
                {
                    'timestamp': h['timestamp'],
                    'condition': h['result']['condition'],
                    'confidence': h['result']['confidence'],
                    'risk_level': h['result']['risk_level'],
                    'sentiment': h['result']['sentiment']['compound']
                }
                for h in st.session_state.history
            ])
            
            # Time series plot
            fig_time = go.Figure()
            
            conditions = history_df['condition'].unique()
            condition_colors = {
                'depression': '#ff4444',
                'anxiety': '#ff8800',
                'stress': '#ffbb33',
                'normal': '#00C851'
            }
            
            for condition in conditions:
                condition_data = history_df[history_df['condition'] == condition]
                fig_time.add_trace(go.Scatter(
                    x=condition_data['timestamp'],
                    y=condition_data['confidence'],
                    mode='markers+lines',
                    name=condition.capitalize(),
                    marker=dict(color=condition_colors.get(condition, 'gray'), size=10),
                    line=dict(color=condition_colors.get(condition, 'gray'))
                ))
            
            fig_time.update_layout(
                title="Mental Health Trends Over Time",
                xaxis_title="Time",
                yaxis_title="Confidence Score",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                # Condition frequency
                condition_counts = history_df['condition'].value_counts()
                fig_freq = px.bar(x=condition_counts.index, y=condition_counts.values,
                                 labels={'x': 'Condition', 'y': 'Frequency'},
                                 title="Condition Frequency",
                                 color=condition_counts.index,
                                 color_discrete_map=condition_colors)
                st.plotly_chart(fig_freq, use_container_width=True)
            
            with col2:
                # Risk level distribution
                risk_counts = history_df['risk_level'].value_counts()
                fig_risk = px.pie(values=risk_counts.values, names=risk_counts.index,
                                 title="Risk Level Distribution",
                                 color_discrete_map={'High': '#ff4444', 
                                                    'Medium': '#ff8800',
                                                    'Low': '#00C851'})
                st.plotly_chart(fig_risk, use_container_width=True)
            
            # Sentiment trends
            st.subheader("😊 Sentiment Analysis")
            fig_sentiment = go.Figure()
            fig_sentiment.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['sentiment'],
                mode='lines+markers',
                name='Sentiment Score',
                line=dict(color='purple', width=3),
                marker=dict(size=8)
            ))
            
            fig_sentiment.update_layout(
                title="Sentiment Trends",
                xaxis_title="Time",
                yaxis_title="Sentiment Score (-1 to 1)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # Recent activity table
            st.subheader("📋 Recent Activity")
            recent_df = history_df.tail(5)[['timestamp', 'condition', 'confidence', 'risk_level']]
            recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("📊 No data yet. Start by analyzing some text in the Analysis tab!")
    
    # History Tab
    with tabs[2]:
        st.header("📈 Analysis History")
        
        if st.session_state.history:
            # Export button
            if st.button("📥 Export History as CSV"):
                history_df = pd.DataFrame([
                    {
                        'timestamp': h['timestamp'],
                        'text': h['text'][:100] + '...' if len(h['text']) > 100 else h['text'],
                        'condition': h['result']['condition'],
                        'confidence': h['result']['confidence'],
                        'risk_level': h['result']['risk_level']
                    }
                    for h in st.session_state.history
                ])
                
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"mental_health_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Clear history button
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.history = []
                st.rerun()
            
            # Display history
            for idx, entry in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Analysis {len(st.session_state.history) - idx} - {entry['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write("**Text:**")
                        st.write(entry['text'])
                        
                        st.write("**Results:**")
                        st.write(f"Condition: {entry['result']['condition'].upper()}")
                        st.write(f"Confidence: {entry['result']['confidence']:.1%}")
                        st.write(f"Risk Level: {entry['result']['risk_level']}")
                    
                    with col2:
                        # Sentiment gauge
                        sentiment_score = entry['result']['sentiment']['compound']
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=sentiment_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Sentiment"},
                            gauge={'axis': {'range': [-1, 1]},
                                  'bar': {'color': "purple"},
                                  'steps': [
                                      {'range': [-1, -0.5], 'color': "red"},
                                      {'range': [-0.5, 0.5], 'color': "yellow"},
                                      {'range': [0.5, 1], 'color': "green"}],
                                  'threshold': {'line': {'color': "black", 'width': 4},
                                              'thickness': 0.75,
                                              'value': sentiment_score}}))
                        fig_gauge.update_layout(height=200)
                        st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("📝 No history yet. Start analyzing text to build your history!")
    
    # Resources Tab
    with tabs[3]:
        st.header("💡 Mental Health Resources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Educational Resources")
            resources = {
                "Understanding Depression": "https://www.nimh.nih.gov/health/topics/depression",
                "Anxiety Disorders": "https://www.nimh.nih.gov/health/topics/anxiety-disorders",
                "Stress Management": "https://www.apa.org/topics/stress",
                "Mental Health Basics": "https://www.mentalhealth.gov/basics"
            }
            
            for title, link in resources.items():
                st.markdown(f"📖 [{title}]({link})")
            
            st.subheader("🎧 Recommended Apps")
            apps = {
                "Headspace": "Meditation and mindfulness",
                "Calm": "Sleep and relaxation",
                "Sanvello": "Anxiety management",
                "Youper": "Emotional health assistant",
                "Talkspace": "Online therapy platform"
            }
            
            for app, desc in apps.items():
                st.write(f"📱 **{app}**: {desc}")
        
        with col2:
            st.subheader("🏥 Professional Help")
            st.write("""
            **When to Seek Professional Help:**
            - Symptoms persist for more than 2 weeks
            - Difficulty functioning in daily life
            - Thoughts of self-harm or suicide
            - Substance abuse concerns
            - Relationship problems
            
            **Types of Mental Health Professionals:**
            - **Psychiatrists**: Medical doctors who can prescribe medication
            - **Psychologists**: Therapy and counseling experts
            - **Counselors**: Support for specific issues
            - **Social Workers**: Community resources and support
            """)
            
            st.subheader("📞 Hotlines & Crisis Support")
            st.error("""
            **Emergency Contacts:**
            - 🆘 Emergency: 911
            - 📞 988 Suicide & Crisis Lifeline
            - 💬 Crisis Text Line: Text HOME to 741741
            - 🏥 SAMHSA National Helpline: 1-800-662-4357
            """)
    
    # Show crisis resources modal if triggered
    if 'show_crisis' in st.session_state and st.session_state.show_crisis:
        with st.expander("🆘 Crisis Resources", expanded=True):
            st.error("""
            **If you're in immediate danger, please call 911**
            
            **Crisis Support:**
            - 988 Suicide & Crisis Lifeline (24/7)
            - Crisis Text Line: Text HOME to 741741
            - Veterans Crisis Line: 1-800-273-8255
            - LGBTQ National Hotline: 1-888-843-4564
            - National Eating Disorders Hotline: 1-800-931-2237
            """)
            if st.button("Close"):
                st.session_state.show_crisis = False
                st.rerun()

if __name__ == "__main__":
    main()