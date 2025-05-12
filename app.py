# app.py

import streamlit as st
import pandas as pd
from mental_health_detector import MentalHealthDetector

# Page config
st.set_page_config(
    page_title="Deteksi Kesehatan Mental",
    page_icon="🧠",
    layout="wide"
)

# Load model
@st.cache_resource
def load_detector():
    detector = MentalHealthDetector()
    detector.load_model('mental_health_model.pkl')
    return detector

# Rekomendasi berdasarkan kondisi
def get_recommendations(condition):
    recommendations = {
        'depresi': [
            '🏥 Konsultasi dengan psikolog atau psikiater',
            '🚶‍♀️ Lakukan aktivitas fisik ringan setiap hari',
            '💬 Berbagi cerita dengan orang terpercaya',
            '😴 Jaga pola tidur yang teratur',
            '🧘‍♀️ Coba meditasi atau mindfulness'
        ],
        'kecemasan': [
            '🫁 Latihan pernapasan dalam',
            '🧘‍♂️ Praktikkan yoga atau meditasi',
            '☕ Kurangi konsumsi kafein',
            '📝 Tulis jurnal untuk mengidentifikasi pemicu',
            '🎯 Fokus pada hal-hal yang bisa dikontrol'
        ],
        'stress': [
            '⏸️ Ambil istirahat secara teratur',
            '🎯 Prioritaskan tugas-tugas penting',
            '🤝 Minta bantuan jika diperlukan',
            '🎨 Lakukan hobi yang menyenangkan',
            '🏃‍♂️ Olahraga untuk melepas ketegangan'
        ],
        'normal': [
            '✅ Pertahankan pola hidup sehat',
            '💪 Terus kembangkan diri',
            '👥 Jaga hubungan sosial yang positif',
            '🎉 Rayakan pencapaian kecil',
            '🙏 Praktikkan rasa syukur'
        ]
    }
    return recommendations.get(condition, [])

# Main app
st.title("🧠 Sistem Deteksi Kesehatan Mental")
st.markdown("---")

# Sidebar
st.sidebar.header("📋 Informasi")
st.sidebar.info(
    "Aplikasi ini menggunakan AI untuk mendeteksi kondisi kesehatan mental "
    "berdasarkan teks yang Anda masukkan. "
    "\n\n⚠️ **Penting**: Ini hanya alat screening, bukan diagnosis medis!"
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Masukkan Teks")
    
    user_input = st.text_area(
        "Ceritakan perasaan atau kondisi Anda:",
        height=150,
        placeholder="Contoh: Hari ini saya merasa..."
    )
    
    if st.button("🔍 Analisis", type="primary"):
        if user_input:
            with st.spinner("Menganalisis..."):
                # Load model
                detector = load_detector()
                
                # Predict
                result = detector.predict(user_input)
                
                # Show results
                st.success("Analisis selesai!")
                
                # Condition
                condition = result['condition']
                confidence = result['confidence']
                
                # Display with color
                colors = {
                    'depresi': 'red',
                    'kecemasan': 'orange',
                    'stress': 'blue',
                    'normal': 'green'
                }
                
                st.markdown(f"### Kondisi: :{colors.get(condition, 'gray')}[{condition.upper()}]")
                st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                
                # Scores detail
                st.subheader("📊 Skor Detail")
                scores_df = pd.DataFrame([result['all_scores']])
                st.bar_chart(scores_df.T)
                
                # Recommendations
                st.subheader("💡 Rekomendasi")
                recommendations = get_recommendations(condition)
                for rec in recommendations:
                    st.write(rec)
        else:
            st.error("Silakan masukkan teks terlebih dahulu!")

with col2:
    st.header("ℹ️ Tentang Kondisi")
    
    with st.expander("🔴 Depresi"):
        st.write("""
        **Ciri-ciri:**
        - Perasaan sedih yang mendalam
        - Kehilangan minat/motivasi
        - Merasa tidak berharga
        - Kelelahan ekstrem
        
        **Kapan harus ke profesional:**
        Jika gejala berlangsung > 2 minggu
        """)
    
    with st.expander("🟠 Kecemasan"):
        st.write("""
        **Ciri-ciri:**
        - Kekhawatiran berlebihan
        - Sulit konsentrasi
        - Gangguan tidur
        - Gejala fisik (jantung berdebar, keringat)
        
        **Kapan harus ke profesional:**
        Jika mengganggu aktivitas harian
        """)
    
    with st.expander("🔵 Stress"):
        st.write("""
        **Ciri-ciri:**
        - Merasa tertekan/overwhelmed
        - Mudah marah/frustasi
        - Sakit kepala/otot tegang
        - Perubahan pola makan/tidur
        
        **Kapan harus ke profesional:**
        Jika tidak bisa mengatasi sendiri
        """)
    
    with st.expander("🟢 Normal"):
        st.write("""
        **Ciri-ciri:**
        - Mood stabil
        - Mampu mengatasi stress
        - Tidur nyenyak
        - Energi cukup
        
        **Tips maintenance:**
        Jaga pola hidup sehat!
        """)

# Footer
st.markdown("---")
st.markdown(
    "💡 **Disclaimer**: Aplikasi ini bukan pengganti konsultasi profesional. "
    "Jika Anda mengalami masalah kesehatan mental serius, segera hubungi psikolog/psikiater."
)