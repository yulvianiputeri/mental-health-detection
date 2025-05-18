import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from mental_health_detector import AdvancedMentalHealthDetector
import random
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Deteksi Kesehatan Mental",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Memuat model
@st.cache_resource
def load_detector():
    detector = AdvancedMentalHealthDetector()
    
    # Cek apakah model sudah ada
    if not os.path.exists('mental_health_model_advanced.pkl'):
        st.error("""
        ❌ **Model belum tersedia!**
        
        🔧 **Solusi:**
        1. Tutup aplikasi ini (Ctrl+C di terminal)
        2. Jalankan perintah berikut di terminal:
           ```
           python train_model.py
           ```
        3. Setelah training selesai, jalankan lagi aplikasi:
           ```
           streamlit run app.py
           ```
        
        ℹ️ **Catatan:** Training model memerlukan waktu beberapa menit.
        """)
        st.stop()
    
    try:
        detector.load_model('mental_health_model_advanced.pkl')
        # st.success("✅ Model berhasil dimuat!")
        return detector
    except Exception as e:
        st.error(f"""
        ❌ **Error saat memuat model:**
        ```
        {str(e)}
        ```
        
        🔧 **Solusi:**
        1. Model mungkin corrupt. Latih ulang dengan:
           ```
           python train_model.py
           ```
        2. Pastikan semua dependencies terinstall:
           ```
           pip install -r requirements.txt
           ```
        """)
        st.stop()

# Rekomendasi yang ditingkatkan
def get_enhanced_recommendations(condition, risk_level):
    recommendations = {
        'depresi': {
            'tinggi': [
                '🚨 **DARURAT**: Segera hubungi profesional kesehatan mental',
                '📞 Hotline Krisis: 119 ext 8 (Sejiwa)',
                '🏥 Kunjungi UGD terdekat jika ada pikiran menyakiti diri',
                '👥 Jangan sendirian - hubungi teman/keluarga terpercaya',
                '💊 Pertimbangkan konsultasi obat dengan psikiater'
            ],
            'sedang': [
                '🩺 Jadwalkan pertemuan dengan psikolog/psikiater',
                '🏃‍♀️ Mulai rutinitas olahraga ringan (jalan 10 menit)',
                '😴 Prioritaskan kebersihan tidur (7-9 jam)',
                '📱 Gunakan aplikasi kesehatan mental (Headspace, Riliv)',
                '📓 Mulai journaling untuk memonitor mood'
            ],
            'rendah': [
                '🌱 Lakukan aktivitas self-care',
                '☀️ Berjemur di bawah sinar matahari setiap hari',
                '🥗 Makan makanan bergizi',
                '🧘‍♀️ Coba meditasi atau yoga',
                '🎨 Ikut aktivitas kreatif'
            ]
        },
        'kecemasan': {
            'tinggi': [
                '🚨 Cari bantuan profesional segera',
                '🫁 Latih teknik pernapasan darurat',
                '💊 Diskusikan obat anti-cemas dengan dokter',
                '🚫 Hindari kafein dan stimulan',
                '📱 Download aplikasi bantuan serangan panik'
            ],
            'sedang': [
                '🧘‍♂️ Pelajari teknik relaksasi',
                '📝 Buat jurnal kecemasan',
                '🏃‍♂️ Olahraga aerobik teratur',
                '🎯 Lawan pikiran negatif',
                '👥 Bergabung dengan grup support kecemasan'
            ],
            'rendah': [
                '☕ Batasi asupan kafein',
                '📅 Pertahankan rutinitas',
                '🌿 Coba teh herbal',
                '🎵 Dengar musik yang menenangkan',
                '📚 Baca buku self-help'
            ]
        },
        'stress': {
            'tinggi': [
                '🚨 Ambil cuti stress jika memungkinkan',
                '🩺 Konsultasi dokter tentang gejala stress',
                '🧘‍♀️ Wajib lakukan latihan mengurangi stress harian',
                '❌ Belajar mengatakan TIDAK untuk tanggung jawab tambahan',
                '💤 Prioritaskan istirahat dan pemulihan'
            ],
            'sedang': [
                '⏰ Terapkan strategi manajemen waktu',
                '🎯 Tetapkan tujuan dan batasan yang realistis',
                '🏖️ Rencanakan istirahat dan liburan rutin',
                '💪 Delegasikan tugas jika memungkinkan',
                '🧘‍♂️ Praktikkan meditasi mindfulness'
            ],
            'rendah': [
                '📱 Gunakan aplikasi produktivitas',
                '🎨 Lakukan hobi',
                '🌳 Habiskan waktu di alam',
                '👥 Jaga koneksi sosial',
                '🏃‍♀️ Aktivitas fisik teratur'
            ]
        },
        'normal': {
            'rendah': [
                '✅ Lanjutkan kebiasaan sehat saat ini',
                '📊 Check-in kesehatan mental rutin',
                '💪 Bangun kemampuan resiliensi',
                '🎯 Tetapkan tujuan pertumbuhan pribadi',
                '🤝 Bantu orang lain yang membutuhkan'
            ]
        }
    }
    
    return recommendations.get(condition, {}).get(risk_level.lower(), [])

# Aplikasi utama
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Sistem Deteksi Kesehatan Mental</h1>', unsafe_allow_html=True)
    
    # Inisialisasi session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Panel Kontrol")
        
        # Profil pengguna (demo)
        st.subheader("👤 Profil Pengguna")
        user_name = st.text_input("Nama", "Anonim")
        user_age = st.number_input("Umur", 18, 100, 25)
        
        st.markdown("---")
        
        # Info
        st.info(
            "Sistem AI ini menggunakan NLP dan machine learning canggih untuk mendeteksi pola kesehatan mental. "
            "Menganalisis sentimen teks, kata kunci, dan pola linguistik untuk memberikan wawasan."
        )
        
        st.warning(
            "⚠️ Ini BUKAN pengganti diagnosis profesional kesehatan mental. "
            "Silakan konsultasi dengan penyedia layanan kesehatan untuk saran medis."
        )
    
    # Area konten utama
    tabs = st.tabs(["🔍 Analisis", "📊 Dashboard", "📈 Riwayat"])
    
    # Tab Analisis
    with tabs[0]:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.header("💬 Analisis Teks")
            
            # Pilihan metode input
            input_method = st.radio(
                "Pilih metode input:",
                ["Pesan Tunggal", "Riwayat Chat", "Catatan Suara (Beta)"],
                horizontal=True
            )
            
            if input_method == "Pesan Tunggal":
                user_input = st.text_area(
                    "Bagikan pikiran dan perasaan Anda:",
                    height=150,
                    placeholder="Saya merasa... (Anda bisa pakai emoji juga! 😊😔)"
                )
                
                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
                with col_btn1:
                    analyze_btn = st.button("🔍 Analisis", type="primary", use_container_width=True)
                with col_btn2:
                    clear_btn = st.button("🗑️ Hapus", use_container_width=True)
                
                if analyze_btn and user_input:
                    with st.spinner("🧠 AI sedang menganalisis teks Anda..."):
                        # Memuat model
                        detector = load_detector()
                        
                        # Dapatkan prediksi
                        result = detector.predict(user_input)
                        
                        # Tambah ke riwayat
                        st.session_state.history.append({
                            'timestamp': datetime.now(),
                            'text': user_input,
                            'result': result
                        })
                        
                        # Tampilkan hasil
                        # st.success("✅ Analisis selesai!")
                        
                        # Kartu kondisi
                        condition_colors = {
                            'depresi': '#ff4444',
                            'kecemasan': '#ff8800',
                            'stress': '#ffbb33',
                            'normal': '#00C851'
                        }
                        
                        st.markdown(f"""
                        <div style="background-color: {condition_colors.get(result['condition'], '#gray')}; 
                                    color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                            <h2 style="margin: 0;">Kondisi Terdeteksi: {result['condition'].upper()}</h2>
                            <p style="margin: 0.5rem 0;">Confidence: {result['confidence']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Level risiko
                        risk_class = f"risk-{result['risk_level'].lower()}"
                        st.markdown(f"<p class='{risk_class}'>Level Risiko: {result['risk_level']}</p>", 
                                   unsafe_allow_html=True)
                        
                        # Metrik detail
                        st.subheader("📊 Analisis Detail")
                        met_col1, met_col2, met_col3 = st.columns(3)
                        
                        with met_col1:
                            st.metric("Skor Sentimen", f"{result['sentiment']['compound']:.3f}")
                        with met_col2:
                            st.metric("Margin Confidence", f"{result['confidence_margin']:.2%}")
                        with met_col3:
                            st.metric("Jumlah Kata", len(user_input.split()))
                        
                        # Distribusi probabilitas
                        st.subheader("🎯 Probabilitas Kondisi")
                        prob_df = pd.DataFrame(list(result['probabilities'].items()), 
                                              columns=['Kondisi', 'Probabilitas'])
                        fig = px.bar(prob_df, x='Kondisi', y='Probabilitas', 
                                    color='Kondisi', 
                                    color_discrete_map=condition_colors,
                                    title="Distribusi Keyakinan AI")
                        st.plotly_chart(fig, use_container_width=True)
                
                if clear_btn:
                    st.session_state.clear()
                    st.rerun()
            
            elif input_method == "Riwayat Chat":
                st.write("📱 Tempel riwayat chat Anda (satu pesan per baris):")
                chat_input = st.text_area(
                    "Pesan chat:",
                    height=200,
                    placeholder="Pesan 1\nPesan 2\nPesan 3\n..."
                )
                
                if st.button("📊 Analisis Riwayat", type="primary"):
                    if chat_input:
                        messages = [msg.strip() for msg in chat_input.split('\n') if msg.strip()]
                        
                        with st.spinner("🔄 Menganalisis riwayat chat..."):
                            detector = load_detector()
                            
                            # Buat timestamp sintetis
                            timestamps = [datetime.now() - timedelta(hours=i) for i in range(len(messages))]
                            
                            results = detector.analyze_chat_history(messages, timestamps)
                            
                            st.success("✅ Analisis riwayat chat selesai!")
                            
                            # Metrik ringkasan
                            summary = results['summary']
                            st.subheader("📈 Statistik Ringkasan")
                            
                            sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                            with sum_col1:
                                st.metric("Total Pesan", summary['total_messages'])
                            with sum_col2:
                                st.metric("Kondisi Dominan", summary['dominant_condition'].upper())
                            with sum_col3:
                                st.metric("Pesan Risiko Tinggi", summary['high_risk_messages'])
                            with sum_col4:
                                st.metric("Rerata Confidence", f"{summary['average_confidence']:.1%}")
                            
                            # Diagram pie distribusi kondisi
                            dist_df = pd.DataFrame(list(summary['condition_distribution'].items()),
                                                  columns=['Kondisi', 'Jumlah'])
                            fig_pie = px.pie(dist_df, values='Jumlah', names='Kondisi',
                                           title="Distribusi Kondisi dalam Riwayat Chat")
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # Analisis pesan individual
                            st.subheader("🔍 Analisis Pesan Individual")
                            for idx, result in enumerate(results['individual_results']):
                                with st.expander(f"Pesan {idx+1}: {result['message_preview']}"):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.write(f"**Kondisi:** {result['condition']}")
                                    with col2:
                                        st.write(f"**Confidence:** {result['confidence']:.1%}")
                                    with col3:
                                        st.write(f"**Risiko:** {result['risk_level']}")
            
            else:  # Catatan Suara
                st.info("🎤 Analisis suara segera hadir! Fitur ini akan memungkinkan Anda berbicara pikiran Anda.")
                st.write("Saat ini, silakan gunakan input teks.")
        
        with col2:
            st.header("💡 Rekomendasi")
            
            if 'history' in st.session_state and st.session_state.history:
                latest = st.session_state.history[-1]
                condition = latest['result']['condition']
                risk_level = latest['result']['risk_level']
                
                recommendations = get_enhanced_recommendations(condition, risk_level)
                
                # Styling berdasarkan risiko
                if risk_level == "High":
                    st.error("⚠️ RISIKO TINGGI TERDETEKSI - Tindakan Segera Disarankan")
                elif risk_level == "Medium":
                    st.warning("⚠️ Risiko Sedang - Bantuan Profesional Disarankan")
                else:
                    st.info("ℹ️ Risiko Rendah - Tindakan Pencegahan Disarankan")
                
                # Tampilkan rekomendasi
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
                
                # Sumber daya darurat
                if risk_level in ["High", "Medium"]:
                    st.markdown("---")
                    st.subheader("🆘 Sumber Daya Darurat")
                    st.error("""
                    **Hotline Krisis:**
                    - 119 ext 8 (Sejiwa) - Hotline Bunuh Diri & Krisis
                    - 500-454 (Halodoc) - Konsultasi Psikolog
                    - Darurat: 112/119
                    """)
                    
                    with st.expander("Cari Sumber Daya Lokal"):
                        st.write("🏥 [Direktori Psikolog & Psikiater](https://www.halodoc.com/psikolog)")
                        st.write("👥 [Grup Support](https://pijarpsikologi.org/)")
                        st.write("📱 [Aplikasi Kesehatan Mental](https://www.riliv.co/)")
            else:
                st.info("👈 Masukkan teks dan klik analisis untuk mendapat rekomendasi yang dipersonalisasi")
                
                # Tips umum
                st.subheader("🌟 Tips Kesehatan Mental Umum")
                tips = [
                    "🧘‍♀️ Lakukan mindfulness setiap hari",
                    "🏃‍♂️ Olahraga teratur meningkatkan mood",
                    "😴 Prioritaskan kualitas tidur",
                    "🥗 Jaga nutrisi seimbang",
                    "👥 Tetap terhubung secara sosial",
                    "📝 Buat jurnal mood",
                    "🎯 Tetapkan tujuan yang realistis",
                    "🌳 Habiskan waktu di alam"
                ]
                for tip in tips:
                    st.write(tip)
    
    # Tab Dashboard
    with tabs[1]:
        st.header("📊 Dashboard Kesehatan Mental")
        
        if st.session_state.history:
            # Konversi riwayat ke DataFrame
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
            
            # Grafik time series
            fig_time = go.Figure()
            
            conditions = history_df['condition'].unique()
            condition_colors = {
                'depresi': '#ff4444',
                'kecemasan': '#ff8800',
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
                title="Tren Kesehatan Mental Sepanjang Waktu",
                xaxis_title="Waktu",
                yaxis_title="Skor Confidence",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Statistik ringkasan
            col1, col2 = st.columns(2)
            
            with col1:
                # Frekuensi kondisi
                condition_counts = history_df['condition'].value_counts()
                fig_freq = px.bar(x=condition_counts.index, y=condition_counts.values,
                                 labels={'x': 'Kondisi', 'y': 'Frekuensi'},
                                 title="Frekuensi Kondisi",
                                 color=condition_counts.index,
                                 color_discrete_map=condition_colors)
                st.plotly_chart(fig_freq, use_container_width=True)
            
            with col2:
                # Distribusi level risiko
                risk_counts = history_df['risk_level'].value_counts()
                fig_risk = px.pie(values=risk_counts.values, names=risk_counts.index,
                                 title="Distribusi Level Risiko",
                                 color_discrete_map={'High': '#ff4444', 
                                                    'Medium': '#ff8800',
                                                    'Low': '#00C851'})
                st.plotly_chart(fig_risk, use_container_width=True)
            
            # Tren sentimen
            st.subheader("😊 Analisis Sentimen")
            fig_sentiment = go.Figure()
            fig_sentiment.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['sentiment'],
                mode='lines+markers',
                name='Skor Sentimen',
                line=dict(color='purple', width=3),
                marker=dict(size=8)
            ))
            
            fig_sentiment.update_layout(
                title="Tren Sentimen",
                xaxis_title="Waktu",
                yaxis_title="Skor Sentimen (-1 sampai 1)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # Tabel aktivitas terkini
            st.subheader("📋 Aktivitas Terkini")
            recent_df = history_df.tail(5)[['timestamp', 'condition', 'confidence', 'risk_level']]
            recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("📊 Belum ada data. Mulai dengan menganalisis teks di tab Analisis!")
    
    # Tab Riwayat
    with tabs[2]:
        st.header("📈 Riwayat Analisis")
        
        if st.session_state.history:
            # Tombol ekspor
            if st.button("📥 Ekspor Riwayat sebagai CSV"):
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
                    label="Unduh CSV",
                    data=csv,
                    file_name=f"riwayat_kesehatan_mental_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Tombol hapus riwayat
            if st.button("🗑️ Hapus Riwayat", type="secondary"):
                st.session_state.history = []
                st.rerun()
            
            # Tampilkan riwayat
            for idx, entry in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Analisis {len(st.session_state.history) - idx} - {entry['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write("**Teks:**")
                        st.write(entry['text'])
                        
                        st.write("**Hasil:**")
                        st.write(f"Kondisi: {entry['result']['condition'].upper()}")
                        st.write(f"Confidence: {entry['result']['confidence']:.1%}")
                        st.write(f"Level Risiko: {entry['result']['risk_level']}")
                    
            with col2:
                    # Gauge sentimen dengan key unik
                    sentiment_score = entry['result']['sentiment']['compound']
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=sentiment_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Sentimen"},
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
                    
                    # Tambahkan key unik di sini
                    st.plotly_chart(
                        fig_gauge, 
                        use_container_width=True,
                        key=f"sentiment_gauge_{entry['timestamp'].timestamp()}"  # Key unik
                    )
        else:
            st.info("📝 Belum ada riwayat. Mulai menganalisis teks untuk membangun riwayat Anda!")
    

if __name__ == "__main__":
    main()