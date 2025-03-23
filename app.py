import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pickle
import base64
from io import BytesIO

# Sayfa yapılandırması
st.set_page_config(
    page_title="Iris Türleri Sınıflandırması",
    page_icon="🌸",
    layout="wide"
)

# Başlık ve açıklama
st.title('Iris Türleri Sınıflandırması')
st.markdown("""
Bu uygulama, makine öğrenimi kullanarak Iris çiçek türlerinin sınıflandırılmasını gösterir.
SVM algoritması kullanılarak, çiçek özellikleri (taç yaprak ve çanak yaprak ölçüleri) üzerinden sınıflandırma yapılır.
""")

# Sidebar
st.sidebar.header('Uygulama Menüsü')
page = st.sidebar.radio('Sayfa Seçin:', 
    ['Ana Sayfa', 'Veri Keşfi', 'Model Eğitimi', 'Tahmin'])

# Veri yükleme fonksiyonu
@st.cache_data
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # DataFrame oluştur
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]
    df['target'] = y
    
    return df, X, y, feature_names, target_names

# Grafiği indirilebilir hale getiren yardımcı fonksiyon
def get_image_download_link(fig, filename, text):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Veriyi yükle
df, X, y, feature_names, target_names = load_data()

# Model eğitimi fonksiyonu
@st.cache_resource
def train_model(X, y, optimize=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if optimize:
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        model = SVC(probability=True)
        model.fit(X_train, y_train)
        best_params = None
        
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, X_train, X_test, y_train, y_test, y_pred, accuracy, report, best_params

# Ana Sayfa
if page == 'Ana Sayfa':
    st.header('Iris Veri Seti Hakkında')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Iris Veri Seti**, makine öğrenimi alanında en çok bilinen veri setlerinden biridir. Üç farklı Iris türünün (Setosa, Versicolor ve Virginica) dört farklı özelliğini içerir:
        
        1. Çanak yaprak uzunluğu (sepal length)
        2. Çanak yaprak genişliği (sepal width)
        3. Taç yaprak uzunluğu (petal length)
        4. Taç yaprak genişliği (petal width)
        
        Bu uygulama, Destek Vektör Makineleri (SVM) kullanarak bu üç farklı Iris türünü sınıflandırmayı göstermektedir.
        """)
    
    with col2:
        st.markdown("### Örnek Iris Türleri")
        st.image("https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/images/iris.png", 
                caption="Iris Setosa, Iris Versicolor, Iris Virginica", use_column_width=True)
    
    st.markdown("### Veri Seti Özeti")
    st.write(df.describe())
    
    # Veri setinde kaç örnek var?
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Toplam Örnek Sayısı", df.shape[0])
    with col2:
        st.metric("Özellik Sayısı", len(feature_names))
    with col3:
        st.metric("Sınıf Sayısı", len(target_names))
    
    # Her türden kaç örnek var?
    st.markdown("### Sınıf Dağılımı")
    fig, ax = plt.subplots(figsize=(8, 4))
    df['species'].value_counts().plot(kind='bar', ax=ax)
    plt.title('Iris Türlerinin Dağılımı')
    plt.ylabel('Örnek Sayısı')
    plt.xticks(rotation=0)
    st.pyplot(fig)

# Veri Keşfi
elif page == 'Veri Keşfi':
    st.header('Veri Keşfi ve Görselleştirme')
    
    viz_type = st.selectbox(
        'Görselleştirme Türü:',
        ['Pairplot', 'Violin Plot', 'PCA', 't-SNE 3D', 'Parallel Coordinates']
    )
    
    if viz_type == 'Pairplot':
        st.subheader('Pairplot: Tüm Özellikler Arası İlişkiler')
        fig = sns.pairplot(df, hue='species', palette='viridis')
        st.pyplot(fig)
        
    elif viz_type == 'Violin Plot':
        st.subheader('Violin Plot')
        feature = st.selectbox('Özellik Seçin:', feature_names)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(x='species', y=feature, data=df, palette='viridis', ax=ax)
        plt.title(f'{feature} by Species')
        st.pyplot(fig)
        
    elif viz_type == 'PCA':
        st.subheader('PCA: 2 Boyuta İndirgeme')
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        pca_df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'species': df['species']})
        
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='species', 
                         title='2D PCA: Iris Türleri', 
                         color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(width=700, height=500)
        st.plotly_chart(fig)
        
        # PCA açıklanan varyans oranı
        explained_variance = pca.explained_variance_ratio_
        st.write(f"PC1 açıklanan varyans: {explained_variance[0]:.2f}")
        st.write(f"PC2 açıklanan varyans: {explained_variance[1]:.2f}")
        st.write(f"Toplam açıklanan varyans: {sum(explained_variance):.2f}")
        
    elif viz_type == 't-SNE 3D':
        st.subheader('t-SNE: 3 Boyutlu Görselleştirme')
        
        # t-SNE parametrelerini ayarlama seçeneği
        perplexity = st.slider('Perplexity:', min_value=5, max_value=50, value=30)
        
        # t-SNE uygula
        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X)
        
        # 3D görselleştirme
        fig = px.scatter_3d(
            x=X_tsne[:,0], y=X_tsne[:,1], z=X_tsne[:,2],
            color=df['species'],
            labels={'color': 'Tür'},
            title='3B t-SNE: Iris Türleri',
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig.update_traces(marker_size=5)
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig)
        
    elif viz_type == 'Parallel Coordinates':
        st.subheader('Paralel Koordinatlar')
        # Paralel koordinatlar görselleştirmesi için plotly kullan
        fig = px.parallel_coordinates(
            df, 
            color="species",
            dimensions=feature_names,
            color_continuous_scale=px.colors.diverging.Tealrose,
            color_continuous_midpoint=1
        )
        fig.update_layout(width=900, height=600)
        st.plotly_chart(fig)

# Model Eğitimi
elif page == 'Model Eğitimi':
    st.header('SVM Model Eğitimi ve Değerlendirme')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        optimize = st.checkbox('Grid Search ile Hiperparametre Optimizasyonu', value=False)
        if optimize:
            st.info('Hiperparametre optimizasyonu biraz zaman alabilir.')
    
    with col2:
        scale_data = st.checkbox('Verileri Ölçeklendir', value=True)
        if scale_data:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            data_to_use = X_scaled
            st.info('Veriler ölçeklendirildi.')
        else:
            data_to_use = X
    
    if st.button('Modeli Eğit'):
        with st.spinner('Model eğitiliyor...'):
            model, X_train, X_test, y_train, y_test, y_pred, accuracy, report, best_params = train_model(data_to_use, y, optimize)
            
            st.success(f'Model eğitimi tamamlandı! Doğruluk: {accuracy:.4f}')
            
            if optimize and best_params:
                st.subheader('En İyi Hiperparametreler')
                st.json(best_params)
            
            # Sınıflandırma raporu
            st.subheader('Sınıflandırma Raporu')
            report_df = pd.DataFrame(report).transpose()
            st.write(report_df)
            
            # Karışıklık matrisi
            st.subheader('Karışıklık Matrisi')
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_names, yticklabels=target_names)
            plt.xlabel('Tahmin Edilen')
            plt.ylabel('Gerçek')
            plt.title('Karışıklık Matrisi')
            st.pyplot(fig)
            
            # Model dosyasını kaydet
            model_data = {
                'model': model,
                'scaler': scaler if scale_data else None,
                'feature_names': feature_names,
                'target_names': target_names,
                'scale_data': scale_data
            }
            
            # Model indirme bağlantısı
            pickle_buffer = BytesIO()
            pickle.dump(model_data, pickle_buffer)
            pickle_buffer.seek(0)
            
            st.download_button(
                label="Modeli İndir (pickle)",
                data=pickle_buffer,
                file_name="iris_svm_model.pkl",
                mime="application/octet-stream"
            )

# Tahmin
elif page == 'Tahmin':
    st.header('Yeni Veri ile Tahmin')
    
    # Örnek veriler
    if st.checkbox('Örnek veriler kullan'):
        sample_idx = st.selectbox('Örnek seçin:', range(len(X)))
        sepal_length = X[sample_idx, 0]
        sepal_width = X[sample_idx, 1]
        petal_length = X[sample_idx, 2]
        petal_width = X[sample_idx, 3]
        
        st.info(f"Gerçek tür: {target_names[y[sample_idx]]}")
    else:
        # Kullanıcı girdisi
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.slider('Çanak yaprak uzunluğu (cm):', 4.0, 8.0, 5.8, 0.1)
            sepal_width = st.slider('Çanak yaprak genişliği (cm):', 2.0, 4.5, 3.0, 0.1)
        with col2:
            petal_length = st.slider('Taç yaprak uzunluğu (cm):', 1.0, 7.0, 4.0, 0.1)
            petal_width = st.slider('Taç yaprak genişliği (cm):', 0.1, 2.5, 1.3, 0.1)
    
    # Özellik değerlerini göster
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    st.markdown('### Girilen Özellikler')
    input_df = pd.DataFrame({
        'Özellik': feature_names,
        'Değer': input_data[0]
    })
    st.write(input_df)
    
    # Tahmin yap
    if st.button('Tahmin Et'):
        # Model eğitimi
        use_scaling = st.checkbox('Verileri Ölçeklendir', value=True)
        
        with st.spinner('Tahmin yapılıyor...'):
            # Model eğitimi
            model, _, _, _, _, _, _, _, _ = train_model(X, y, optimize=False)
            
            # Veri ölçeklendirme
            if use_scaling:
                scaler = StandardScaler()
                scaler.fit(X)
                input_data_scaled = scaler.transform(input_data)
                prediction = model.predict(input_data_scaled)
            else:
                prediction = model.predict(input_data)
            
            # Tahmin sonucu
            pred_class = target_names[prediction[0]]
            
            # Olasılıklar (SVC için probability=True ayarlanmalı)
            probs = model.predict_proba(input_data_scaled if use_scaling else input_data)[0]
            
            # Sonuçları göster
            st.success(f'Tahmin Edilen Tür: **{pred_class}**')
            
            # Olasılık çubuğu
            prob_df = pd.DataFrame({
                'Tür': target_names,
                'Olasılık': probs
            })
            
            fig = px.bar(prob_df, x='Tür', y='Olasılık', color='Tür',
                        color_discrete_sequence=px.colors.qualitative.Vivid,
                        title='Sınıf Olasılıkları')
            fig.update_layout(xaxis_title='Iris Türü', yaxis_title='Olasılık')
            st.plotly_chart(fig)
            
            # Bu tahminin konumu (PCA ile)
            st.subheader('Bu Örneğin PCA Uzayındaki Konumu')
            
            # Original veri ile yeni veriyi birleştir
            combined_data = np.vstack([X, input_data])
            
            # PCA uygula
            pca = PCA(n_components=2)
            combined_pca = pca.fit_transform(combined_data)
            
            # İlk 150 nokta orijinal veri, son nokta yeni giriş
            X_pca = combined_pca[:-1]
            new_point_pca = combined_pca[-1]
            
            # Görselleştirme
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, label='Eğitim verileri')
            ax.scatter(new_point_pca[0], new_point_pca[1], color='red', s=100, edgecolor='k', marker='*', label='Yeni veri')
            ax.set_title('PCA: Yeni Verinin Konumu')
            ax.set_xlabel('Birinci Temel Bileşen')
            ax.set_ylabel('İkinci Temel Bileşen')
            plt.legend()
            
            # Colorbar ekle
            legend1 = ax.legend(*scatter.legend_elements(),
                              loc="upper right", title="Sınıflar")
            ax.add_artist(legend1)
            
            st.pyplot(fig)
            
# Footer
st.markdown('---')
st.markdown('Iris Türleri Sınıflandırma Uygulaması - Made with Streamlit')
