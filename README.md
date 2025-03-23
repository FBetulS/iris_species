# Iris Species Classification

## 📖 Proje Açıklaması
Bu proje, Iris veri setini kullanarak tür sınıflandırması yapmak için çeşitli veri analizi ve makine öğrenimi tekniklerini uygulamaktadır. Iris veri seti, istatistiksel ve makine öğrenimi algoritmalarını test etmek için yaygın olarak kullanılan başlangıç düzeyinde bir veri setidir. Proje, verilerin analizi, görselleştirilmesi ve destek vektör makineleri (SVM) kullanarak sınıflandırılması üzerine odaklanmaktadır.

⚠️ Not
3D grafiklerim ve görselleştirmelerim maalesef gözükmüyor. Bu durum, bazı tarayıcı veya platform uyumsuzluklarından kaynaklanabilir.

## 🔗 Veri Kümesi
Veri kümesi, [Iris Species](https://www.kaggle.com/datasets/uciml/iris/code?datasetId=19&sortBy=commentCount) adresinden alınmıştır. Bu veri seti, üç Iris türüne ait farklı özellikleri içermektedir ve her tür için 50 örnek bulunmaktadır.

## 🔗 Hugging Face Uygulaması
Ayrıca, projenin etkileşimli bir versiyonu [Iris Species - Hugging Face Space](https://huggingface.co/spaces/btulftma/iris-species) adresinde bulunmaktadır.

## 🛠️ Kullanılan Kütüphaneler
- `pandas`: Veri analizi ve manipülasyonu için.
- `matplotlib`: Görselleştirme için.
- `seaborn`: İleri düzey görselleştirme için.
- `plotly`: Etkileşimli görselleştirmeler için.
- `sklearn`: Makine öğrenimi uygulamaları için.

## 📊 Veri Analizi ve Görselleştirme
Proje, verilerin daha iyi anlaşılabilmesi için çeşitli görselleştirme teknikleri kullanır:
- **Pairplot**: Her tür için özelliklerin dağılımını gösterir.
- **Violin Plot**: Türlere göre petal uzunluğunu görselleştirir.
- **PCA ve t-SNE**: Veriyi düşük boyutlu uzaya indirgemek için kullanılır.

## 💻 Makine Öğrenimi
### Model Eğitimi
- **Veri Setinin Bölünmesi**: Veriler %80 eğitim ve %20 test olarak bölünmüştür.
- **Destek Vektör Makineleri (SVM)**: Model, SVM algoritması ile eğitilmiş ve doğruluğu değerlendirilmiştir.

### Model Performansı
Modelin doğruluğu ve sınıflandırma raporu aşağıdaki gibi elde edilmiştir:
```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
