# PyTorch Softmax Regression Assignment

Bu proje CMPE 442 (Introduction to Machine Learning) dersi "Programming Assignment-1" için bir iskelet kod (skeleton code) ve çalışma alanıdır.

## Gereksinimler ve Kurulum
Projenin ihtiyaç duyduğu Python kütüphaneleri `requirements.txt` dosyasında listelenmiştir. Python Virtual Environment (`venv`) ortamını kurup çalıştırarak sisteme etki etmeden kütüphaneleri kurabilirsiniz.

Sizin için `venv` isminde bir virtual environment klasörü oluşturulmuştur.

### 1- Ortamı (Environment) Aktif Etme
Windows komut satırında proje ana dizininde (bu klasörde) terminal açın:

```cmd
venv\Scripts\activate
```

Kurulum başarılı şekilde tamamlandığında satırın solunda `(venv)` yazacaktır.

### 2- Gerekli Paketlerin Yüklenmesi
```cmd
pip install -r requirements.txt
```

Bu kod; PyTorch, Scikit-Learn, Matplotlib, Numpy ve Pandas gibi ödevde belirtilen paketleri yükler.

---

## Proje Dosya Yapısı ve Çözüm Yolu

Kod modüler olması ve karmaşanın engellenmesi amacıyla `src/` klasörü içerisinde parçalara ayrılmıştır:

- `src/dataset.py`: Iris veri setinin yüklenmesi, belirtilen `%.70, %.15, %.15` veri ayırma işlemi ve polinom özellik genişletme bölümünü gerçekleştirir. "TODO" şeklinde belirtilen lineer, kuadratik, 3. dereceden polinom terimlerini (Örn. $x_i^2$, $x_i x_j$) genişletme adımlarını burada tamamlayabilirsiniz.
- `src/model.py`: Modelinizi inşa edeceğiniz yer. Ödevde, `nn.Linear` kullanmanız **yasaklanmış** ve `Ağırlık (Weight)` ve `Eğim (Bias)` değerlerini Matrix operasyonlarıyla kullanarak Softmax yapılması istenmiştir. Bu matris ağırlıkları bu dosyada PyTorch Tensörleri olarak oluşturulmuştur.
- `src/train.py`: Cross Validation (K=3) ile en iyi dereceyi seçmek ve modelin train dongüsü ve regularizasyonlarin (Ridge, Lasso, ElasticNet) `for` loop içinde çalıştırıldığı yerdir.
- `src/plot.py`: Train, Validation, Test kayıplarının (Loss) Epoch değerine göre çizilmesi ve "Accuracy, Precision, Recall, F1" hesaplarının olduğu matplotlib kodlarıdır.
- `src/main.py`: Bütün parçaları birbirine bağlayan ana akıştır. 3 farkli Regularizasyon x 3 farklı öğrenme oranı (Learning Rate) = Toplam 9 modeli teker teker bu dosya eğitir (train).

Ayrıca yazılacak raporda eklenmesi istenilen `Bias-Variance` analizi gibi metinleri ve oluşan 9 model için olan figür çizimlerini değerlendirerek `report.pdf` şeklinde projenize eklemelisiniz.

---
**Önemli Notlar:**
- Virtual Environment kullanmayı unutmayınız.
- Ödevinizde bir `nn.Linear()` fonksiyonunun kullanılmamasına çok dikkat ediniz.

Başarılar!
