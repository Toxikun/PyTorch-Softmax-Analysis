# PyTorch Softmax Regression (Iris Dataset)

[English](#english) | [Türkçe](#türkçe)

---

## English

This project is a workspace and skeleton code for the CMPE 442 (Introduction to Machine Learning) "Programming Assignment-1".

### Requirements & Setup
The required Python libraries for the project are listed in `requirements.txt`. You can install the libraries without affecting your system by setting up and running a Python Virtual Environment (`venv`).

A virtual environment folder named `venv` has already been created for you.

#### 1- Activating the Environment
Open a terminal in the project's root directory (this folder) on Windows:

```cmd
venv\Scripts\activate
```

When successfully activated, you will see `(venv)` on the left side of the line.

#### 2- Installing Required Packages
```cmd
pip install -r requirements.txt
```

This command installs the packages specified in the assignment such as PyTorch, Scikit-Learn, Matplotlib, Numpy, Pandas, and Seaborn.

---

### Project File Structure and Solution Path

To keep the code modular and avoid clutter, it is divided into parts within the `src/` folder:

- `src/dataset.py`: Imports the Iris dataset from Sklearn, splits the data into 70%, 15%, 15% train/val/test sets, and converts it to PyTorch tensors. The feature expansion for linear, quadratic ($x_i^2$, $x_i x_j$), and 3rd order (cubic) terms have been completed here.
- `src/model.py`: The place where you build your model. In the assignment, using `nn.Linear` is **forbidden**, and it is required to use Matrix operations with `Weight (W)` and `Bias (b)` values for Softmax. These matrix weights are created here as PyTorch Tensors.
- `src/train.py`: Contains the K-Fold (K=3) cross-validation to select the best polynomial degree, the training loop, and where regularizations (Ridge, Lasso, ElasticNet) are applied. A Learning Rate Scheduler has also been added here to improve convergence.
- `src/plot.py`: Contains matplotlib/seaborn code to plot the Train, Validation, and Test losses based on Epoch values, and calculates the "Accuracy, Precision, Recall, F1" metrics for the best model.
- `src/main.py`: The main flow that connects all parts together. It evaluates 3 different Regularizations x 3 different Learning Rates = a total of 9 models, trains them sequentially, and prints the evaluation metrics to the console.

Additionally, you should review terms like `Bias-Variance` analysis and evaluate the generated figure lines for the 9 models, and include them in your `report.pdf`.

---
**Important Notes:**
- Do not forget to use the Virtual Environment!
- Pay close attention to the rule of not using the `nn.Linear()` function in your assignment.

Good luck!

<br>

---
---

<br>

## Türkçe

Bu proje CMPE 442 (Introduction to Machine Learning) dersi "Programming Assignment-1" için bir çalışma alanı ve tamamlanmış iskelet koddur.

### Gereksinimler ve Kurulum
Projenin ihtiyaç duyduğu Python kütüphaneleri `requirements.txt` dosyasında listelenmiştir. Python Virtual Environment (`venv`) ortamını kurup çalıştırarak sisteme etki etmeden kütüphaneleri kurabilirsiniz.

Sizin için `venv` isminde bir virtual environment klasörü oluşturulmuştur.

#### 1- Ortamı (Environment) Aktif Etme
Windows komut satırında proje ana dizininde (bu klasörde) terminal açın:

```cmd
venv\Scripts\activate
```

Kurulum başarılı şekilde tamamlandığında satırın solunda `(venv)` yazacaktır.

#### 2- Gerekli Paketlerin Yüklenmesi
```cmd
pip install -r requirements.txt
```

Bu kod; PyTorch, Scikit-Learn, Matplotlib, Numpy, Pandas ve Seaborn gibi ödevde belirtilen paketleri yükler.

---

### Proje Dosya Yapısı ve Çözüm Yolu

Kod modüler olması ve karmaşanın engellenmesi amacıyla `src/` klasörü içerisinde parçalara ayrılmıştır:

- `src/dataset.py`: Iris veri setinin Sklearn içinden yüklenmesi, belirtilen `%70, %15, %15` veri ayırma işlemi ve verinin PyTorch tensörlerine çevrilmesi işlemini gerçekleştirir. Lineer, kuadratik ($x_i^2$, $x_i x_j$), ve 3. dereceden ($x_i^3$, $x_i x_j x_k$) polinom terimlerini genişletme adımları koda dahil edilmiştir.
- `src/model.py`: Modelinizi inşa edeceğiniz yer. Ödevde, `nn.Linear` kullanmanız **yasaklanmış** ve `Ağırlık (Weight)` ve `Eğim (Bias)` değerlerini Matrix operasyonlarıyla kullanarak Softmax yapılması istenmiştir. Bu matris ağırlıkları bu dosyada PyTorch Tensörleri olarak oluşturulmuştur.
- `src/train.py`: Cross Validation (K=3) ile en iyi dereceyi seçmek, modelin train döngüsü ve regularizasyonların (Ridge, Lasso, ElasticNet) uygulandığı yerdir. Ayrıca hedefe yaklaşırken yakınsamayı kolaylaştırmak için Learning Rate Scheduler eklenmiştir.
- `src/plot.py`: Train, Validation, Test kayıplarının (Loss) Epoch değerine göre çizilmesi ve en başarılı model üzerinden "Accuracy, Precision, Recall, F1" hesaplarının yapıldığı matplotlib/seaborn kodlarıdır.
- `src/main.py`: Bütün parçaları birbirine bağlayan ana akıştır. 3 farklı Regularizasyon x 3 farklı öğrenme oranı (Learning Rate) = Toplam 9 modeli teker teker eğitir, figürleri çıkarır ve sonuç verilerini konsola basar.

Ayrıca yazılacak raporda eklenmesi istenilen `Bias-Variance` analizi gibi metinleri ve oluşan 9 model için olan figür çizimlerini değerlendirerek `report.pdf` şeklinde projenize eklemelisiniz.

---
**Önemli Notlar:**
- Virtual Environment kullanmayı unutmayınız.
- Ödevinizde bir `nn.Linear()` fonksiyonunun kullanılmamasına çok dikkat ediniz.

Başarılar!
