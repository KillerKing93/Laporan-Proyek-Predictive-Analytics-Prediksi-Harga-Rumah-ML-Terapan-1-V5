# **Laporan Proyek Machine Learning: Prediksi Harga Rumah**

- **Nama:** Alif Nurhidayat
- **Email:** alifnurhidayatwork@gmail.com | mc189d5y0351@student.devacademy.id
- **ID Dicoding:** Alif Nurhidayat | MC189D5Y0351

## **Pendahuluan**

Proyek _predictive analytics_ ini bertujuan untuk mengembangkan model _machine learning_ yang mampu memprediksi harga jual rumah (SalePrice) secara akurat berdasarkan berbagai atribut atau fitur properti. Laporan ini akan menguraikan seluruh alur kerja proyek, mulai dari pemahaman domain masalah dan data, persiapan data, proses pemodelan dengan dua algoritma regresi yang berbeda (Regresi Linear dan Random Forest Regressor), hingga evaluasi performa model untuk menentukan solusi prediktif terbaik. Dataset yang digunakan adalah "House Prices \- Advanced Regression Techniques" dari platform kompetisi Kaggle.

## **1\. Domain Proyek**

### **Latar Belakang**

Industri real estat merupakan salah satu sektor ekonomi vital yang mempengaruhi berbagai aspek kehidupan, mulai dari keputusan investasi individu hingga stabilitas pasar keuangan. Harga properti, khususnya rumah tinggal, dipengaruhi oleh beragam faktor yang kompleks dan seringkali saling berinteraksi. Faktor-faktor ini dapat mencakup karakteristik fisik properti (seperti luas bangunan, jumlah kamar, tahun pembangunan), kualitas material dan konstruksi, lokasi (lingkungan, aksesibilitas), hingga kondisi pasar saat penjualan.

Kemampuan untuk memprediksi harga rumah secara akurat memiliki nilai signifikan bagi berbagai pihak:

- **Bagi penjual dan agen real estat:** Prediksi harga yang tepat membantu dalam menetapkan harga jual yang kompetitif namun tetap menguntungkan, serta menyusun strategi pemasaran yang efektif.
- **Bagi pembeli:** Pemahaman mengenai faktor-faktor yang mempengaruhi harga dan estimasi harga yang wajar dapat membantu dalam proses negosiasi dan pengambilan keputusan pembelian yang lebih bijak.
- **Bagi investor properti:** Model prediktif dapat menjadi alat bantu dalam mengidentifikasi properti dengan potensi apresiasi nilai yang baik atau menilai kewajaran harga akuisisi.
- **Bagi lembaga keuangan:** Penilaian properti yang akurat penting dalam proses pemberian kredit pemilikan rumah (KPR) untuk mitigasi risiko.

### **Masalah yang Diangkat**

Penilaian harga rumah secara tradisional seringkali melibatkan proses manual yang memakan waktu dan dapat bersifat subjektif, bergantung pada pengalaman dan intuisi penilai. Dengan banyaknya variabel yang perlu dipertimbangkan, menentukan harga yang "benar" menjadi tantangan. Inkonsistensi dalam penilaian juga dapat terjadi. Oleh karena itu, diperlukan sebuah pendekatan yang lebih objektif, konsisten, dan berbasis data untuk memprediksi harga rumah.

### **Mengapa dan Bagaimana Masalah Harus Diselesaikan**

Masalah ini perlu diselesaikan untuk meningkatkan efisiensi, transparansi, dan objektivitas dalam transaksi jual beli properti. Dengan model prediktif harga rumah yang handal:

- Proses penilaian dapat dipercepat.
- Keputusan terkait harga dapat didasarkan pada analisis data yang komprehensif, mengurangi bias subjektif.
- Informasi mengenai faktor-faktor penentu harga menjadi lebih mudah diakses dan dipahami oleh berbagai pihak.

Solusinya adalah dengan menerapkan teknik _machine learning_, khususnya model regresi, untuk mempelajari pola hubungan antara berbagai atribut rumah dan harga jualnya dari data historis. Model yang telah dilatih kemudian dapat digunakan untuk memberikan estimasi harga untuk properti baru atau properti yang belum dinilai.

### **Referensi**

1. **Dataset:** Dataset yang digunakan adalah "House Prices \- Advanced Regression Techniques" yang bersumber dari kompetisi Kaggle ([https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)). Dataset ini disediakan oleh Dean De Cock dan sering digunakan sebagai _benchmark_ untuk masalah regresi dalam prediksi harga properti. Deskripsi lengkap mengenai setiap kolom dan asal-usul data telah didokumentasikan dengan baik.
2. **Teknik Machine Learning untuk Prediksi Harga Properti:**
   - Penggunaan model regresi, termasuk Regresi Linear dan model _ensemble_ seperti Random Forest, merupakan pendekatan umum dalam literatur prediksi harga rumah. Studi seperti yang dilakukan oleh Gu, S., Kelly, B., & Xiu, D. (2020) dalam "Empirical Asset Pricing via Machine Learning" (_Review of Financial Studies_, 33(3), 1007-1057), meskipun fokus pada aset keuangan, menunjukkan efektivitas teknik _machine learning_ dalam memodelkan hubungan kompleks yang relevan juga untuk prediksi harga aset riil.
   - Random Forest, yang dikembangkan oleh Breiman, L. (2001) ("Random Forests," _Machine Learning_, 45(1), 5-32), dikenal karena kemampuannya menangani hubungan non-linear, interaksi antar fitur, dan umumnya memberikan performa prediktif yang kuat pada berbagai jenis dataset.
   - Pentingnya pra-pemrosesan data, termasuk penanganan nilai yang hilang, _encoding_ fitur kategorikal, dan _scaling_ fitur numerik, juga merupakan praktik standar yang ditekankan dalam berbagai sumber, seperti buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" oleh Aurélien Géron (O'Reilly Media).

## **2\. Business Understanding**

### **Problem Statements**

1. Bagaimana cara membangun model _machine learning_ regresi yang akurat dan handal untuk memprediksi harga jual rumah (SalePrice) berdasarkan serangkaian atribut properti yang beragam (mencakup fitur numerik dan kategorikal)?
2. Fitur-fitur properti apa saja yang memiliki pengaruh paling signifikan terhadap variasi harga jual rumah di Ames, Iowa, berdasarkan analisis data dan interpretasi model (jika memungkinkan)?
3. Di antara dua pendekatan model regresi yang diusulkan—Regresi Linear sebagai model dasar dan Random Forest Regressor dengan optimasi hyperparameter—model manakah yang memberikan performa prediksi yang lebih superior, diukur menggunakan metrik evaluasi standar seperti R-squared (R²), Mean Absolute Error (MAE), dan Root Mean Squared Error (RMSE)?

### **Goals**

1. Mengembangkan model prediksi harga rumah dengan kemampuan generalisasi yang baik, yang mampu menjelaskan sebagian besar varians dalam data harga jual rumah. Target R² untuk model dasar (Regresi Linear) adalah \> 0.65 (sesuai hasil notebook yaitu 0.6866) dan untuk model yang lebih canggih (Random Forest Tuned) adalah R² ≥ 0.88 (sesuai hasil notebook yaitu 0.8918).
2. Menghasilkan model dengan tingkat kesalahan prediksi yang minimal, yang diindikasikan dengan nilai MAE dan RMSE yang serendah mungkin pada data uji.
3. Melakukan perbandingan objektif antara performa model Regresi Linear dan Random Forest yang telah dioptimalkan untuk menentukan solusi prediktif yang paling akurat dan dapat diandalkan untuk masalah ini.
4. Mengidentifikasi fitur-fitur kunci yang paling berpengaruh dalam menentukan harga rumah, jika memungkinkan dari interpretasi model terbaik.

### **Solution Statement**

Untuk mencapai tujuan-tujuan tersebut, pendekatan solusi yang diajukan adalah sebagai berikut, dengan mengimplementasikan dua model regresi dan mengevaluasi performanya:

**Solusi A (Model Baseline): Menggunakan Regresi Linear**

- **Langkah-langkah:**
  1. Melakukan pra-pemrosesan data secara komprehensif:
     - Memisahkan fitur dan target.
     - Mengidentifikasi kolom numerik dan kategorikal.
     - Menerapkan imputasi untuk menangani nilai yang hilang (strategi 'mean' untuk numerik, 'most_frequent' untuk kategorikal).
     - Menerapkan _scaling_ (StandardScaler) pada fitur numerik.
     - Menerapkan _one-hot encoding_ (OneHotEncoder dengan drop='first' dan handle_unknown='ignore') pada fitur kategorikal.
  2. Melatih model Regresi Linear pada data latih yang telah diproses.
  3. Mengevaluasi performa model pada data uji menggunakan metrik R², MAE, MSE, dan RMSE.
- **Pengukuran Keberhasilan Solusi A:** Model ini akan dianggap berhasil sebagai _baseline_ jika mencapai R² \> 0.65 pada data uji (hasil aktual notebook: 0.6866). Nilai MAE dan RMSE akan menjadi acuan untuk perbandingan.

**Solusi B (Model yang Ditingkatkan): Menggunakan Random Forest Regressor dengan Hyperparameter Tuning**

- **Langkah-langkah:**
  1. Menggunakan data yang telah dipra-pemroses dengan cara yang sama seperti pada Solusi A.
  2. Melatih model Random Forest Regressor.
  3. Melakukan optimasi hyperparameter untuk model Random Forest menggunakan GridSearchCV. Parameter yang di-_tuning_ (sesuai dengan yang diuji di notebook Project1_MLTerapan_V5 (1).ipynb):
     - regressor\_\_n_estimators: \[100, 150\]
     - regressor\_\_max_depth: \[10, 20\]
     - regressor\_\_min_samples_split: \[2, 5\]
     - regressor\_\_min_samples_leaf: \[1, 4\]
  4. Menggunakan validasi silang (cv=3) dan metrik R² sebagai skor untuk pemilihan model terbaik dalam GridSearchCV.
  5. Mengevaluasi performa model Random Forest yang telah dioptimalkan pada data uji menggunakan metrik R², MAE, MSE, dan RMSE.
- **Pengukuran Keberhasilan Solusi B:** Model ini akan dianggap berhasil jika menunjukkan peningkatan performa yang signifikan dibandingkan Regresi Linear, dengan target R² ≥ 0.88 (hasil aktual notebook: 0.8918), serta penurunan MAE dan RMSE yang substansial.

Pemilihan model akhir sebagai solusi terbaik akan didasarkan pada perbandingan metrik evaluasi dari kedua solusi ini. Model dengan R² tertinggi serta MAE dan RMSE terendah akan dipilih.

## **3\. Data Understanding**

### **Sumber Data**

Dataset: "House Prices \- Advanced Regression Techniques" dari Kaggle.  
Tautan: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data  
File untuk training: train.csv.

### **Jumlah Data**

File train.csv memiliki 1460 sampel dan 81 kolom. Kolom 'Id' dijadikan index, sehingga tersisa 80 kolom untuk analisis (79 fitur \+ 1 target SalePrice).

### **Variabel/Fitur pada Data**

Dataset terdiri dari 79 fitur dan 1 variabel target (SalePrice). Fitur-fitur tersebut mencakup:

- **Informasi Umum & Zonasi:** MSSubClass, MSZoning, LotFrontage, LotArea, Street, Alley, LotShape, LandContour, Utilities, LotConfig, LandSlope, Neighborhood.
- **Kondisi & Kualitas Properti:** OverallQual, OverallCond, YearBuilt, YearRemodAdd, ExterQual, ExterCond.
- **Struktur & Material Bangunan:** RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, MasVnrArea, Foundation.
- **Informasi Basement:** BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinSF1, BsmtFinType2, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF.
- **Sistem & Utilitas Internal:** Heating, HeatingQC, CentralAir, Electrical.
- **Informasi Ruangan & Luas:** 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, KitchenQual, TotRmsAbvGrd.
- **Fungsionalitas & Fitur Tambahan:** Functional, Fireplaces, FireplaceQu.
- **Informasi Garasi:** GarageType, GarageYrBlt, GarageFinish, GarageCars, GarageArea, GarageQual, GarageCond.
- **Area Eksternal & Fitur Lain:** PavedDrive, WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, PoolQC, Fence, MiscFeature, MiscVal.
- **Informasi Penjualan:** MoSold, YrSold, SaleType, SaleCondition.
- **Variabel Target:** SalePrice (harga jual rumah dalam dolar AS).

Deskripsi detail untuk setiap variabel tersedia dalam file data_description.txt yang menyertai dataset. Tipe data terdiri dari: float64 (3 kolom), int64 (34 kolom), dan object (43 kolom).

### **Analisis Data Eksploratif (EDA) & Visualisasi**

(Ringkasan temuan dari Project1_MLTerapan_V5 (1).ipynb)

- **Statistik Deskriptif SalePrice (berdasarkan output df\['SalePrice'\].describe()):**
  - Rata-rata (mean): $180,921.20
  - Standar Deviasi (std): $79,442.50
  - Nilai Minimum (min): $34,900.00
  - Persentil ke-25 (25%): $129,975.00
  - Median (50%): $163,000.00
  - Persentil ke-75 (75%): $214,000.00
  - Nilai Maksimum (max): $755,000.00
  - Distribusi SalePrice condong ke kanan (_right-skewed_).
- **Nilai yang Hilang (berdasarkan output df.isnull().sum().sort_values(ascending=False).head(20)):**
  - Kolom dengan jumlah nilai hilang tertinggi:
    - PoolQC: 1453
    - MiscFeature: 1406
    - Alley: 1369
    - Fence: 1179
  - Kolom lain dengan nilai hilang signifikan:
    - MasVnrType: 872
    - FireplaceQu: 690
    - LotFrontage: 259
    - Fitur terkait garasi: GarageCond (81), GarageQual (81), GarageFinish (81), GarageType (81), GarageYrBlt (81).
    - Fitur terkait _basement_: BsmtExposure (38), BsmtFinType2 (38), BsmtCond (37), BsmtFinType1 (37), BsmtQual (37).
    - MasVnrArea: 8
    - Electrical: 1
- **Distribusi Fitur Numerik Penting (Termasuk Target SalePrice):**
  - SalePrice: _Right-skewed_.
  - GrLivArea (Luas Area Tinggal di Atas Permukaan Tanah): _Right-skewed_, menunjukkan hubungan positif kuat dengan SalePrice.
  - TotalBsmtSF (Total Luas Basement): Terkonsentrasi di nilai rendah, beberapa _outlier_ besar. Sejumlah rumah tanpa _basement_.
  - LotArea (Luas Lahan): Sangat _right-skewed_.
  - YearBuilt (Tahun Dibangun): Menunjukkan dua puncak distribusi.
  - OverallQual (Kualitas Keseluruhan): Distribusi cenderung normal, hubungan positif kuat dengan SalePrice.
- **Boxplot SalePrice vs Fitur Kategorikal Pilihan:**
  - OverallQual: Kualitas lebih tinggi berkorelasi dengan harga lebih tinggi.
  - Neighborhood: Variasi harga signifikan antar lingkungan (misalnya, 'NoRidge', 'NridgHt' cenderung mahal).
  - CentralAir: Rumah dengan AC sentral (Y) harganya jauh lebih tinggi.
  - MSZoning: Tipe zonasi 'FV' dan 'RL' cenderung memiliki harga lebih tinggi.
- **Scatter Plot Fitur Numerik Pilihan vs SalePrice (diwarnai OverallQual):**
  - GrLivArea vs SalePrice: Hubungan positif kuat.
  - TotalBsmtSF vs SalePrice: Hubungan positif.
  - 1stFlrSF vs SalePrice: Mirip dengan TotalBsmtSF.
  - GarageArea vs SalePrice: Hubungan positif, namun ada titik jenuh.
- **Heatmap Korelasi (Fitur Numerik):**
  - SalePrice berkorelasi positif kuat dengan: OverallQual (0.79), GrLivArea (0.71), GarageCars (0.64), GarageArea (0.62), TotalBsmtSF (0.61), 1stFlrSF (0.61), FullBath (0.56), TotRmsAbvGrd (0.53), YearBuilt (0.52), dan YearRemodAdd (0.51).
  - Multikolinearitas teridentifikasi antara GarageCars & GarageArea (0.88), serta TotalBsmtSF & 1stFlrSF (0.82).

## **4\. Data Preparation**

Tahapan persiapan data dilakukan untuk membersihkan, mentransformasi, dan menyusun data agar siap digunakan untuk melatih model _machine learning_. Urutan dan teknik yang digunakan sesuai dengan implementasi pada notebook Project1_MLTerapan_V5 (1).ipynb.

1. **Pemisahan Fitur (X) dan Target (y):**
   - Proses: Kolom SalePrice dipisahkan dari DataFrame utama (df) untuk menjadi variabel target (y). Sisa kolom (setelah menghapus SalePrice dan 'Id' yang sudah menjadi index) menjadi matriks fitur (X).
   - Alasan: Ini adalah langkah fundamental dalam _supervised learning_ untuk membedakan antara variabel yang ingin diprediksi (target) dan variabel yang digunakan untuk membuat prediksi (fitur).
2. **Definisi Kolom Numerik dan Kategorikal untuk Preprocessing:**
   - Proses: Nama-nama kolom dalam X diidentifikasi dan dikelompokkan menjadi dua daftar: numerical_cols_prep untuk fitur numerik (berdasarkan tipe data np.number, 36 fitur) dan categorical_cols_prep untuk fitur kategorikal (berdasarkan tipe data object atau category, 43 fitur).
   - Alasan: Fitur numerik dan kategorikal memerlukan teknik pra-pemrosesan yang berbeda. Pemisahan ini memungkinkan penerapan transformasi yang sesuai menggunakan ColumnTransformer.
3. **Pembuatan Pipeline Pra-pemrosesan dengan ColumnTransformer:** Sebuah ColumnTransformer (preprocessor) dibuat untuk menerapkan serangkaian transformasi spesifik pada jenis kolom yang berbeda:
   - **Untuk Fitur Numerik (numerical_cols_prep):**
     - SimpleImputer(strategy='mean'):
       - Proses: Mengisi nilai-nilai yang hilang (NaN) pada fitur numerik dengan nilai rata-rata (_mean_) dari masing-masing kolom.
       - Alasan: Strategi imputasi umum untuk data hilang numerik, memastikan tidak ada NaN yang masuk ke model.
     - StandardScaler():
       - Proses: Menstandarisasi fitur numerik (mean=0, std=1).
       - Alasan: Penting untuk algoritma sensitif skala, memastikan kontribusi fitur seimbang dan membantu konvergensi.
   - **Untuk Fitur Kategorikal (categorical_cols_prep):**
     - SimpleImputer(strategy='most_frequent'):
       - Proses: Mengisi NaN dengan nilai yang paling sering muncul (modus).
       - Alasan: Strategi umum untuk data hilang kategorikal.
     - OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False):
       - Proses: Mengubah variabel kategorikal menjadi fitur biner.
       - Alasan: Model memerlukan input numerik. handle_unknown='ignore' meningkatkan robustisitas. drop='first' menghindari multikolinearitas. sparse_output=False menghasilkan _dense array_.
   - Alasan Penggunaan ColumnTransformer: Menerapkan transformasi spesifik secara efisien dan konsisten ke subset kolom yang berbeda.
4. **Pembagian Data Latih dan Data Uji (train_test_split):**
   - Proses: Matriks fitur X dan target y dibagi 80% latih (1168 sampel) dan 20% uji (292 sampel). random_state=42 untuk reproduktifitas.
   - Alasan: Mengevaluasi model pada data yang belum pernah dilihat untuk mengukur kemampuan generalisasi.

## **5\. Modeling**

Dua model regresi diimplementasikan: Regresi Linear sebagai _baseline_ dan Random Forest Regressor sebagai model yang lebih kompleks dengan optimasi hyperparameter. Keduanya diintegrasikan dalam Pipeline Scikit-learn.

### **Model 1: Regresi Linear (Baseline)**

- **Tahapan Pemodelan:**
  - Pipeline (lr_pipeline) dibuat dengan preprocessor dan LinearRegression().
  - Dilatih dengan X_train dan y_train.
- **Parameter yang Digunakan:** Parameter default Scikit-learn.
- **Kelebihan:** Sederhana, mudah diinterpretasi koefisiennya (dengan catatan), komputasi cepat, tidak banyak hyperparameter.
- **Kekurangan:** Mengasumsikan hubungan linear, sensitif terhadap _outlier_ dan multikolinearitas (meskipun drop='first' pada OHE membantu), kurang fleksibel untuk pola data kompleks.

### **Model 2: Random Forest Regressor (dengan Hyperparameter Tuning)**

- **Tahapan Pemodelan:**
  - Pipeline (rf_pipeline) dibuat dengan preprocessor dan RandomForestRegressor(random_state=42, n_jobs=-1).
  - Optimasi hyperparameter menggunakan GridSearchCV.
- **Proses Improvement dengan Hyperparameter Tuning (GridSearchCV):**
  - GridSearchCV secara sistematis menguji kombinasi hyperparameter dari param_grid_rf untuk menemukan yang terbaik.
  - param_grid_rf yang diuji (sesuai notebook Project1_MLTerapan_V5 (1).ipynb):
    - regressor\_\_n_estimators: \[100, 150\]
    - regressor\_\_max_depth: \[10, 20\]
    - regressor\_\_min_samples_split: \[2, 5\]
    - regressor\_\_min_samples_leaf: \[1, 4\]
  - Konfigurasi GridSearchCV: cv=3 (3-_fold cross-validation_), n_jobs=-1 (menggunakan semua _core_ CPU), verbose=2, scoring='r2'.
  - Dilatih pada X_train dan y_train.
  - **Parameter Terbaik Hasil Tuning (sesuai notebook):** {'regressor\_\_max_depth': 10, 'regressor\_\_min_samples_leaf': 1, 'regressor\_\_min_samples_split': 2, 'regressor\_\_n_estimators': 150}.
- **Kelebihan:** Umumnya memberikan akurasi tinggi, mampu menangani non-linearitas dan interaksi fitur, robust terhadap _outlier_, mengurangi _overfitting_ melalui _bagging_ dan pemilihan fitur acak, dapat memberikan estimasi _feature importance_.
- **Kekurangan:** Lebih bersifat _black-box_ (kurang _interpretable_) dibandingkan Regresi Linear, komputasi lebih intensif, performa bergantung pada _tuning hyperparameter_, bisa bias pada fitur dengan kardinalitas sangat tinggi jika tidak ditangani.

### **Pemilihan Model Terbaik sebagai Solusi**

Berdasarkan hasil evaluasi (dijelaskan di bawah), **Random Forest (Tuned)** dipilih sebagai model terbaik karena menunjukkan performa yang secara signifikan lebih baik daripada Regresi Linear pada semua metrik evaluasi (R² lebih tinggi, MAE dan RMSE lebih rendah).

## **6\. Evaluasi**

Performa model dievaluasi pada data uji (X_test, y_test) menggunakan metrik berikut:

1. **Mean Absolute Error (MAE):**
   - Formula: MAE \= (1/n) \* Σ |yᵢ \- ŷᵢ|
   - Cara Kerja: Mengukur rata-rata selisih absolut antara nilai aktual (yᵢ) dan prediksi (ŷᵢ). Memberikan bobot sama untuk semua error dan memiliki unit yang sama dengan target (dolar AS). Nilai lebih rendah lebih baik.
   - Kesesuaian: Sangat relevan untuk prediksi harga, memberikan gambaran langsung dampak finansial rata-rata dari ketidakakuratan.
2. **Mean Squared Error (MSE):**
   - Formula: MSE \= (1/n) \* Σ (yᵢ \- ŷᵢ)²
   - Cara Kerja: Menghitung rata-rata kuadrat selisih. Memberikan "hukuman" lebih besar untuk prediksi dengan error besar. Unitnya adalah kuadrat unit target (dolar AS kuadrat), kurang intuitif. Nilai lebih rendah lebih baik.
   - Kesesuaian: Berguna jika error besar dianggap lebih merugikan.
3. **Root Mean Squared Error (RMSE):**
   - Formula: RMSE \= √MSE
   - Cara Kerja: Akar kuadrat MSE, kembali ke unit target (dolar AS). Lebih mudah diinterpretasi daripada MSE, sambil tetap memberi bobot lebih pada error besar. Nilai lebih rendah lebih baik.
   - Kesesuaian: Metrik pilihan karena keseimbangan interpretabilitas dan sensitivitas error besar.
4. **R-squared (R²) Score (Koefisien Determinasi):**
   - Formula: R² \= 1 \- (SSres / SStot), di mana SSres adalah _sum of squares of residuals_ dan SStot adalah _total sum of squares_.
   - Cara Kerja: Proporsi varians variabel dependen yang dapat dijelaskan oleh variabel independen dalam model. Nilai berkisar \-∞ hingga 1\. R² \= 1 prediksi sempurna, R² \= 0 model setara prediksi rata-rata. Nilai lebih tinggi lebih baik.
   - Kesesuaian: Memberikan ukuran relatif "goodness-of-fit" model, penting untuk membandingkan kemampuan penjelasan model.

### **Hasil Proyek Berdasarkan Metrik Evaluasi (dari Notebook Project1_MLTerapan_V5 (1).ipynb)**

| Model                 | MAE       | MSE              | RMSE      | R2     |
| :-------------------- | :-------- | :--------------- | :-------- | :----- |
| Linear Regression     | 20,111.24 | 2,403,958,950.40 | 49,030.18 | 0.6866 |
| Random Forest (Tuned) | 17,856.53 | 829,641,491.96   | 28,803.50 | 0.8918 |

### **Interpretasi Hasil**

- **Regresi Linear (Baseline):**
  - MAE 20,111.24: Rata-rata, prediksi harga rumah oleh model ini menyimpang sekitar $20,111.24 dari harga aktual.
  - RMSE 49,030.18: Merupakan ukuran standar deviasi dari error prediksi.
  - R² 0.6866: Model ini mampu menjelaskan sekitar 68.66% variabilitas dalam harga rumah. Ini adalah performa yang cukup sebagai _baseline_.
- **Random Forest (Tuned):**
  - MAE 17,856.53: Rata-rata kesalahan prediksi harga oleh model ini lebih rendah, yaitu sekitar $17,856.53. Ini menunjukkan peningkatan akurasi sekitar 11.2% dalam MAE dibandingkan Regresi Linear.
  - RMSE 28,803.50: RMSE juga menunjukkan penurunan yang signifikan (sekitar 41.3%) dibandingkan Regresi Linear, mengindikasikan bahwa model ini lebih baik dalam menghindari error prediksi yang besar.
  - R² 0.8918: Model ini mampu menjelaskan sekitar 89.18% variabilitas dalam harga rumah. Ini menunjukkan peningkatan performa yang sangat baik dibandingkan Regresi Linear dan telah mencapai target R² ≥ 0.88.

Berdasarkan semua metrik evaluasi ini, **Random Forest (Tuned)** jelas merupakan model yang superior untuk memprediksi harga rumah dalam konteks dataset dan fitur yang digunakan.

## **Penyimpanan Model**

Model terbaik (Random Forest (Tuned)) yang mencakup _preprocessor_ dan _regressor_ yang telah di-_tuning_ disimpan ke file random_forest_tuned_house_price_model.joblib menggunakan pustaka joblib.

## **(Opsional) Contoh Pemuatan Model dan Prediksi**

Model yang telah disimpan berhasil dimuat kembali dari file random_forest_tuned_house_price_model.joblib. Kemudian, model ini digunakan untuk melakukan prediksi pada dua sampel data baru yang diambil dari X_test (rumah dengan Id 417 dan 775, sesuai output notebook):

- Data ke-1 (Id 417\) \=\> Prediksi Harga: $157,520.52
- Data ke-2 (Id 775\) \=\> Prediksi Harga: $308,738.42  
  Ini mendemonstrasikan bahwa model dapat digunakan kembali untuk prediksi di masa mendatang.

## **Kesimpulan**

Proyek ini berhasil mengembangkan dan mengevaluasi dua model _machine learning_—Regresi Linear dan Random Forest Regressor (dengan _hyperparameter tuning_)—untuk memprediksi harga jual rumah.

Setelah melalui tahapan pemahaman domain, analisis data eksploratif, persiapan data yang cermat, serta proses pemodelan dan optimasi, ditemukan bahwa model **Random Forest Regressor yang telah dioptimalkan** menunjukkan performa yang jauh lebih unggul.

- Model Random Forest (Tuned) mencapai skor **R² sebesar 0.8918** pada data uji, yang berarti mampu menjelaskan sekitar 89.18% variabilitas dalam harga rumah.
- Kesalahan prediksi rata-rata (MAE) untuk model Random Forest adalah **$17,856.53**, dan RMSE adalah **$28,803.50**.
- Sebagai perbandingan, model Regresi Linear menghasilkan R² sebesar 0.6866, MAE sebesar $20,111.24, dan RMSE sebesar $49,030.18.

Peningkatan performa yang signifikan pada semua metrik evaluasi menunjukkan bahwa Random Forest lebih mampu menangkap hubungan kompleks dan non-linear dalam data harga rumah. Model Random Forest (Tuned) telah disimpan untuk potensi penggunaan di masa mendatang.

Meskipun target R² yang sangat ambisius (misalnya, ≥ 0.95) belum tercapai sepenuhnya, hasil R² 0.8918 sudah sangat baik dan menunjukkan potensi besar dari pendekatan _machine learning_ untuk masalah ini. Sebagai pengembangan lebih lanjut, beberapa area yang dapat dieksplorasi meliputi:

- _Feature engineering_ yang lebih mendalam.
- Penanganan _outlier_ yang lebih canggih.
- Eksplorasi algoritma regresi lain (misalnya, Gradient Boosting, XGBoost, LightGBM).
- _Hyperparameter tuning_ yang lebih ekstensif.
- Analisis interpretasi model yang lebih dalam (misalnya, menggunakan SHAP _values_).
