# Laporan Proyek Machine Learning - Salma Fauziah Azzahra
## Domain Proyek

Stroke adalah gejala-gejala defisit fungsi saraf yang diakibatkan oleh penyakit pembuluh darah otak, bukan oleh sebab yang lain (WHO). Gangguan fungsi syaraf pada stroke disebabkan oleh gangguan peredaran darah otak non traumatik[1]. 

Menurut Organisasi Kesehatan Dunia (WHO) stroke adalah penyebab kematian ke-2 secara global, bertanggung jawab atas sekitar 11% dari total kematian[2]. Data World Health Organization (WHO) tahun 2012 menunjukkan sekitar 31% dari 56,5 juta orang atau 17,7 juta orang di seluruh dunia meninggal akibat penyakit jantung dan pembuluh darah. Dari seluruh kematian akibat penyakit kardiovaskuler, sebesar 7,4 juta disebabkan oleh Penyakit Jantung Koroner, dan 6,7 juta disebabkan oleh stroke[1].

Oleh karena itu deteksi dini dirasa sangat penting untuk mencengah dan meminimalisir terjadinya stroke. Untuk membantu dalam mempermudah deteksi dini, maka dibuatlah proyek _machine learning predictive analytics_ untuk tedeksi awal penyakit stroke berdasarkan faktor-faktor yang ada. Selain mempermudah, proyek ini juga mempercepat proses deteksi.

## Business Understanding
#### Problem Statements

- Bagaimana mendeteksi stroke?
- Faktor apa saja yang mempengaruhi terjadinya stroke?
- Faktor apa yang paling mempengaruhi terjadinya stroke?

#### Goals

- Dapat membuat model deteksi stroke 
- Mengetahui faktor apa saja yang mempengaruhi terjadinya stroke
- Mengetahui faktor yang paling mempengaruhi terjadinya stroke

#### Solution statements
- Menyiapkan data agar bisa digunakan untuk membangun model
-  Menganalisis data dengan melakukan univariate analysis dan multivariate analysis. Memahami data dengan disertai dengan visualisasi yang dapat membantu untuk mengetahui kolerasi antar fitur dan mendeteksi outlier.
-  Melakukan hyperparameter tuning menggunakan grid search dan menentukan solusi permasalahan dengan klasifikasi. Algoritma yang dipakai dalam proyek ini adalah Random Forest

## Data Understanding
Data yang digunakan dalam proyek ini merupakan dataset untuk memprediksi kemungkinan seseorang terkena stroke berdasarkan parameter input seperti jenis kelamin, usia, berbagai penyakit, dan status merokok. Dataset dapat diunduh pada [Kaggle: Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

Berikut informasi mengenai dataset :
- Dataset memiliki format CSV (Comma-Seperated Values).
- Dataset memiliki 5110 sample dengan 12 fitur.
- Dataset memiliki 4 fitur bertipe int64, 5 fitur bertipe object, dan 3 fitur bertipe float64

### Variabel-variabel pada Stroke Pediction Dataset adalah sebagai berikut:
- id: pengenal unik
- jenis kelamin: berisi "Male", "Female" atau "Other"
- usia: usia pasien
- hipertensi: 0 jika pasien tidak memiliki hipertensi, 1 jika pasien memiliki hipertensi
- penyakit_jantung: 0 jika pasien tidak memiliki penyakit jantung, 1 jika pasien memiliki penyakit jantung
- ever_married: berisi "No" atau "Yes"
- work_type: jenis pekerjaan, berisi "children", "Govt_jov", "Never_worked", "Private" atau "Self-employed"
- Residence_type: jenis tempt tinggal, berisi "Rural" atau "Urban"
- avg_glucose_level: kadar glukosa rata-rata dalam darah
- bmi: indeks massa tubuh
- smoking_status: berisi "formerly smoked", "never smoked", "smokes" atau "Unknown"
- stroke: 1 jika pasien mengalami stroke atau 0 jika tidak

Dari ke-12 fitur di atas dapat dilihat bahwa fitur id tidak mempengaruhi prediksi  stroke sehingga akan dihapus.

### Univarite Analysis
Univariate Analysis adalah menganalisis setiap fitur secara terpisah. Fitur pada dataset dibagi menjadi dua, yaitu numerical features dan categorical features.
```sh
numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
```

Membuat grafik bar pada setiap fitur:

![gender](https://user-images.githubusercontent.com/109077279/198947253-f6639ba0-1f84-4957-bf96-752cc6497024.png)

Gambar 1. **Fitur gender**

Pada Gambar 1, terlihat bahwa pada fitur genre terdapat 2994 data "Female" (persentase 58.6%) dan 2115 data "Male" (persentase 41.4%)



![ever_married](https://user-images.githubusercontent.com/109077279/198947433-ec23dd0d-0088-4828-ab10-4e525b156b76.png)

Gambar 2. **Fitur ever_married**

Pada Gambar 2, terlihat bahwa pada fitur ever_married terdapat 3353 data "Yes" (persentase 65.6%) dan 1757 data "No" (persentase 34.4%)



![work_type](https://user-images.githubusercontent.com/109077279/198947631-092c230e-6f1f-4e10-8fb9-d7a2222e3b4b.png)

Gambar 3. **Fitur work_type**

Pada Gambar 3, terlihat bahwa pada fitur work_type terdapat 2925 data "Private" (persentase 57.2%), 819 data "Self-employed" (persentase 16.0%), 687 data "children" (persentase 13.4%), 657 data "Govt_job" (persentase 12.9%), dan 22 data "Never_worked" (persentase 0.4%)



![Residence_type](https://user-images.githubusercontent.com/109077279/198947691-870f8823-9479-41b7-8681-4556dbc49c03.png)

Gambar 4. **Fitur Residence_type**

Pada Gambar 4, terlihat bahwa pada fitur Residence_type terdapat 2596 data "Urban" (persentase 50.8%) dan 2514 data "Rural" (persentase 49.2%)



![smoking_status](https://user-images.githubusercontent.com/109077279/198947756-6a1dc43b-d359-462b-9335-d73a56746597.png)

Gambar 5. **Fitur smoking_status**

Pada Gambar 5, terlihat bahwa pada fitur smoking status terdapat 1892 data "never smoked" (persentase 37%), 1544 data "Unknow" (persentase 30.2%), 855 data "formerly smoked" (persentase 17.3%), dan 789 data "smokes" (persentase 15.4%)



![univariate analisis (numerical features)](https://user-images.githubusercontent.com/109077279/198948005-a27fddea-57e7-4dae-b2b1-cd3943e013f2.png)

Gambar 6. **Analisis sebaran pada setiap fitur numerik**

Pada Gambar 6, berfokus pada data target, yaitu stroke memiliki 4861 untuk 0 (tidak stroke) dan 249 data untuk 1 (stroke). Dapat dilihat bahwa data pada stroke merupakan data yang tidak seimbang sehingga akan mempengaruhi akurasi saat mendeteksi. Permasalahan ini akan diselesaikan pada tahap data preparation.


## Multivariate Analyst
Multivariate Analysis menunjukkan hubungan antara dua atau lebih fitur dalam data.


![gender x stroke](https://user-images.githubusercontent.com/109077279/198948066-9651d76b-0466-4620-a1ee-3eba76a8b0b7.png)

Gambar 7. **Korelasi fitur gender dan stroke**

Pada Gambar 7, terlihat bahwa jumlah pria dan wanita yang terkena stroke hampir sama.


![ever_married x stroke](https://user-images.githubusercontent.com/109077279/198948130-0260e7f1-a6e2-49dd-b7c7-0aa97e705ef2.png)

Gambar 8. **Korelasi fitur ever_married dan stroke**

Pada Gambar 8, terlihat bahwa belum menikah mengurangi risiko stroke karena data setelah menikah memiliki jumlah penderita stroke lebih banyak.



![work_type x stroke](https://user-images.githubusercontent.com/109077279/198948202-24ce2368-a762-4cdd-99d9-70e03fdbb053.png)

Gambar 9. **Korelasi fitur work_type dan stroke**

Pada Gambar 9, terlihat bahwa pekerja pribadi lebih banyak terkena stroke.



![Residence_type x stroke](https://user-images.githubusercontent.com/109077279/198948264-2727079a-10b7-4b27-88f3-e072f9320e63.png)

Gambar 10. **Korelasi fitur Residence_type dan stroke**

Pada Gambar 10, terlihat bahwa Residence_type hampir memiliki jumlah data yang sama.



![smoking_status x stroke](https://user-images.githubusercontent.com/109077279/198948317-fc1390d8-7169-45bf-9e9a-0126b1830b84.png)

Gambar 11. **Korelasi fitur smoking_status dan stroke**

Pada Gambar 11, terlihat bahwa sampel tidak pernah merokok yang paling banyak mengalami stroke. Dan proporsi terkecil dari sampel penderita stroke adalah perokok tetapi orang yang sebelumnya merokok dan yang merokok (gabungan) menunjukkan tanda-tanda stroke jauh lebih banyak daripada orang yang tidak pernah merokok.



![Correlation matrix fitur numerical](https://user-images.githubusercontent.com/109077279/198948399-323c304b-4996-43ec-9874-04aa465ae0c2.png)

Gambar 12. **Korelasi antara semua fitur numerik**

Pada Gambar 12, berfokus pada korelasi stroke dengan fitur numerik lainnya, dapat dilihat bahwa yang memiliki korelasi dengan nilai besar adalah fitur **age**.


**Membuat Korelasi antara fitur age, bmi, dan stroke yang dapat dilihat pada Gambar 13**

![korelasi antara age x bmi x stroke](https://user-images.githubusercontent.com/109077279/198948448-41e2ee6f-e2c1-41d4-bd83-5f2d97351221.png)

Gambar 13. **Korelasi antara fitur age, bmi, dan stroke**

Pada Gambar 13, terlihat bahwa pasien dengan stroke berada pada rentang usia > 40 tahun dan rentang bmi di bawah 60.


## Data Preparation
- One Hot Encoding
One hot encoding adalah teknik mengubah data kategorik menjadi data numerik dimana setiap kategori menjadi kolom baru dengan nilai 0 atau 1. Fitur yang akan diubah menjadi numerik pada proyek ini adalah genre, ever_married, work_type, Resindece_type, dan smoking_status.

- Imbalance Data
Sebelum menguji data, dapat dilihat dari informasi sebelumnya bahwa data target (stroke) memiliki data yang tidak seimbang sehingga memungkinkan untuk mengurangi akurasi model saat proses deteksi. Oleh karena itu, solusi yang diberikan adalah menggunakan Random Over-Sampling.
Oversampling dapat didefinisikan sebagai menambahkan lebih banyak salinan ke kelas minoritas. Oversampling bisa menjadi pilihan yang baik ketika tidak memiliki banyak data untuk dikerjakan.
Fitur yang akan dilakukan oversampling adalah fitur stroke dengan nilai 1 (stroke), sehingga datanya akan berjumlah sama dengan stroke dengan nilai 0 (tidak stroke).

- Train Test Split
Train test split aja proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 9722 (setelah oversampling) dibagi menjadi 7777 (80%) untuk data latih dan 1945 (20%) untuk data uji.

- Normalization
Algoritma machine learning akan memiliki performa lebih baik dan bekerja lebih cepat jika dimodelkan dengan data seragam yang memiliki skala relatif sama. Salah satu teknik normalisasi yang digunakan pada proyek ini adalah Standarisasi dengan sklearn.preprocessing.StandardScaler.

## Modeling

Algoritma Penelitian ini melakukan pemodelan dengan algoritma Random Forest
- Random Forest Algoritma random forest adalah teknik dalam machine learning dengan metode ensemble. Teknik ini beroperasi dengan membangun banyak decision tree pada waktu pelatihan. Proyek ini menggunakan sklearn.ensemble RandomForestClassifier dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :

   - max_depth = Kedalaman maksimum setiap tree.
   - max_features = hyperparameter yang menentukan jumlah variable prediktor yang akan dipertimbangkan di setiap tree.

**Tahapan:**

- Import library yang dibutuhkan
```sh
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
```
Disini terdapat dua library yang akan digunakan, yaitu RandomForestClassifier untuk membuat algoritma machine learning dengan kasus Klasifikasi dan GridSearchCV untuk melakukan Hyperparameter tuning. Hyperparameter tuning adalah cara untuk mendapatkan parameter terbaik dari algoritma dalam membangun model. Salah satu teknik dalam hyperparameter tuning yang digunakan dalam proyek ini adalah grid search. 

- Hyperparameter tuning dengan grid search

```sh
params = {'max_depth': list(range(6,16)), "max_features" : [2,3,4,5,6]}

grid = GridSearchCV(RandomForestClassifier(random_state = 42), params, cv=10, scoring='accuracy', return_train_score=False,verbose=1)

grid.fit(X_train, y_train)

print(grid.best_params_)
```

Berikut adalah hasil dari Grid Search pada proyek ini :

![gridsearch-final](https://user-images.githubusercontent.com/109077279/198955168-49b15c24-7e32-4b05-a5aa-f62ce1bb4d17.png)

Gambar 14. **Hasil Grid View**

Pada Gambar 14, dapat dilihat bahwa parameter max_depth terbaik adalah 14 dan untuk parameter max_features terbaik adalah 2.


- Selanjutnya buatlah model prediksi dengan parameter terbaik dari hasil grid view pada Gammbar 14
```sh
rf = RandomForestClassifier(max_depth = 14, max_features = 2)

rf.fit(X_train, y_train)
```
Setelah model ini dijalankan hasilnya kan disimpan untuk tahap evaluasi.

## Evaluation
Metrik evaluasi yang digunakan pada proyek ini adalah Confusion Matrix. Confusion matrix juga sering disebut _error matrix_. Pada dasarnya confusion matrix memberikan informasi perbandingan hasil klasifikasi yang dilakukan oleh sistem (model) dengan hasil klasifikasi sebenarnya. Confusion matrix berbentuk tabel matriks yang menggambarkan kinerja model klasifikasi pada serangkaian data uji yang nilai sebenarnya diketahui. Pada Gambar 15 dibawah ini merupakan confusion matrix dengan 4 kombinasi nilai prediksi dan nilai aktual yang berbeda.

![matrix](https://user-images.githubusercontent.com/109077279/198958647-dc8651c3-7ef0-490a-8282-409f6569af8d.jpeg)

Gambar 15. **Confusion Matrix**

Pada Gambar 15, terlihat bahwa terdapat 4 istilah sebagai representasi hasil proses klasifikasi pada confusion matrix. Keempat istilah tersebut adalah True Positive (TP), True Negative (TN), False Positive (FP) dan False Negative (FN).

- True Positive (TP)
Merupakan data positif yang diprediksi benar. Contohnya, pasien menderita stroke (class 1) dan dari model yang dibuat memprediksi pasien tersebut menderita stroke (class 1).
- True Negative (TN)
Merupakan data negatif yang diprediksi benar. Contohnya, pasien tidak menderita stroke (class 0) dan dari model yang dibuat memprediksi pasien tersebut tidak menderita stroke (class 0).
- False Postive (FP) — Type I Error
Merupakan data negatif namun diprediksi sebagai data positif. Contohnya, pasien tidak menderita stroke (class 0) tetapi dari model yang telah memprediksi pasien tersebut menderita stroke (class 1).
- False Negative (FN) — Type II Error
Merupakan data positif namun diprediksi sebagai data negatif. Contohnya, pasien menderita stroke (class 1) tetapi dari model yang dibuat memprediksi pasien tersebut tidak menderita stroke (class 0).

Kita dapat menggunakan confusion matrix untuk menghitung berbagai _performance metrics_ untuk mengukur kinerja model yang telah dibuat. Beberapa _performance metrics_ ang sering digunakan: accuracy, precission, dan recall.

- Accuracy menggambarkan seberapa akurat model dapat mengklasifikasikan dengan benar.

$$ accuracy = {TP + TN   \over  TP + TN + FP + FN}  $$

- Precision menggambarkan tingkat keakuratan antara data yang diminta dengan hasil prediksi yang diberikan oleh model.

$$ precision = {TP \over  TP + FP}  $$

- Recall menggambarkan keberhasilan model dalam menemukan kembali sebuah informasi.
 
$$ precision = {TP \over  TP + FN}  $$

Berikut hasil evaluasi dari proyek ini :

- Akurasi

![akurasi](https://user-images.githubusercontent.com/109077279/198959570-d5a92553-15a4-4f73-a3b1-0fef9faa76c3.png)

Gambar 16. **Hasil Akurasi**

Pada Gambar 16, terlihat bahwa hasil akurasi dari proses train dan test sangat tinggi, yaitu 1.0


- Confusion Matrix

![matrix-mlt](https://user-images.githubusercontent.com/109077279/198960110-c71468e0-d4ab-43f5-b099-6105884d117f.png)

Gambar 17. **Hasil Confusion Matrix**

Pada Gambar 17, ternilai bahwa hasil dari confusion matrix memiliki nilai yang bagus. Nilai antara True dan False seimbang.


- Perfomance Matrix

![prediksi2](https://user-images.githubusercontent.com/109077279/198961494-9400098c-079b-458b-97b2-5f3d39371ba9.png)

Gambar 18. Hasil Performance Matrix

Pada Gambar 18, terlihat bahwa setiap performance matrix memiliki keluaran nilai yang baik.

F1 score: rata-rata antara precision dan recall.

AUC: hubungan antara  true-positive rate dan false-positive rate.

- Prediksi

Untuk memastikan apakah model dapat memprediksi dengan baik maka dilakukan prediksi terhadap model yang telah dilatih. Dapat dilihat pada Tabel 1, terdapat dua kolom utama, yaitu y_test dan y_pred. Terdapat sepuluh percobaan prediksi dan dapat dilihat bahwa hasil prediksi yang ditampilkan akurat.

Tabel 1. **Hasil Prediksi Model**
|      | y_test | y_pred |
|-----:|-------:|-------:|
| 6633 |      0 |      0 |
|  641 |      1 |      1 |
| 7785 |      0 |      0 |
|  988 |      1 |      1 |
| 4796 |      1 |      1 |
| 6299 |      0 |      0 |
| 5364 |      0 |      0 |
| 1677 |      1 |      1 |
| 5735 |      0 |      0 |
| 7001 |      0 |      0 |


## Conclusion

Dari hasil evaluasi dapat dilihat bahwa model algoritma Random Forest dengan kasus klasifikasi memiliki akurasi yang tinggi dan hasil prediksi yang akurat sehingga model ini dapat membantu dalam memprediksi stroke. Faktor-faktor yang mempengaruhi stroke di antaranya: jenis kelamin, usia, hipertensi, penyakit hati, kadar glukosa, berat badan, daerah tempat tinggal, status merokok, jenis pekerjaan, dan pernikahan. Di antara semua faktor tersebut, faktor yang paling menentukan resiko terkena stroke adalah faktor usia. 


## REFERENCES

[1] Kemenkes RI, P2PTM. (2017, Oktober 25). Germas Cegah Stroke. Diakses dari http://p2ptm.kemkes.go.id/tag/germas-cegah-stroke

[2] FEDESORIANO. 2021. "Stroke Prediction Dataset". Diakses dari https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
