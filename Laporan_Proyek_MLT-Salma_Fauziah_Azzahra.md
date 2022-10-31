# Laporan Proyek Machine Learning - Salma Fauziah Azzahra
## Domain Proyek

Stroke adalah gejala-gejala defisit fungsi saraf yang diakibatkan oleh penyakit pembuluh darah otak, bukan oleh sebab yang lain (WHO). Gangguan fungsi syaraf pada stroke disebabkan oleh gangguan peredaran darah otak non traumatik. 

Menurut Organisasi Kesehatan Dunia (WHO) stroke adalah penyebab kematian ke-2 secara global, bertanggung jawab atas sekitar 11% dari total kematian. Data World Health Organization (WHO) tahun 2012 menunjukkan sekitar 31% dari 56,5 juta orang atau 17,7 juta orang di seluruh dunia meninggal akibat penyakit jantung dan pembuluh darah. Dari seluruh kematian akibat penyakit kardiovaskuler, sebesar 7,4 juta disebabkan oleh Penyakit Jantung Koroner, dan 6,7 juta disebabkan oleh stroke.

Oleh karena itu deteksi dini dirasa sangat penting untuk mencengah dan meminimalisir terjadinya stroke. Untuk membantu dalam mempermudah deteksi dini, maka dibuatlah proyek machine learning predictive analytics untuk tedeksi awal penyakit stroke berdasarkan faktor-faktor yang ada. Selain mempermudah, proyek ini juga mempercepat proses deteksi.

**Referensi:** 
- [Kementerian Kesehatan Republik Indonesia: Germas Cegah Stroke](http://p2ptm.kemkes.go.id/tag/germas-cegah-stroke)
- [Kaggle: Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

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
-  Melakukan hyperparameter tuning menggunakan grid search dan menentukan solusi permasalahan dengan klasifikasi. ALgoritma yang dipakai dalam proyek ini adalah Random Forest

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
**Fitur gender**

Pada fitur genre terdapat 2994 data "Female" (persentase 58.6%) dan 2115 data "Male" (persentase 41.4%)

![gender](https://github.com/salmafaa/image-for-mlt/blob/main/gender.png)

**Fitur ever_married**

Pada fitur ever_married terdapat 3353 data "Yes" (persentase 65.6%) dan 1757 data "No" (persentase 34.4%)

![ever_married](https://github.com/salmafaa/image-for-mlt/blob/main/ever_married.png)

**Fitur work_type**

Pada fitur work_type terdapat 2925 data "Private" (persentase 57.2%), 819 data "Self-employed" (persentase 16.0%), 687 data "children" (persentase 13.4%), 657 data "Govt_job" (persentase 12.9%), dan 22 data "Never_worked" (persentase 0.4%)

![work_type](https://github.com/salmafaa/image-for-mlt/blob/main/work_type.png)

**Fitur Residence_type**

Pada fitur Residence_type terdapat 2596 data "Urban" (persentase 50.8%) dan 2514 data "Rural" (persentase 49.2%)

![Residence_type](https://github.com/salmafaa/image-for-mlt/blob/main/Residence_type.png)

**Fitur smoking_status**

Pada fitur smoking status terdapat 1892 data "never smoked" (persentase 37%), 1544 data "Unknow" (persentase 30.2%), 855 data "formerly smoked" (persentase 17.3%), dan 789 data "smokes" (persentase 15.4%)

![smoking_status](https://github.com/salmafaa/image-for-mlt/blob/main/smoking_status.png)

**Analisis sebaran pada setiap fitur numerik**

![numerical features](https://github.com/salmafaa/image-for-mlt/blob/main/univariate%20analisis%20(numerical%20features).png)

Berfokus pada data target, yaitu stroke memiliki 4861 untuk 0 (tidak stroke) dan 249 data untuk 1 (stroke). Dapat dilihat bahwa data pada stroke merupakan data yang tidak seimbang sehingga akan mempengaruhi akurasi saat mendeteksi. Permasalahan ini akan diselesaikan pada tahap data preparation.

## Multivariate Analyst
Multivariate Analysis menunjukkan hubungan antara dua atau lebih fitur dalam data.

**Korelasi fitur gender dan stroke**

Terlihat bahwa jumlah pria dan wanita yang terkena stroke hampir sama

![genderxstroke](https://github.com/salmafaa/image-for-mlt/blob/main/gender%20x%20stroke.png)

**Korelasi fitur ever_married dan stroke**

Terlihat bahwa belum menikah mengurangi risiko stroke karena data setelah menikah memiliki jumlah penderita stroke lebih banyak

![ever_marriedxstroke](https://github.com/salmafaa/image-for-mlt/blob/main/ever_married%20x%20stroke.png)

**Korelasi fitur work_type dan stroke**

Terlihat bahwa pekerja pribadi lebih banyak terkena stroke

![work_typexstroke](https://github.com/salmafaa/image-for-mlt/blob/main/work_type%20x%20stroke.png)

**Korelasi fitur Residence_type dan stroke**

Terlihat bahwa Residence_type hampir memiliki jumlah data yang sama

![Residence_typexstroke](https://github.com/salmafaa/image-for-mlt/blob/main/Residence_type%20x%20stroke.png)

**Korelasi fitur smoking_status dan stroke**

Terlihat bahwa sampel tidak pernah merokok yang paling banyak mengalami stroke. Dan proporsi terkecil dari sampel penderita stroke adalah perokok tetapi orang yang sebelumnya merokok dan yang merokok (gabungan) menunjukkan tanda-tanda stroke jauh lebih banyak daripada orang yang tidak pernah merokok.

![smoking_statusxstroke](https://github.com/salmafaa/image-for-mlt/blob/main/smoking_status%20x%20stroke.png)

**Korelasi antara semua fitur numerik**

![korelasi fitur numerik](https://github.com/salmafaa/image-for-mlt/blob/main/Correlation%20matrix%20fitur%20numerical.png)

Berfokus pada korelasi stroke dengan fitur numerik lainnya, dapat dilihat bahwa yang memiliki korelasi dengan nilai besar adalah age.

**Korelasi antara fitur age, bmi, dan stroke**

![korelasi agexbmixstroke](https://github.com/salmafaa/image-for-mlt/blob/main/korelasi%20antara%20age%20x%20bmi%20x%20stroke.png)

Terlihat bahwa pasien dengan stroke berada pada rentang usia > 40 tahun dan rentang bmi di bawah 60.


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
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

Algoritma Penelitian ini melakukan pemodelan dengan algoritma Random Forest
- Random Forest Algoritma random forest adalah teknik dalam machine learning dengan metode ensemble. Teknik ini beroperasi dengan membangun banyak decision tree pada waktu pelatihan. Proyek ini menggunakan sklearn.ensemble RandomForestClassifier dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :

  - max_depth = Kedalaman maksimum setiap tree.
   - max_features = hyperparameter yang menentukan jumlah variable prediktor yang akan dipertimbangkan di setiap tree.

- Hyperparameter Tuning (Grid Search) Hyperparameter tuning adalah cara untuk mendapatkan parameter terbaik dari algoritma dalam membangun model. Salah satu teknik dalam hyperparameter tuning yang digunakan dalam proyek ini adalah grid search. Berikut adalah hasil dari Grid Search pada proyek ini :

![grid search](https://github.com/salmafaa/image-for-mlt/blob/main/gridsearch-final.png)

## Evaluation
Metrik evaluasi yang digunakan pada proyek ini adalah Confusion Matrix. Confusion matrix juga sering disebut error matrix. Pada dasarnya confusion matrix memberikan informasi perbandingan hasil klasifikasi yang dilakukan oleh sistem (model) dengan hasil klasifikasi sebenarnya. Confusion matrix berbentuk tabel matriks yang menggambarkan kinerja model klasifikasi pada serangkaian data uji yang nilai sebenarnya diketahui. Gambar dibawah ini merupakan confusion matrix dengan 4 kombinasi nilai prediksi dan nilai aktual yang berbeda.

![confunsion matrix](https://github.com/salmafaa/image-for-mlt/blob/main/matrix.jpeg)

Terdapat 4 istilah sebagai representasi hasil proses klasifikasi pada confusion matrix. Keempat istilah tersebut adalah True Positive (TP), True Negative (TN), False Positive (FP) dan False Negative (FN).

- True Positive (TP)
Merupakan data positif yang diprediksi benar. Contohnya, pasien menderita stroke (class 1) dan dari model yang dibuat memprediksi pasien tersebut menderita stroke (class 1).
- True Negative (TN)
Merupakan data negatif yang diprediksi benar. Contohnya, pasien tidak menderita stroke (class 0) dan dari model yang dibuat memprediksi pasien tersebut tidak menderita stroke (class 0).
- False Postive (FP) — Type I Error
Merupakan data negatif namun diprediksi sebagai data positif. Contohnya, pasien tidak menderita stroke (class 0) tetapi dari model yang telah memprediksi pasien tersebut menderita stroke (class 1).
- False Negative (FN) — Type II Error
Merupakan data positif namun diprediksi sebagai data negatif. Contohnya, pasien menderita stroke (class 1) tetapi dari model yang dibuat memprediksi pasien tersebut tidak menderita stroke (class 0).

Kita dapat menggunakan confusion matrix untuk menghitung berbagai performance metrics untuk mengukur kinerja model yang telah dibuat. Beberapa performance metrics ang sering digunakan: accuracy, precission, dan recall.
- Accuracy menggambarkan seberapa akurat model dapat mengklasifikasikan dengan benar.

![accuracy](https://github.com/salmafaa/image-for-mlt/blob/main/Persamaan-nilai-akurasi.png)

- Precision menggambarkan tingkat keakuratan antara data yang diminta dengan hasil prediksi yang diberikan oleh model.

![precision](https://github.com/salmafaa/image-for-mlt/blob/main/persamaan-nilai-precision.jpeg)

- Recall menggambarkan keberhasilan model dalam menemukan kembali sebuah informasi.
 
![recall](https://github.com/salmafaa/image-for-mlt/blob/main/persamaan-nilai-recall.jpeg)

Berikut hasil evaluasi dari proyek ini :
- Akurasi

![akurasi](https://github.com/salmafaa/image-for-mlt/blob/main/akurasi.png)

- Confusion Matrix

![confusion matrix](https://github.com/salmafaa/image-for-mlt/blob/main/matrix-mlt.png)

- Perfomance Matrix

![performance](https://github.com/salmafaa/image-for-mlt/blob/main/prediksi2.png)

F1 score: rata-rata harmonik presisi dan daya ingat.
AUC: hubungan antara  true-positive rate dan false-positive rate.

- Prediksi

![prediksi](https://github.com/salmafaa/image-for-mlt/blob/main/prediksi.png)

Dari hasil evaluasi dapat dilihat bahwa model algoritma Random Forest dengan kasus klasifikasi memiliki akurasi yang tinggi dan hasil prediksi yang tepat. 
