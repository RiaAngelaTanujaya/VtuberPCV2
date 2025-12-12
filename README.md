# Rabbit VTuber V10 - Real-time Motion Capture Avatar

Proyek ini adalah aplikasi VTuber (Virtual YouTuber) berbasis Python yang mengubah gerakan tubuh dan ekspresi wajah pengguna secara real-time menjadi animasi karakter kelinci lucu menggunakan computer vision.

## Fitur Utama

- **Motion Tracking Real-time**: Menangkap gerakan tubuh lengkap (pose, wajah, dan tangan)
- **Avatar Kelinci Ekspresif**: Karakter kelinci yang responsif dengan animasi wajah yang detail
- **Deteksi Ekspresi Wajah**: Mata berkedip, mulut membuka/menutup, dan pergerakan kepala
- **Tracking Tangan Detail**: Menampilkan gerakan jari-jari tangan secara akurat
- **Animasi Tubuh Lengkap**: Mencakup kepala, telinga, badan, lengan, dan kaki
- **Dual Display**: Menampilkan video asli dengan mesh tracking dan hasil avatar secara bersamaan

## Teknologi yang Digunakan

- **OpenCV (cv2)**: Untuk pengolahan video dan rendering grafis
- **MediaPipe**: Untuk deteksi pose tubuh, wajah, dan tangan
- **NumPy**: Untuk komputasi array dan transformasi geometris
- **Python 3.x**: Bahasa pemrograman utama

## Persyaratan Sistem

### Software
- Python 3.7 atau lebih baru
- Webcam yang terhubung ke komputer

### Library Python
```bash
pip install opencv-python
pip install mediapipe
pip install numpy
```

## Cara Menggunakan

1. **Instalasi Dependencies**
   ```bash
   pip install opencv-python mediapipe numpy
   ```

2. **Menjalankan Aplikasi**
   ```bash
   python rabbit_vtuber.py
   ```

3. **Penggunaan**
   - Posisikan diri di depan webcam
   - Pastikan tubuh bagian atas (kepala hingga kaki) terlihat oleh kamera
   - Aplikasi akan otomatis mendeteksi dan melacak gerakan Anda
   - Avatar kelinci akan mengikuti gerakan tubuh dan ekspresi wajah Anda

4. **Keluar dari Aplikasi**
   - Tekan tombol `ESC` untuk menutup aplikasi

## Desain Karakter

### Palet Warna
- **Bulu Kelinci**: Putih bersih (252, 252, 252)
- **Bayangan**: Abu-abu muda (200, 200, 205)
- **Perut**: Lavender muda (245, 235, 255)
- **Telinga Dalam**: Pink muda (255, 185, 210)
- **Hidung**: Pink (255, 150, 150)
- **Pipi**: Pink pucat (255, 210, 220)
- **Mata**: Hitam gelap (40, 40, 40)
- **Mulut**: Coklat tua (80, 40, 40)
- **Background**: Abu-abu gelap (40, 44, 52)

### Komponen Avatar
1. **Kepala**: Lingkaran dengan proporsi yang disesuaikan
2. **Telinga**: Oval panjang dengan rotasi mengikuti kemiringan kepala
3. **Mata**: Responsif terhadap kedipan dan arah pandangan
4. **Hidung**: Bentuk oval kecil di tengah wajah
5. **Mulut**: Animasi membuka/menutup dengan lidah saat mulut terbuka lebar
6. **Pipi**: Bulatan pink di kedua sisi wajah
7. **Badan**: Oval besar dengan area perut yang lebih terang
8. **Lengan**: Garis tebal dengan tangan yang menunjukkan detail jari
9. **Kaki**: Dari pinggul hingga pergelangan kaki dengan tapak kaki oval

## Fitur Teknis

### Sistem Tracking
- **Model Complexity**: Level 2 (akurasi tinggi)
- **Detection Confidence**: 0.5 (threshold deteksi)
- **Tracking Confidence**: 0.5 (threshold pelacakan)
- **Face Landmarks Refinement**: Aktif untuk detail wajah yang lebih baik

### Auto-Zoom dan Framing
- Sistem zoom otomatis yang menyesuaikan dengan jarak pengguna
- Padding dinamis untuk memastikan seluruh tubuh terlihat
- Skalasi proporsional untuk menjaga aspek rasio

### Deteksi Ekspresi
- **Kedipan Mata**: Berdasarkan rasio jarak vertikal kelopak mata
- **Bukaan Mulut**: Kalkulasi dinamis dengan batas maksimum
- **Kemiringan Kepala**: Rotasi telinga mengikuti sudut kepala

## Landmark yang Digunakan

### Face Landmarks (MediaPipe)
- Landmark 4: Hidung
- Landmark 33, 263: Mata kiri dan kanan
- Landmark 159, 386: Pusat mata
- Landmark 468, 473: Iris mata
- Landmark 13, 14: Bibir atas dan bawah
- Landmark 10, 152: Referensi wajah

### Pose Landmarks (MediaPipe)
- Landmark 11, 12: Bahu kiri dan kanan
- Landmark 13, 14: Siku kiri dan kanan
- Landmark 15, 16: Pergelangan tangan
- Landmark 23, 24: Pinggul kiri dan kanan
- Landmark 25, 26: Lutut kiri dan kanan
- Landmark 27, 28: Pergelangan kaki

### Hand Landmarks (MediaPipe)
- Landmark 0: Pergelangan tangan
- Landmark 4, 8, 12, 16, 20: Ujung jari
- Landmark 2, 5, 9, 13, 17: Buku jari

## Optimasi dan Performa

- **Visibility Threshold**: 0.5 untuk filtering landmark yang tidak terlihat
- **Anti-aliasing**: Menggunakan `cv2.LINE_AA` untuk garis yang halus
- **Layer-based Drawing**: Urutan gambar yang optimal (telinga → badan → kepala → wajah → tangan)
- **Bayangan**: Shadow layer untuk efek depth 3D

## Troubleshooting

### Kamera tidak terdeteksi
```python
# Ganti index kamera di baris:
cap = cv2.VideoCapture(0)  # Coba 1, 2, dst.
```

### Tracking tidak stabil
- Pastikan pencahayaan ruangan cukup
- Hindari background yang terlalu ramai
- Jaga jarak optimal dari kamera (1-2 meter)

### Performa lambat
- Kurangi `model_complexity` dari 2 ke 1 atau 0
- Turunkan resolusi webcam
- Tutup aplikasi lain yang menggunakan kamera

## Catatan Pengembangan

Versi ini (V10) merupakan versi final yang mencakup:
- Perbaikan stabilitas wajah (hidung, mulut, pipi tidak lagi mengalami rotasi manual berlebihan)
- Penambahan sistem kaki lengkap (paha, betis, tapak kaki)
- Optimasi koneksi lengan ke badan
- Peningkatan sistem auto-zoom dan framing

## Lisensi

Proyek ini dibuat untuk tujuan edukasi dan eksperimen. Silakan gunakan dan modifikasi sesuai kebutuhan.

## Kontribusi

Kontribusi, saran, dan perbaikan bug sangat diterima! Silakan buat issue atau pull request.

## Video Demonstrasi

.

---

**Dibuat menggunakan Python, OpenCV, dan MediaPipe**
