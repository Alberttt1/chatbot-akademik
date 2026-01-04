import random
from flask import Flask, render_template, request
import os, random, pickle
import pickle
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

responses = {
  "salam": [
    "Halo! Ada yang bisa saya bantu hari ini?",
    "Hai! Silakan tanyakan seputar informasi akademik.",
    "Halo, saya siap membantu kamu.",
    "Selamat datang! Mau tanya apa?",
    "Hai, saya chatbot akademik kampus.",
    "Halo! Butuh bantuan terkait perkuliahan?",
    "Hai, silakan sampaikan pertanyaan kamu.",
    "Halo! Semoga harimu menyenangkan.",
    "Selamat datang di layanan informasi akademik.",
    "Halo! Saya siap menjawab pertanyaan kamu."
  ],

  "jadwal_kuliah": [
    "Jadwal kuliah dapat kamu lihat melalui portal akademik kampus.",
    "Silakan cek jadwal perkuliahan di sistem SIAKAD.",
    "Informasi jadwal mata kuliah tersedia di website resmi kampus.",
    "Kamu bisa melihat jadwal kuliah di akun mahasiswa masing-masing.",
    "Jadwal kuliah diumumkan melalui portal mahasiswa.",
    "Untuk detail jadwal, silakan login ke sistem akademik.",
    "Cek jadwal kuliah melalui menu perkuliahan di portal kampus.",
    "Jadwal perkuliahan dapat berubah, silakan pantau secara berkala.",
    "Informasi jadwal terbaru tersedia di sistem akademik.",
    "Silakan buka website akademik untuk melihat jadwal kuliah."
  ],

  "info_krs": [
    "Pengisian KRS biasanya dibuka pada awal semester melalui portal SIAKAD.",
    "Silakan login ke portal mahasiswa untuk melakukan pengisian KRS.",
    "Periode pengisian KRS diumumkan oleh bagian akademik.",
    "Untuk mengisi KRS, pastikan kamu sudah melakukan pembayaran UKT.",
    "Informasi lengkap mengenai KRS tersedia di website kampus.",
    "Pengambilan mata kuliah dilakukan melalui menu KRS di sistem akademik.",
    "Jika mengalami kendala KRS, hubungi bagian akademik.",
    "Batas waktu pengisian KRS dapat dilihat di portal mahasiswa.",
    "Pastikan KRS diisi sebelum batas waktu yang ditentukan.",
    "Pengisian KRS dilakukan secara online melalui sistem kampus."
  ],

  "info_ujian": [
    "Jadwal UTS dan UAS diumumkan melalui portal akademik.",
    "Silakan cek jadwal ujian di website resmi kampus.",
    "Informasi ujian dapat dilihat melalui menu akademik.",
    "Jadwal ujian setiap mata kuliah berbeda-beda.",
    "Pastikan kamu mengecek jadwal ujian secara berkala.",
    "Pengumuman jadwal ujian disampaikan oleh pihak kampus.",
    "Detail ujian tersedia di portal mahasiswa.",
    "Jika jadwal belum muncul, silakan hubungi bagian akademik.",
    "Jadwal ujian bisa berubah sewaktu-waktu.",
    "Cek informasi UTS dan UAS melalui sistem akademik."
  ],

  "info_nilai": [
    "Nilai diumumkan melalui portal mahasiswa.",
    "Silakan login ke sistem akademik untuk melihat nilai.",
    "Pengumuman nilai dilakukan oleh dosen melalui SIAKAD.",
    "Nilai mata kuliah dapat diakses setelah proses input selesai.",
    "Jika nilai belum muncul, silakan cek kembali beberapa hari kemudian.",
    "Informasi nilai tersedia di menu hasil studi.",
    "Nilai dapat dilihat pada akun mahasiswa masing-masing.",
    "Pengumuman nilai biasanya dilakukan setelah UAS selesai.",
    "Silakan hubungi dosen jika terdapat kesalahan nilai.",
    "Pastikan koneksi internet stabil saat membuka nilai."
  ],

  "terima_kasih": [
    "Sama-sama, semoga membantu.",
    "Dengan senang hati!",
    "Terima kasih kembali.",
    "Semoga informasi ini bermanfaat.",
    "Baik, silakan hubungi saya lagi jika perlu.",
    "Senang bisa membantu kamu.",
    "Semoga sukses selalu.",
    "Terima kasih sudah menggunakan chatbot ini.",
    "Silakan bertanya kembali kapan saja.",
    "Saya siap membantu kapan pun."
  ]
}


HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Chatbot Akademik</title>
<style>
body {
  background: linear-gradient(120deg,#667eea,#764ba2);
  font-family: 'Segoe UI', sans-serif;
}

.chat-container {
  width: 420px;
  margin: 80px auto;
  background: white;
  border-radius: 12px;
  box-shadow: 0 0 20px rgba(0,0,0,.3);
  padding: 20px;
}

.chat-container h2 {
  text-align:center;
  color:#444;
}

input[type=text] {
  width: 78%;
  padding: 10px;
  border-radius: 8px;
  border:1px solid #ccc;
}

button {
  padding: 10px 14px;
  border:none;
  background:#667eea;
  color:white;
  border-radius:8px;
  cursor:pointer;
}

.chat-box {
  margin-top:15px;
  background:#f5f5f5;
  padding:10px;
  border-radius:8px;
  min-height:50px;
}
</style>
</head>
<body>
<div class="chat-container">
<h2> Chatbot Akademik</h2>
<form method="post">
  <input type="text" name="message" placeholder="Tanyakan sesuatu..." required>
  <button>Kirim</button>
</form>
<div class="chat-box">
<b>Bot:</b> {{reply}}
</div>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def chat():
    reply = ""
    if request.method == "POST":
        msg = request.form["message"]
        clean = preprocess(msg)
        vec = vectorizer.transform([clean])
        proba = model.predict_proba(vec)[0]
        max_proba = max(proba)
        pred = model.classes_[proba.argmax()]

        if max_proba < 0.6:
            reply = "Maaf, saya belum yakin dengan pertanyaan kamu."
        elif pred in responses:
            reply = random.choice(responses[pred])
        else:
            reply = "Maaf, saya belum memahami pertanyaan tersebut."

        return reply

    return render_template("index.html")


if __name__=="__main__":
    app.run(debug=True)
