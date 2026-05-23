import logging

logger = logging.getLogger(__name__)


POSITIVE_LEXICON = {
    "bagus", "baik", "mantap", "keren", "hebat", "luar biasa",
    "sempurna", "istimewa", "memuaskan", "terbaik", "unggul",

    "berhasil", "sukses", "berjalan", "lancar", "efektif",
    "efisien", "tepat", "benar", "merata",
    
    "setuju", "mendukung", "dukung", "apresiasi", "hargai",
    "memuji", "pujian",
    
    "bergizi", "sehat", "segar", "enak", "lezat", "bersih",
    "higienis", "layak",
    
    "senang", "gembira", "puas", "bahagia", "syukur",
    "semangat", "bangga", "harap", "harapan",
    
    "manfaat", "bermanfaat", "gratis", "bantuan", "peduli",
    "perhatian", "amanah", "jujur", "transparan", "adil",
    "mulia", "sejahtera", "maju",
    
    "terima kasih", "lanjut", "terus", "pertahankan",
}

# Kata-kata bermuatan NEGATIF
NEGATIVE_LEXICON = {
    
    "buruk", "jelek", "parah", "rusak", "kacau", "berantakan",
    "tidak masuk akal", "ngawur", "tidak layak", "salah", "keliru",
    
    "basi", "busuk", "kotor", "jorok", "bau", "tidak bersih",
    "mentah", "keracunan", "sakit", "berbahaya",
    
    "gagal", "gagal total", "sia-sia", "mubazir", "percuma",
    "buang-buang", "pemborosan", "boros", "inefisien",
    
    "korupsi", "bohong", "tipu", "penipuan", "pencitraan",
    "gimmick", "sandiwara", "hipokrit", "munafik", "serakah",
    "tamak",
    
    "marah", "benci", "muak", "jijik", "kecewa", "sedih",
    "takut", "khawatir", "menyedihkan", "memprihatinkan",
    "kasihan",
    
    "hentikan", "stop", "tutup", "bubarkan", "ganti",
    "turunkan", "tolak",
    
    "merugikan", "rugikan", "bahaya", "tidak adil",
    "diskriminasi", "tidak berguna", "tidak tepat",
    
    "lambat", "lemah", "rendah","membunuh", "merusak", "menghancurkan", "membahayakan","merampok", "mencuri", "menipu", "korup", "curang",
    ""
    
    "tidak", "tidak masuk", "tidak bagus",
    "babi", "anjing", "sialan", "bangsat","sampah","jahanam",
    "biadab","lol","goblok","bodoh","tolol","idiot","dungu","gila",
    "sinting","bego", "bajingan", "brengsek", "kampret", "kontol", 
    "memek", "perek", "banci", "bencong", "pelacur", "prostitusi", 
}


NEGATION_WORDS = {
    "tidak", "bukan", "jangan", "belum", "tanpa",
    "kurang", "jarang", "hampir tidak",
}


# KAMUS ASPEK

ASPECT_KEYWORDS = {
    "kualitas_makanan": {
        "basi", "busuk", "kotor", "jorok", "bau",
        "tidak layak", "tidak bergizi", "kurang bergizi",
        "gizi", "sehat", "enak", "tidak enak", "segar",
        "mentah", "matang", "porsi", "makanan", "menu",
        "makan", "bergizi", "lauk", "sayur", "buah",
        "keracunan", "higienis",
    },
    "distribusi": {
        "lambat", "terlambat", "tidak merata", "merata",
        "distribusi", "penyebaran", "tidak sampai",
        "tidak terima", "belum terima", "tepat waktu",
        "pengiriman", "logistik", "sampai",
    },
    "anggaran": {
        "anggaran", "uang", "biaya", "korupsi", "mubazir",
        "boros", "pemborosan", "hemat", "efisien", "inefisien",
        "dana", "subsidi", "gratis", "bayar", "iuran",
        "alokasi", "rupiah", "triliun",
    },
    "program": {
        "program", "kebijakan", "pemerintah", "presiden",
        "makan bergizi gratis", "lanjut", "hentikan", "stop",
        "tutup", "pencitraan", "gimmick", "gagal", "berhasil",
        "badan gizi nasional", "presiden prabowo",
        "mantan presiden jokowi",
    },
    "guru_sekolah": {
        "guru", "sekolah", "murid", "siswa", "anak",
        "pelajar", "pendidikan", "honor", "gaji",
        "guru honorer", "tenaga pendidik", "kepala sekolah",
        "tk", "sd", "smp", "sma",
    },
}


class SentimentLabeler:
  

    def get_sentiment(self, text: str) -> str:
        
        if not isinstance(text, str) or not text.strip():
            return "netral"

        words = text.split()
        pos_score = 0
        neg_score = 0
        i = 0

        while i < len(words):
            w = words[i]
            prev = words[i - 1] if i > 0 else ""

            
            # Bigram lebih spesifik → prioritas lebih tinggi
            bigram = f"{w} {words[i + 1]}" if i + 1 < len(words) else ""

            if bigram in NEGATIVE_LEXICON:
                neg_score += 2   # bobot bigram negatif lebih besar
                i += 2
                continue

            if bigram in POSITIVE_LEXICON:
                # Cek apakah ada negasi sebelum bigram ini
                if prev in NEGATION_WORDS:
                    neg_score += 2
                else:
                    pos_score += 2
                i += 2
                continue

           
            if w in NEGATIVE_LEXICON:
                neg_score += 1
            elif w in POSITIVE_LEXICON:
                # Negasi membalik makna kata positif
                if prev in NEGATION_WORDS:
                    neg_score += 1
                else:
                    pos_score += 1

            i += 1

        # Keputusan akhir 
        if pos_score > neg_score:
            return "positif"
        elif neg_score > pos_score:
            return "negatif"
        else:
            return "netral"

    def get_aspect(self, text: str) -> str:
        
        if not isinstance(text, str) or not text.strip():
            return "umum"

        word_set = set(text.split())
        scores   = {
            asp: len(word_set & kws)
            for asp, kws in ASPECT_KEYWORDS.items()
        }

        best_score = max(scores.values())
        if best_score == 0:
            return "umum"   # tidak ada kata kunci aspek yang cocok

        return max(scores, key=scores.get)

    def label_dataframe(self, df, text_col: str = "text_clean"):
        
        logger.info("Memulai pelabelan sentimen & aspek ...")
        df = df.copy()
        df["sentimen"] = df[text_col].apply(self.get_sentiment)
        df["aspek"]    = df[text_col].apply(self.get_aspect)

        # Log distribusi
        sent_dist = df["sentimen"].value_counts().to_dict()
        asp_dist  = df["aspek"].value_counts().to_dict()
        logger.info("Distribusi sentimen: %s", sent_dist)
        logger.info("Distribusi aspek: %s", asp_dist)

        return df
