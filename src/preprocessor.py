import re
import logging
from config import SENTIMENT_PRESERVE

logger = logging.getLogger(__name__)


SLANG_DICT = {
    # ── Singkatan Program MBG ─────────────────────────────────
    "mbg"          : "makan bergizi gratis",
    "mgb"          : "makan bergizi gratis",
    "bgn"          : "badan gizi nasional",

    # ── Tokoh Politik ─────────────────────────────────────────
    "wowo"         : "prabowo",
    "prabowo"      : "presiden prabowo",
    "jokowi"       : "mantan presiden jokowi",
    "jk"           : "mantan presiden jokowi",
    "pak prabowo"  : "presiden prabowo",

    # ── Kata Negasi & Penegas ─────────────────────────────────
    "gak"          : "tidak",
    "ga"           : "tidak",
    "nggak"        : "tidak",
    "ngga"         : "tidak",
    "engga"        : "tidak",
    "enggak"       : "tidak",
    "gk"           : "tidak",
    "tdk"          : "tidak",
    "g"            : "tidak",
    "bkn"          : "bukan",

    # ── Kata Ganti Orang ──────────────────────────────────────
    "gue"          : "saya",
    "gw"           : "saya",
    "aq"           : "saya",
    "sy"           : "saya",
    "w"            : "saya",
    "lo"           : "kamu",
    "lu"           : "kamu",
    "elo"          : "kamu",
    "km"           : "kamu",

    # ── Kata Penjelas ─────────────────────────────────────────
    "yg"           : "yang",
    "yng"          : "yang",
    "dg"           : "dengan",
    "dgn"          : "dengan",
    "krn"          : "karena",
    "karna"        : "karena",
    "utk"          : "untuk",
    "tuk"          : "untuk",
    "buat"         : "untuk",
    "jg"           : "juga",
    "jga"          : "juga",
    "udh"          : "sudah",
    "udah"         : "sudah",
    "sdh"          : "sudah",
    "dah"          : "sudah",
    "blm"          : "belum",
    "belom"        : "belum",
    "bgt"          : "sangat",
    "bngt"         : "sangat",
    "banget"       : "sangat",
    "kyk"          : "seperti",
    "kek"          : "seperti",
    "kayak"        : "seperti",
    "emang"        : "memang",
    "emg"          : "memang",
    "mmg"          : "memang",
    "bener"        : "benar",
    "bnr"          : "benar",
    "gimana"       : "bagaimana",
    "gmn"          : "bagaimana",
    "gini"         : "begini",
    "gitu"         : "begitu",
    "aja"          : "saja",
    "aj"           : "saja",
    "doang"        : "saja",
    "sm"           : "sama",
    "ama"          : "sama",
    "tp"           : "tetapi",
    "tpi"          : "tetapi",
    "krg"          : "kurang",
    "krang"        : "kurang",
    "byk"          : "banyak",
    "hrs"          : "harus",
    "msh"          : "masih",
    "sdg"          : "sedang",
    "tlg"          : "tolong",
    "plis"         : "tolong",
    "pliss"        : "tolong",
    "skrg"         : "sekarang",
    "skrng"        : "sekarang",
    "klo"          : "kalau",
    "klu"          : "kalau",
    "kalo"         : "kalau",
    "jd"           : "jadi",
    "jdi"          : "jadi",
    "td"           : "tadi",
    "pd"           : "pada",
    "ntar"         : "nanti",
    "btw"          : "ngomong-ngomong",
    "cuma"         : "hanya",
    "mending"      : "lebih baik",
    "makin"        : "semakin",
    "bakal"        : "akan",
    "liat"         : "lihat",
    "bilang"       : "mengatakan",
    "ngomong"      : "mengatakan",
    "dikasih"      : "diberikan",
    "dikasi"       : "diberikan",
    "ngasih"       : "memberikan",
    "pake"         : "menggunakan",
    "dipake"       : "digunakan",
    "ngerti"       : "mengerti",
    "ngerasa"      : "merasa",
    "ngeliat"      : "melihat",
    "ngedenger"    : "mendengar",
    "mikir"        : "berpikir",
    "keliatan"     : "terlihat",
    "kelihatan"    : "terlihat",
    "nntn"         : "nonton",
    "asli"         : "sungguh",
    "beneran"      : "sungguh",
    "kayaknya"     : "sepertinya",
    "kayanya"      : "sepertinya",

    # ── Kata Nilai / Sentimen Kolokial ────────────────────────
    "parah"        : "buruk",
    "koplak"       : "tidak masuk akal",
    "kasian"       : "kasihan",
    "anjir"        : "",      # kata makian dihapus
    "anjing"       : "",      # kata makian dihapus
    "njir"         : "",      # kata makian dihapus

    # ── Filler / Partikel (dihapus) ───────────────────────────
    "sih"          : "",
    "dong"         : "",
    "deh"          : "",
    "lah"          : "",
    "noh"          : "",
    "wkwk"         : "",
    "wkwkwk"       : "",
    "wkwkwkwk"     : "",
    "haha"         : "",
    "hehe"         : "",
    "xixi"         : "",
}

# ─────────────────────────────────────────────────────────────
# KELAS PREPROCESSOR UTAMA
# ─────────────────────────────────────────────────────────────
class IndonesianPreprocessor:
    def __init__(self):
        """
        Inisialisasi Stemmer dan StopWordRemover dari PySastrawi.
        Dilakukan sekali saat objek dibuat untuk efisiensi memori.
        """
        # ── Impor Sastrawi (hanya di sini agar error lebih jelas) ──
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
                StopWordRemoverFactory,
            )
        except ImportError:
            raise ImportError(
                "PySastrawi belum terinstal. Jalankan: pip install PySastrawi"
            )

        # Inisialisasi stemmer Sastrawi
        self.stemmer = StemmerFactory().create_stemmer()

        # Bangun set stopword gabungan
        sw_factory   = StopWordRemoverFactory()
        base_sw      = set(sw_factory.get_stop_words())

        # Stopword tambahan yang tidak bermakna dalam analisis sentimen
        extra_sw = {
            "yuk", "iya", "ok", "oke", "ya", "yo", "yap", "yep",
            "hmm", "hm", "eh", "ah", "oh", "ih", "aduh", "walah",
            "https", "http", "www", "com", "id", "co", "nbsp",
            "amp", "gt", "lt", "quot",
        }

        # Gabungkan & buang kata-kata yang membawa makna sentimen
        self.stopwords = (base_sw | extra_sw) - SENTIMENT_PRESERVE
        logger.info(
            "Preprocessor siap. Stopword: %d kata, Stemmer: Sastrawi",
            len(self.stopwords),
        )

    #  Metode internal 
    @staticmethod
    def _clean_noise(text: str) -> str:
        """Hapus noise: mention, hashtag, URL, email, angka, tanda baca."""
        text = text.lower()
        text = re.sub(r"@\w+", "", text)           # hapus mention
        text = re.sub(r"#\w+", "", text)           # hapus hashtag
        text = re.sub(r"http\S+|www\.\S+", "", text)  # hapus URL
        text = re.sub(r"\S+@\S+", "", text)        # hapus email
        text = text.encode("ascii", "ignore").decode("ascii")  # hapus emoji/unicode
        text = re.sub(r"\d+", "", text)            # hapus angka
        text = re.sub(r"[^\w\s]", " ", text)       # hapus tanda baca
        text = re.sub(r"\s+", " ", text).strip()   # normalisasi spasi
        return text

    @staticmethod
    def _normalize_repeated(text: str) -> str:
        
        return re.sub(r"(.)\1{2,}", r"\1\1", text)

    @staticmethod
    def _normalize_slang(text: str) -> str:
        
        tokens = text.split()
        result = [SLANG_DICT.get(t, t) for t in tokens]
        # Hapus token kosong (kata yang dikosongkan di kamus)
        return " ".join(t for t in result if t)

    def _remove_stopwords(self, tokens: list) -> list:
        
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]

    def _stem(self, tokens: list) -> list:
        
        return [self.stemmer.stem(t) for t in tokens]

    # ── API Publik ────────────────────────────────────────────
    def transform(self, text: str) -> str:
       
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self._clean_noise(text)
        text = self._normalize_repeated(text)
        text = self._normalize_slang(text)

        tokens = text.split()
        tokens = self._remove_stopwords(tokens)
        tokens = self._stem(tokens)

        return " ".join(t for t in tokens if t)

    def transform_batch(self, texts) -> list:
       
        result = []
        total  = len(texts)
        for i, t in enumerate(texts):
            result.append(self.transform(t))
            # Tampilkan progress setiap 1000 data
            if (i + 1) % 1000 == 0:
                logger.info("Preprocessing: %d/%d selesai", i + 1, total)
        logger.info("Preprocessing selesai: %d teks diproses", total)
        return result
