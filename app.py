"""
Mini Project — Conversational Analytics
Use Case C · Aset & Gangguan
Chatbot text-to-SQL: pertanyaan natural -> SQL -> PostgreSQL -> jawaban + grafik.

Jalankan lokal:  streamlit run app.py
Deploy: Streamlit Community Cloud (isi GEMINI_API_KEY & DB_URL di Secrets)
"""
import os
import re
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

st.set_page_config(page_title='Conversational Analytics — Aset & Gangguan',
                    page_icon='⚡', layout='centered')

# ── TODO 1 (setara) — Konfigurasi LLM ───────────────────────────────
# Di Colab: GEMINI_API_KEY diisi manual di variabel.
# Di Streamlit Cloud: isi lewat menu Settings -> Secrets, format:
#   GEMINI_API_KEY = "isi_key_disini"
#   DB_URL = "postgresql://user:password@host:5432/namadb"
GEMINI_API_KEY = st.secrets.get('GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY', ''))
DB_URL = st.secrets.get('DB_URL', os.environ.get('DB_URL', ''))
MODEL_NAME = 'gemini-2.5-flash'

if not GEMINI_API_KEY:
    st.error('GEMINI_API_KEY belum diisi. Tambahkan di Streamlit Secrets.')
    st.stop()
if not DB_URL:
    st.error('DB_URL belum diisi. Tambahkan di Streamlit Secrets.')
    st.stop()


def panggil_gemini(prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return resp.text


# ── Database ───────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


engine = get_engine()

# === [DISEDIAKAN] Skema sebagai teks untuk di-inject ke prompt ===
SCHEMA_STR = """assets(asset_id, nama, jenis, lokasi)
outages(outage_id, asset_id, mulai, selesai, durasi_menit, penyebab)

Relasi:
- outages.asset_id -> assets.asset_id
Catatan: kolom 'mulai' & 'selesai' bertipe TIMESTAMP.
         durasi_menit = lama gangguan dalam menit."""


# ── TODO 2 — System prompt (schema injection) ───────────────────────
def build_prompt(question: str) -> str:
    prompt = f"""Anda adalah ahli SQL PostgreSQL. Berikut skema database:

{SCHEMA_STR}

Tugas Anda: ubah pertanyaan berbahasa natural di bawah menjadi SATU query PostgreSQL SELECT.

Aturan ketat:
- Balas HANYA dengan query SQL, tanpa penjelasan, tanpa markdown, tanpa ```sql```.
- Hanya gunakan tabel & kolom yang ada di skema di atas.
- Gunakan JOIN antara assets dan outages bila pertanyaan butuh data dari kedua tabel.
- Jangan gunakan DROP, DELETE, UPDATE, INSERT, ALTER, atau perintah pengubah data lainnya.
- Kolom 'mulai' dan 'selesai' bertipe TIMESTAMP. Jika pertanyaan menyebut periode waktu relatif
  (mis. "bulan ini", "tahun ini"), gunakan date_trunc('month', mulai) = date_trunc('month', CURRENT_DATE)
  atau EXTRACT(...) langsung pada kolom tersebut tanpa perlu CAST tambahan.

Contoh:
Pertanyaan: Berapa total gangguan untuk setiap aset?
SQL: SELECT a.nama, COUNT(o.outage_id) AS jumlah_gangguan FROM assets a JOIN outages o ON a.asset_id = o.asset_id GROUP BY a.nama ORDER BY jumlah_gangguan DESC

Pertanyaan: Berapa jumlah gangguan bulan ini?
SQL: SELECT COUNT(*) AS jumlah_gangguan FROM outages WHERE date_trunc('month', mulai) = date_trunc('month', CURRENT_DATE)

Pertanyaan: {question}
SQL:"""
    return prompt


# ── TODO 3 — generate_sql(): panggil LLM, ambil SQL bersih ──────────
def generate_sql(question: str) -> str:
    prompt = build_prompt(question)
    resp_text = panggil_gemini(prompt)
    sql = resp_text.strip()
    if sql.startswith('```'):
        sql = re.sub(r'^```(?:sql)?', '', sql).strip()
        sql = sql.rstrip('`').strip()
    m = re.search(r'(select|with)\b.+', sql, re.I | re.S)
    if m:
        sql = m.group(0)
    return sql.rstrip(';').strip()


# ── TODO 4 — validate_sql(): guardrail sebelum eksekusi ─────────────
FORBIDDEN = ["drop", "delete", "update", "insert", "alter", "truncate", "create", "grant"]


def validate_sql(sql: str) -> bool:
    if not sql or not sql.strip():
        return False
    teks = sql.strip().rstrip(';').strip()
    low = teks.lower()
    if not low.startswith('select'):
        return False
    if ';' in teks:
        return False
    for kata in FORBIDDEN:
        if re.search(rf'\b{kata}\b', low):
            return False
    return True


# === [DISEDIAKAN] run_sql(): eksekusi SQL -> DataFrame ===
def run_sql(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


# ── TODO 5 — visualize(): pilih & buat grafik (versi Streamlit) ─────
TEAL = '#0E8388'


def visualize(df: pd.DataFrame):
    """
    Versi Streamlit dari visualize(): return fig matplotlib (atau None)
    alih-alih plt.show(), supaya bisa dirender lewat st.pyplot().
    """
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    if len(cols) != 2:
        return None

    kol_x, kol_y = cols[0], cols[1]
    if not pd.api.types.is_numeric_dtype(df[kol_y]):
        return None

    nama_x = kol_x.lower()
    is_waktu = any(k in nama_x for k in ['mulai', 'tanggal', 'bulan', 'periode', 'tahun', 'date'])

    fig, ax = plt.subplots(figsize=(7, 4))
    if is_waktu:
        df_sorted = df.sort_values(kol_x)
        ax.plot(df_sorted[kol_x].astype(str), df_sorted[kol_y], marker='o', color=TEAL)
    else:
        ax.bar(df[kol_x].astype(str), df[kol_y], color=TEAL)

    ax.set_xlabel(kol_x)
    ax.set_ylabel(kol_y)
    ax.set_title(f'{kol_y} per {kol_x}')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    return fig


# ── TODO 6 — ask(): rangkai pipeline (versi Streamlit) ───────────────
def ask(question: str):
    """
    Pipeline:
      1) sql = generate_sql(question)
      2) jika not validate_sql(sql): retry 1x; jika masih gagal -> fallback
      3) jalankan run_sql(sql) (try/except -> fallback bila error)
      4) return dict berisi sql, df, fig, dan pesan (untuk dirender UI)
    """
    sql = generate_sql(question)

    if not validate_sql(sql):
        sql = generate_sql(question)
        if not validate_sql(sql):
            return {'ok': False, 'pesan': 'Maaf, tidak bisa menyusun query yang aman untuk pertanyaan ini.'}

    try:
        df = run_sql(sql)
    except Exception as e:
        return {'ok': False, 'pesan': f'Query gagal dieksekusi: {e}', 'sql': sql}

    fig = visualize(df) if not df.empty else None
    return {'ok': True, 'sql': sql, 'df': df, 'fig': fig}


# ── UI Streamlit ──────────────────────────────────────────────────
st.title('⚡ Conversational Analytics — Aset & Gangguan')
st.caption('Tanya data aset & gangguan dalam bahasa biasa.')

with st.sidebar:
    st.subheader('Contoh pertanyaan')
    contoh_pertanyaan = [
        'Berapa jumlah gangguan per gardu/aset pada bulan ini?',
        'Berapa rata-rata durasi pemulihan per jenis aset?',
        'Apa penyebab gangguan yang paling sering terjadi?',
    ]
    for ex in contoh_pertanyaan:
        st.caption('• ' + ex)
    if st.button('🗑️ Bersihkan chat'):
        st.session_state.messages = []
        st.rerun()

if 'messages' not in st.session_state:
    st.session_state.messages = []


def _render(hasil):
    if not hasil['ok']:
        st.error(hasil['pesan'])
        if hasil.get('sql'):
            with st.expander('SQL yang gagal'):
                st.code(hasil['sql'], language='sql')
        return
    with st.expander('🔎 SQL'):
        st.code(hasil['sql'], language='sql')
    if hasil['df'].empty:
        st.info('Query berhasil, tapi tidak ada data yang cocok.')
    else:
        st.dataframe(hasil['df'], use_container_width=True)
        if hasil['fig'] is not None:
            st.pyplot(hasil['fig'])


for m in st.session_state.messages:
    with st.chat_message(m['role']):
        if m['role'] == 'user':
            st.markdown(m['content'])
        else:
            _render(m['hasil'])

q = st.chat_input('Tanya tentang data aset & gangguan…')
if q:
    st.session_state.messages.append({'role': 'user', 'content': q})
    with st.chat_message('user'):
        st.markdown(q)
    with st.chat_message('assistant'):
        with st.spinner('Memproses…'):
            hasil = ask(q)
        _render(hasil)
    st.session_state.messages.append({'role': 'assistant', 'hasil': hasil})
