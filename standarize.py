# prepare_text_clean.py
# pip install ftfy
import re, unicodedata
from pathlib import Path
import ftfy  # arregla mojibake y cosas raras

def clean_text(s: str, lowercase=True, remove_control=True, normalize_unicode='NFKC'):
    # arregla mojibake y reemplaza caracteres invisibles comunes
    s = ftfy.fix_text(s)

    # normalización unicode (NFKC o NFC)
    if normalize_unicode:
        s = unicodedata.normalize(normalize_unicode, s)

    # reemplazos comunes (comillas “ ” — guiones — etc)
    s = s.replace('\u00A0', ' ')          # NBSP -> espacio
    s = s.replace('\u200B', '')           # zero-width space
    s = s.replace('\u2013', '-')          # en-dash
    s = s.replace('\u2014', '-')          # em-dash
    s = s.replace('\u2018', "'").replace('\u2019', "'")
    s = s.replace('\u201C', '"').replace('\u201D', '"')
    s = re.sub(r'\s+', ' ', s)            # many spaces -> one
    if lowercase:
        s = s.lower()
    if remove_control:
        s = ''.join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in ("\n","\t"))
    return s.strip()

if __name__ == '__main__':
    IN = Path('minillm.txt')
    OUT = Path('minillm_clean.txt')
    text = IN.read_text(encoding='utf-8-sig')
    cleaned = clean_text(text, lowercase=False)  # en español tal vez quieras mantener mayúsc/minúsc
    OUT.write_text(cleaned, encoding='utf-8')
    print(f"Texto limpio guardado en {OUT} (long {len(cleaned)} chars)")
