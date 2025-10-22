#!/usr/bin/env python3
# spider_launcher.py
"""
Launcher interactivo (CLI guiado) + GUI mínima (Tkinter) para spider_ocr_pipeline_v2.py

Características:
- CLI guiado paso a paso para configurar opciones y lanzar el spider.
- GUI con selección de seeds file, parámetros, botones Start/Stop, y log en ventana.
- Crea un seeds.txt plantilla si no existe.
- Ejecuta el pipeline como subproceso y redirige su stdout/stderr en vivo.
- No fuerza instalación: usa el Python del sistema (ajusta 'python_cmd' si hace falta).

Uso:
    python spider_launcher.py
"""

import os
import sys
import argparse
import threading
import subprocess
import shlex
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Ajusta si tu python es 'python3' u otro
PYTHON_CMD = sys.executable  # utiliza el mismo intérprete

# Nombre del script pipeline que lanzaremos (ajusta si tu script tiene otro nombre)
PIPELINE_SCRIPT = "main.py"

# Plantilla seeds por defecto
SEEDS_TEMPLATE = """# seeds.txt: URLs semilla (una por línea)
# Ejemplos:
https://es.wikipedia.org/wiki/Historia_de_Espa%C3%B1a
# Añade aquí tus URLs...
"""


def ensure_seeds_template(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SEEDS_TEMPLATE, encoding="utf-8")
        print(f"Creado fichero de seeds de ejemplo en: {path}")


def build_command(pipeline_script: str, args: dict) -> list:
    """
    Construye lista de argumentos para subprocess a partir del dict 'args'.
    """
    cmd = [PYTHON_CMD, pipeline_script]

    # Mapeo simple: sólo añadimos opciones que tengan valor
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        elif v is None:
            continue
        elif isinstance(v, list):
            for item in v:
                cmd.append(f"--{k}")
                cmd.append(str(item))
        else:
            cmd.append(f"--{k}")
            cmd.append(str(v))
    return cmd


def stream_subprocess(cmd, text_callback=None, stop_event=None):
    """
    Ejecuta subproceso y stream stdout/stderr línea a línea. text_callback(line) para UI.
    stop_event es threading.Event para solicitar parada (termina proceso).
    """
    print("Ejecutando:", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)

    try:
        for line in proc.stdout:
            if stop_event and stop_event.is_set():
                proc.terminate()
                break
            if text_callback:
                text_callback(line)
            else:
                sys.stdout.write(line)
    except Exception as e:
        print("Error en streaming:", e)
    finally:
        proc.wait()
        return proc.returncode


# ---------------------------
# CLI guiado (console)
# ---------------------------
def interactive_cli():
    print("\n=== MiniLLM Spider Launcher — CLI guiado ===\n")

    default_seeds = Path("seeds.txt")
    ensure_seeds_template(default_seeds)

    seeds_path = input(f"Ruta a seeds file [{default_seeds}]: ").strip() or str(default_seeds)
    seeds_path = Path(seeds_path)
    if not seeds_path.exists():
        create = input(f"{seeds_path} no existe. Crear plantilla? [Y/n]: ").strip().lower() or "y"
        if create.startswith("y"):
            ensure_seeds_template(seeds_path)
        else:
            print("Abortando.")
            return

    outdir = input("Output directory [downloads]: ").strip() or "downloads"
    db = input("DB path [spider_meta.db]: ").strip() or "spider_meta.db"
    concurrency = input("HTTP concurrency (int) [6]: ").strip() or "6"
    max_depth = input("Max crawl depth (int) [2]: ").strip() or "2"
    max_pages = input("Max pages overall (int) [1000]: ").strip() or "1000"
    ocr_lang = input("OCR language (tessdata) [spa]: ").strip() or "spa"
    poppler_path = input("Poppler path (Windows only, optional) [leave empty]: ").strip() or None
    allowed_domains = input("Allowed domains (comma separated, optional) [leave empty]: ").strip()
    allowed = [d.strip() for d in allowed_domains.split(",")] if allowed_domains else None

    args = {
        "seeds": str(seeds_path),
        "outdir": outdir,
        "db": db,
        "concurrency": concurrency,
        "max-depth": max_depth,
        "max-pages": max_pages,
        "ocr-lang": ocr_lang,
        "politeness": 0.5,
    }
    if poppler_path:
        args["poppler-path"] = poppler_path
    if allowed:
        args["allowed-domains"] = allowed

    cmd = build_command(PIPELINE_SCRIPT, args)

    print("\nComando a ejecutar:")
    print(" ".join([shlex.quote(c) for c in cmd]))
    run = input("¿Ejecutar ahora? [Y/n]: ").strip().lower() or "y"
    if not run.startswith("y"):
        print("Aborta ejecución.")
        return

    stop_event = threading.Event()

    def print_cb(line):
        print(line, end="")

    thread = threading.Thread(target=stream_subprocess, args=(cmd, print_cb, stop_event), daemon=True)
    thread.start()

    try:
        while thread.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nSolicitud de parada recibida. Matando subproceso...")
        stop_event.set()
        thread.join()
    print("Proceso terminado.")


# ---------------------------
# GUI (Tkinter)
# ---------------------------
class SpiderGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MiniLLM Spider Launcher — GUI")
        self.geometry("900x700")
        self.proc_thread = None
        self.stop_event = threading.Event()

        # Variables UI
        self.var_seeds = tk.StringVar(value=str(Path("seeds.txt")))
        self.var_outdir = tk.StringVar(value="downloads")
        self.var_db = tk.StringVar(value="spider_meta.db")
        self.var_concurrency = tk.IntVar(value=6)
        self.var_max_depth = tk.IntVar(value=12)
        self.var_max_pages = tk.IntVar(value=10000)
        self.var_ocr_lang = tk.StringVar(value="spa")
        self.var_poppler = tk.StringVar(value="/poppler/Library/bin")
        self.var_allowed = tk.StringVar(value="")

        self._build()

    def _build(self):
        frm_top = ttk.Frame(self)
        frm_top.pack(side="top", fill="x", padx=8, pady=8)

        # seeds
        ttk.Label(frm_top, text="Seeds file:").grid(row=0, column=0, sticky="w")
        seeds_entry = ttk.Entry(frm_top, textvariable=self.var_seeds, width=70)
        seeds_entry.grid(row=0, column=1, sticky="w")
        ttk.Button(frm_top, text="Browse", command=self.browse_seeds).grid(row=0, column=2, padx=4)
        ttk.Button(frm_top, text="Create template", command=self.create_seeds_template).grid(row=0, column=3, padx=4)

        # outdir / db
        ttk.Label(frm_top, text="Outdir:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_outdir, width=40).grid(row=1, column=1, sticky="w")
        ttk.Label(frm_top, text="DB:").grid(row=1, column=2, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_db, width=30).grid(row=1, column=3, sticky="w")

        # concurrency / depth / pages
        ttk.Label(frm_top, text="Concurrency:").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_concurrency, width=10).grid(row=2, column=1, sticky="w")
        ttk.Label(frm_top, text="Max depth:").grid(row=2, column=2, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_max_depth, width=10).grid(row=2, column=3, sticky="w")
        ttk.Label(frm_top, text="Max pages:").grid(row=3, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_max_pages, width=10).grid(row=3, column=1, sticky="w")

        # OCR / poppler / allowed domains
        ttk.Label(frm_top, text="OCR lang:").grid(row=4, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_ocr_lang, width=10).grid(row=4, column=1, sticky="w")
        ttk.Label(frm_top, text="Poppler path (opt):").grid(row=4, column=2, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_poppler, width=30).grid(row=4, column=3, sticky="w")
        ttk.Label(frm_top, text="Allowed domains (csv):").grid(row=5, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.var_allowed, width=50).grid(row=5, column=1, columnspan=2, sticky="w")

        # Buttons
        btn_frame = ttk.Frame(frm_top)
        btn_frame.grid(row=6, column=0, columnspan=4, pady=8)
        ttk.Button(btn_frame, text="Start", command=self.start).pack(side="left", padx=8)
        ttk.Button(btn_frame, text="Stop", command=self.stop).pack(side="left", padx=8)
        ttk.Button(btn_frame, text="Open outdir", command=self.open_outdir).pack(side="left", padx=8)

        # Log area
        self.log = scrolledtext.ScrolledText(self, height=30, font=("Consolas", 10))
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

    def browse_seeds(self):
        fn = filedialog.askopenfilename(title="Selecciona seeds.txt", filetypes=[("Text files", "*.txt"), ("All", "*.*")])
        if fn:
            self.var_seeds.set(fn)

    def create_seeds_template(self):
        path = Path(self.var_seeds.get() or "seeds.txt")
        if not path.exists():
            ensure_seeds_template(path)
            messagebox.showinfo("Creado", f"Fichero de seeds creado en {path}")
        else:
            messagebox.showinfo("Existe", f"{path} ya existe.")

    def open_outdir(self):
        out = Path(self.var_outdir.get() or "downloads")
        out.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(out))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(out)])
            else:
                subprocess.Popen(["xdg-open", str(out)])
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start(self):
        if self.proc_thread and self.proc_thread.is_alive():
            messagebox.showwarning("Run", "Ya hay un proceso en ejecución")
            return
        seeds_path = Path(self.var_seeds.get())
        if not seeds_path.exists():
            res = messagebox.askyesno("Seeds no encontrado", f"{seeds_path} no existe. Crear plantilla?")
            if res:
                ensure_seeds_template(seeds_path)
            else:
                return

        args = {
            "seeds": str(seeds_path),
            "outdir": str(self.var_outdir.get()),
            "db": str(self.var_db.get()),
            "concurrency": str(self.var_concurrency.get()),
            "max-depth": str(self.var_max_depth.get()),
            "max-pages": str(self.var_max_pages.get()),
            "ocr-lang": str(self.var_ocr_lang.get()),
            "politeness": "0.5",
        }
        popp = self.var_poppler.get().strip()
        if popp:
            args["poppler-path"] = popp
        allowed = [d.strip() for d in self.var_allowed.get().split(",") if d.strip()]
        if allowed:
            args["allowed-domains"] = allowed

        cmd = build_command(PIPELINE_SCRIPT, args)

        # clear text
        self.log.delete("1.0", "end")
        self.stop_event.clear()
        self.proc_thread = threading.Thread(target=self._run_process_thread, args=(cmd,), daemon=True)
        self.proc_thread.start()

    def stop(self):
        if not self.proc_thread or not self.proc_thread.is_alive():
            messagebox.showinfo("Stop", "No hay proceso en ejecución")
            return
        self.stop_event.set()
        messagebox.showinfo("Stop", "Se ha solicitado parada; el proceso será terminado en breve.")

    def _append_log(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def _run_process_thread(self, cmd):
        def cb(line):
            self.after(0, self._append_log, line)
        rc = stream_subprocess(cmd, text_callback=cb, stop_event=self.stop_event)
        self.after(0, self._append_log, f"\nProceso finalizado con código {rc}\n")

# ---------------------------
# Main entry
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Launcher CLI/GUI para spider_ocr_pipeline_v2")
    parser.add_argument("--nogui", action="store_true", help="Lanzar sólo CLI (sin GUI)")
    parser.add_argument("--cli", action="store_true", help="Lanzar modo CLI guiado")
    args = parser.parse_args()

    if args.nogui or args.cli:
        interactive_cli()
    else:
        app = SpiderGUI()
        app.mainloop()


if __name__ == "__main__":
    main()
