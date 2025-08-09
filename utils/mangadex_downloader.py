# !git clone https://github.com/mansuf/mangadex-downloader.git
# %cd mangadex-downloader
# !python setup.py install # or "pip install ."




import os
import subprocess
import shutil

def mangadex_download(link):
    download_dir = "manga_downloader"
    shutil.rmtree(download_dir, ignore_errors=True)
    os.makedirs(download_dir, exist_ok=True)

    result = subprocess.run(["mangadex-dl", link], cwd=download_dir)
    if result.returncode != 0:
        raise RuntimeError("Download gagal. Periksa link atau mangadex-dl.")

    folders = [f for f in os.listdir(download_dir) if os.path.isdir(os.path.join(download_dir, f))]
    if not folders:
        return

    manga_path = os.path.join(download_dir, max(folders, key=lambda f: os.path.getctime(os.path.join(download_dir, f))))
    chapters = [os.path.join(manga_path, d) for d in os.listdir(manga_path) if os.path.isdir(os.path.join(manga_path, d))]

    dest = "folder_ekstrak"
    os.makedirs(dest, exist_ok=True)

    for ch in chapters:
        for f in os.listdir(ch):
            src = os.path.join(ch, f)
            dst = os.path.join(dest, f)
            if os.path.exists(dst):
                base, ext = os.path.splitext(f)
                i = 1
                while os.path.exists(dst):
                    dst = os.path.join(dest, f"{base}_{i}{ext}")
                    i += 1
            shutil.move(src, dst)

    shutil.rmtree(download_dir)
