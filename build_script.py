# build_script.py
import os
import shutil
import subprocess
import sys

def clean():
    """Remove build artifacts"""
    for item in ['build', 'cython_build', 'output']:
        if os.path.exists(item):
            shutil.rmtree(item)
    for f in os.listdir('.'):
        if f.endswith(('.c', '.pyd')):
            os.remove(f)

def build_cython():
    """Compile Python modules to C extensions"""
    os.makedirs('cython_build', exist_ok=True)
    subprocess.check_call([sys.executable, 'setup.py', 'build_ext', '--inplace'])
    
    # Move compiled files to cython_build
    for f in os.listdir('.'):
        if f.endswith('.pyd') or f.endswith('.so'):
            shutil.move(f, os.path.join('cython_build', f))

def build_installer():
    """Package with PyInstaller"""
    cmd = [
    'pyinstaller',
    'main.py',
    '--name', 'ISWITCH_iCapture',
    '--windowed',
    '--add-data', f'icon.png{os.pathsep}.',
    '--hidden-import', 'uvicorn.loops.auto',
    '--hidden-import', 'uvicorn.protocols.http.auto',
    '--hidden-import', 'uvicorn.protocols.websockets.auto',
    '--hidden-import', 'uvicorn.lifespan.on',
    '--hidden-import', 'pystray._win32',
    '--hidden-import', 'huggingface_hub',
    '--hidden-import', 'timm.models.layers',
    '--hidden-import', 'transformers.models.internvl',
    '--hidden-import', 'transformers.models.internlm',
    '--hidden-import', 'transformers.models.clip',
    '--hidden-import', 'timm.models.vision_transformer'
    ]
    subprocess.check_call(cmd)

if __name__ == '__main__':
    clean()
    build_cython()
    build_installer()
    print("\nBuild completed! Executable is in 'dist' directory")