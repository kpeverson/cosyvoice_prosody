# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
_COSYVOICE_ROOT=/mmfs1/gscratch/tial/kpever/workspace/CosyVoice
export PYTHONPATH=${_COSYVOICE_ROOT}:${_COSYVOICE_ROOT}/third_party/Matcha-TTS:$PYTHONPATH

export PATH=/mmfs1/sw/cuda/12.9.1/bin:$PATH
export LD_LIBRARY_PATH=/mmfs1/sw/cuda/12.9.1/lib64:$LD_LIBRARY_PATH