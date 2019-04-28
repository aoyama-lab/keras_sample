import os
import sys
import subprocess
import struct
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.externals import joblib

# メルケプストラム次数
# 実際はパワー項を追加して26次元ベクトルになる
m = 25

def extract_mcep(wav_file, mcep_file, ascii=False):
    cmd = "bcut +s -s 22 %s | x2x +sf | frame -l 400 -p 80 | window -l 400 -L 512 | mcep -l 512 -m %d -a 0.42 | x2x +fa%d > %s" % (wav_file, m, m + 1, mcep_file)
    subprocess.call(cmd, shell=True)

def extract_pitch(wav_file, pitch_file):
    cmd = "bcut +s -s 22 %s | x2x +sf | pitch -a 1 -s 16 -p 80 > %s" % (wav_file, pitch_file)
    subprocess.call(cmd, shell=True)

def synthesis(pitch_file, mcep_file, wav_file):
    cmd = "excite -p 80 %s | mlsadf -m %d -a 0.42 -p 80 %s | clip -y -32000 32000 | x2x +fs > temp.raw" % (pitch_file, m, mcep_file)
    print(cmd)
    subprocess.call(cmd, shell=True)

    cmd = "sox -e signed-integer -c 1 -b 16 -r 16000 temp.raw %s" % (wav_file)
    print(cmd)
    subprocess.call(cmd, shell=True)

    os.remove("temp.raw")

if len(sys.argv) != 4:
    print("usage: python convert_voice.py <wav_path> <converted_mcep_path> <converted_wav_path>")
    exit()

source_wav_file = sys.argv[1]
converted_wav_file = sys.argv[3]

print("extract pitch ...")
source_pitch_file = "source.pitch"
extract_pitch(source_wav_file, source_pitch_file)

source_mcep = np.loadtxt(sys.argv[2]) # 変換後のmcepを読み込む
source_mcep_file = "temp.mcep"
fp = open(source_mcep_file, "wb")

for t in range(len(source_mcep)):
    x_t = source_mcep[t]
    fp.write(struct.pack('f' * (m + 1), * x_t))

fp.close()

# 変換元のピッチと変換したメルケプストラムから再合成
print("synthesis ...")
synthesis(source_pitch_file, source_mcep_file, converted_wav_file)

# 一時ファイルを削除
os.remove(source_mcep_file)
os.remove(source_pitch_file)