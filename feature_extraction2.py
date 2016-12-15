import glob
# The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell,
# although results are returned in arbitrary order.
#  No tilde expansion is done, but *, ?, and character ranges expressed with [] will be correctly matched.
import librosa
import numpy as np

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
#    stft = np.abs(librosa.stft(X))
    mfccs = librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=39,hop_length=int(sample_rate*0.01),n_fft=int(sample_rate*0.02)).T
#    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    print("mfccs :", mfccs,"\n",mfccs.shape)
#    print("chroma : ",chroma,"\n",chroma.shape)
    return mfccs
wav_files = []
for i in range(10):
    all_mfccs = []
    files = glob.glob('./%d/*.wav' % i)
    col = len(files)
    all_mfccs = np.ndarray(shape=[0,39],dtype=np.float32)
    print( all_mfccs.shape)
    print(files)
    for eachfile in files :
        print(eachfile)
        feature = extract_feature(eachfile)
        print(feature.shape)
        all_mfccs = np.append(all_mfccs,feature,axis=0)
        print(eachfile,"is done")
    print(all_mfccs.shape)
    np.savez('%d' %i , X=all_mfccs)


'''
def parse_audio_files(filenames):
    row - len(filenames)
'''
    
