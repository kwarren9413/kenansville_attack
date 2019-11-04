# Kenansville Attack

## Install Dependencies

`pip install numpy scipy scikit-learn ipython sympy nose librosa Pillow SpeechRecognition`

## Execute python script

Run the .py file using the following command:

`python kenan_attack.py --inputfile <audio_file_path> --outputfile <outputfile_name/path> --attack <attack_name>`

where:
audio_file_path - full path to the audio file
outputfile_name/path - full path to the outputfile *or* outfile name (if you want it in the same location as the input file)
attack - fft *or* ssa


