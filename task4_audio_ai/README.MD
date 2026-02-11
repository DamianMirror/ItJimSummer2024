## Task 4 - Audio AI - Guitar Clustering and Melody Transcription
Introduction

If you love the sound of the guitar, this task will be a lot of fun! 
This task consists of two subtasks:

1. You will need to write code to transcribe guitar audio into notes.
2. You should cluster audio by the guitar that was used for recording
3. Guitar Audio Transcription

Description:

You will be provided with a folder containing 4 audio files with a guitar playing. 
Your task is to write Python code that transcribes this music into notes. 
The output should be printed as a list of tuples where each tuple consists of 
(note_name, start_time_in_seconds, end_time_in_seconds).

Requirements:

The main runnable file should be named audio_task_1.py.
The script should take the path of the input audio file as an argument.
The output should be a printed list of tuples.

Example:

```
Command to run the script: python audio_task_1.py path_to_audio_file.wav

Expected output [('A4', 0.001, 0.521), ('C#5', 0.5, 1.0), ('E5', 1.0, 1.5), ...]
```


Guitar Audio Clustering

Description:

In this task, you should use the same folder containing 4 audio files. Two different guitars were used to record two identical melodies. Your task is to write Python code that clusters these 4 audio files by the guitar that was used for the recording. Each cluster should contain two audio files recorded with the same guitar.


Requirements:

The main runnable file should be named audio_task_2.py.
The script should take the path to the folder containing the audio files as an argument.
The output should be a printed list of tuples where each tuple contains the names of the audio files in each cluster.

Example:

```
Command to run the script: python audio_task_2.py path_to_audio_folder

Expected output: [('guitar1_file1.wav', 'guitar1_file2.wav'), ('guitar2_file1.wav', 'guitar2_file2.wav')]
```

Deliverables:

python code with requirements.txt
a pdf report describing all experiments, and the approach used for solving the task