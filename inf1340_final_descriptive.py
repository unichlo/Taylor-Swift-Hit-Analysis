# -*- coding: utf-8 -*-
"""INF1340 Final Descriptive.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pR22ePzwLHp_5ry1Fm89Cwv6jeTyYwuE
"""

# Names: Danika Mariam (1004014880), Chloe Li(1010033321), Rosa Lee (1005089761)
# Class: INF1340 LEC0101
# Final Project
# Date: Dec 1, 2023

# DESCRIPTIVE ANALYTICS

# import the drive
from google.colab import drive, files
drive.mount("/drive", force_remount=True)

# import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

songs = pd.read_csv('/drive/MyDrive/taylor_swift_spotify.csv')
output_file = open("output_descriptive.txt", "w")
output_plots = PdfPages("plots_descriptive.pdf")

# let's look at how many non-null values are in the dataset
songs.info()

# list out album names
def print_album_names():
  album_names = songs["album"].unique()
  num_albums = len(album_names)
  print('Taylor Swift has released', num_albums,  'albums: \n')
  output_file.write('Taylor Swift has released' + str(num_albums)+  'albums: \n')
  for album in album_names:
    print(album)
    output_file.write(str(album))

# some important average scores, (danceability, energy, loudness, speechiness,
# acousticness, instrumentalness, liveness, valence, temp, time_signature, duration_ms)
def print_avg_scores():
  average_scores = songs.mean(numeric_only=True)
  print("Mean of Numeric Columns:")
  print(average_scores[2:].to_frame())
  print()
  output_file.write("Mean of Numeric Columns: \n")
  output_file.write(str(average_scores[2:].to_frame()))
  output_file.write('\n')

#printing the sd of the code
def print_std():
  std_deviation = songs.std(numeric_only=True)
  print("Standard Deviation of Numeric Columns:")
  print(std_deviation[2:].to_frame())
  print()
  output_file.write("Standard Deviation of Numeric Columns:")
  output_file.write(str(std_deviation[2:].to_frame()))
  output_file.write('\n')

# report the max and min of each numeric score
def print_max_min():
  max_danceability = songs.loc[songs["danceability"] == songs['danceability'].max(), "name"].to_list()
  max_danceability = list(set(max_danceability))
  print("Taylor's most danceable song(s):", max_danceability)
  output_file.write("Taylor's most danceable song(s):"+ str(max_danceability)+ "\n")
  min_danceability = songs.loc[songs["danceability"] == songs['danceability'].min(), "name"].to_list()
  min_danceability = list(set(min_danceability))
  print("Taylor's least danceable song(s):", min_danceability)
  output_file.write("Taylor's least danceable song(s):" + str(min_danceability) + "\n")
  print()

  max_energy = songs.loc[songs["energy"] == songs['energy'].max(), "name"].to_list()
  max_energy = list(set(max_energy))
  print("Taylor's most energetic song(s):", max_energy)
  output_file.write("Taylor's most energetic song(s):" + str(max_energy)+ "\n")
  min_energy = songs.loc[songs["energy"] == songs['energy'].min(), "name"].to_list()
  min_energy = list(set(min_energy))
  print("Taylor's least energetic song(s):", min_energy)
  output_file.write("Taylor's least energetic song(s):" + str(min_energy)+ "\n")
  print()


  max_loudness = songs.loc[songs["loudness"] == songs['loudness'].max(), "name"].to_list()
  max_loudness = list(set(max_loudness))
  print("Taylor's most loud song(s):", max_loudness)
  output_file.write("Taylor's most loud song(s):"+ str(max_loudness)+ "\n")
  min_loudness = songs.loc[songs["loudness"] == songs['loudness'].min(), "name"].to_list()
  min_loudness = list(set(min_loudness))
  print("Taylor's least loud song(s):", min_loudness)
  output_file.write("Taylor's least loud song(s):" + str(min_loudness)+ "\n")
  print()

  max_speech = songs.loc[songs["speechiness"] == songs['speechiness'].max(), "name"].to_list()
  max_speech = list(set(max_speech))
  print("Taylor's most wordy song(s):", max_speech)
  output_file.write("Taylor's most wordy song(s):" + str(max_speech)+ "\n")
  min_speech = songs.loc[songs["speechiness"] == songs['speechiness'].min(), "name"].to_list()
  min_speech = list(set(min_speech))
  print("Taylor's least wordy song(s):", min_speech)
  output_file.write("Taylor's least wordy song(s):" + str(min_speech)+ "\n")
  print()

  max_acoustic = songs.loc[songs["acousticness"] == songs['acousticness'].max(), "name"].to_list()
  max_acoustic = list(set(max_acoustic))
  print("Taylor's most acoustic song(s):", max_acoustic)
  output_file.write("Taylor's most acoustic song(s):" + str(max_acoustic)+ "\n")
  min_acoustic = songs.loc[songs["acousticness"] == songs['acousticness'].min(), "name"].to_list()
  min_acoustic = list(set(min_acoustic))
  print("Taylor's least acoustic song(s):", min_acoustic)
  output_file.write("Taylor's least acoustic song(s):" + str(min_acoustic)+ "\n")
  print()

  max_instrumental = songs.loc[songs["instrumentalness"] == songs['instrumentalness'].max(), "name"].to_list()
  max_instrumental = list(set(max_instrumental))
  print("Taylor's most instrumental song(s):", max_instrumental)
  output_file.write("Taylor's most instrumental song(s):" + str(max_instrumental)+ "\n")
  min_instrumental = songs.loc[songs["instrumentalness"] == songs['instrumentalness'].min(), "name"].to_list()
  min_instrumental = list(set(min_instrumental))
  print("Taylor's least instrumental song(s):", min_instrumental)
  output_file.write("Taylor's least instrumental song(s):" + str(min_instrumental)+ "\n")
  print()

  max_liveness = songs.loc[songs["liveness"] == songs['liveness'].max(), "name"].to_list()
  max_liveness = list(set(max_liveness))
  print("Taylor's most live song(s):", max_liveness)
  output_file.write("Taylor's most live song(s):" + str(max_liveness)+ "\n")
  min_liveness = songs.loc[songs["liveness"] == songs['liveness'].min(), "name"].to_list()
  min_liveness = list(set(min_liveness))
  print("Taylor's least live song(s):", min_liveness)
  output_file.write("Taylor's least live song(s):" + str(min_liveness)+ "\n")
  print()

  max_valence = songs.loc[songs["valence"] == songs['valence'].max(), "name"].to_list()
  max_valence = list(set(max_valence))
  print("Taylor's happiest sounding song(s):", max_valence)
  output_file.write("Taylor's happiest sounding song(s):" + str(max_valence)+ "\n")
  min_valence = songs.loc[songs["valence"] == songs['valence'].min(), "name"].to_list()
  min_valence = list(set(min_valence))
  print("Taylor's saddest sounding song(s):", min_valence)
  output_file.write("Taylor's saddest sounding song(s):" + str(min_valence)+ "\n")
  print()

  max_popularity = songs.loc[songs["popularity"] == songs['popularity'].max(), "name"].to_list()
  max_popularity = list(set(max_popularity))
  print("Taylor's most popular song(s):", max_popularity)
  output_file.write("Taylor's most popular song(s):" + str(max_popularity)+ "\n")
  min_popularity = songs.loc[songs["popularity"] == songs['popularity'].min(), "name"].to_list()
  min_popularity = list(set(min_popularity))
  print("Taylor's least popular song(s):", min_popularity)
  output_file.write("Taylor's least popular song(s):" + str(min_popularity)+ "\n")
  print()

  max_tempo = songs.loc[songs["tempo"] == songs['tempo'].max(), "name"].to_list()
  max_tempo = list(set(max_tempo))
  print("Taylor's fastest song(s):", max_tempo)
  output_file.write("Taylor's fastest song(s):" + str(max_tempo)+ "\n")
  min_tempo = songs.loc[songs["tempo"] == songs['tempo'].min(), "name"].to_list()
  min_tempo = list(set(min_tempo))
  print("Taylor's slowest song(s):", min_tempo)
  output_file.write("Taylor's slowest song(s):" + str(min_tempo)+ "\n")
  print()

  max_duration = songs.loc[songs["duration_ms"] == songs['duration_ms'].max(), "name"].to_list()
  max_duration = list(set(max_duration))
  print("Taylor's longest song(s):", max_duration)
  output_file.write("Taylor's longest song(s):" + str(max_duration)+ "\n")
  min_duration = songs.loc[songs["duration_ms"] == songs['duration_ms'].min(), "name"].to_list()
  min_duration = list(set(min_duration))
  print("Taylor's shortest song(s):", min_duration)
  output_file.write("Taylor's shortest song(s):" + str(min_duration)+ "\n")
  print()

# make histogram for popularity
def hist_pop():
  plt.figure()
  sns.histplot(songs["popularity"],bins='auto')
  plt.savefig(output_plots, format='pdf')
  plt.show()

# make histogram for danceability
def hist_dance():
  plt.figure()
  sns.histplot(songs["danceability"],bins='auto')
  plt.savefig(output_plots, format='pdf')
  plt.show()

# make histogram for energy
def hist_energy():
  plt.figure()
  sns.histplot(songs["energy"],bins='auto')
  plt.savefig(output_plots, format='pdf')
  plt.show()

# make histogram for loudness
def hist_loud():
  plt.figure()
  sns.histplot(songs["loudness"],bins='auto')
  plt.savefig(output_plots, format='pdf')
  plt.show()

# make histogram for speechiness
def hist_speech():
  plt.figure()
  sns.histplot(songs["speechiness"],bins='auto')
  plt.savefig(output_plots, format='pdf')
  plt.show()

# make histogram for acousticness
def hist_acoustic():
  plt.figure()
  sns.histplot(songs["acousticness"],bins='auto')
  plt.savefig(output_plots, format='pdf')
  plt.show()

# make histogram for liveness
def hist_liveness():
  plt.figure()
  sns.histplot(songs["liveness"],bins='auto')
  plt.savefig(output_plots, format='pdf')
  plt.show()

# make histogram for valence
def hist_valence():
  plt.figure()
  sns.histplot(songs["valence"],bins='auto')
  plt.savefig(output_plots, format='pdf')
  plt.show()

# make histogram for tempo
def hist_tempo():
  plt.figure()
  sns.histplot(songs["tempo"],bins='auto')
  plt.savefig(output_plots, format='pdf')
  plt.show()

# make histogram for duration
def hist_duration():
  plt.figure()
  sns.histplot(songs["duration_ms"],bins='auto')
  plt.savefig(output_plots, format='pdf')
  plt.show()

# main program
def main():
  print_album_names()
  print_avg_scores()
  print_std()
  print_max_min()
  hist_pop()
  hist_dance()
  hist_energy()
  hist_loud()
  hist_speech()
  hist_acoustic()
  hist_liveness()
  hist_valence()
  hist_tempo()
  hist_duration()
  output_file.close()
  output_plots.close()

main()
files.download('output_descriptive.txt')
files.download('plots_descriptive.pdf')