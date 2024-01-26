# Taylor-Swift-Hit-Analysis

Taylor Swift’s Spotify Songs – Descriptive, Predictive, Diagnostic Analysis
The Final Descriptive script outputs information about Taylor Swift's songs, providing insights into the characteristics of her music. The data file had a total of 528 rows and 17 columns. The script generates a histogram plot for song popularity using the Seaborn library. 

The below information was identified:
•	Overview of non-null values in the dataset.
•	List of Taylor Swift's albums.
•	Average scores and standard deviation of numeric columns.
•	Identification of songs with the maximum and minimum values for various attributes.
•	Histogram plot of song popularity.

The Final Diagnostic script specifically explores the correlation between various song characteristics in Taylor Swift's dataset. We leverage correlation matrix and heatmap visualization to analyze the relationships among acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence, duration, and popularity.
•	Data loading and correlation matrix
•	Heatmap visualization
•	Scatterplots

The Final Predictive script conducts predictive analytics on Taylor Swift's songs using machine learning model and implements both Linear Regression and Logistic Regression models to predict the popularity of songs.
•	Linear Regression:
o	Data Preparation
o	Model Training
o	Prediction and Evaluation
•	Logistic Regression:
o	Data Preparation
o	Data Validation
o	Model Training and Evalution
o	Model Coefficients and Feature Importance
o	Probability Plots Motivation:

The motivation for this data analysis is I'm big Taylor Swift Fans and wanted to summarize the data as we are aware she has a lot of different types of songs that are popular.
 
Build Status:
There are currently no bugs for all three scripts. Code Style:
I used Google Colab with pandas incorporating functions, so please install pandas before running the file.

Running the Script:
•	Open the script in a Google Colab environment.
•	Run each code cell sequentially. 
 
Importing the data into panda data frame
import pandas as pd

The following code imports the csv file from Google
from google.colab import drive, files
drive.mount("/drive", force_remount=True)

Descriptive stats:
Importing panda matplot.lib.pyplot, and seaborn packages.
mport pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

songs = pd.read_csv('/drive/MyDrive/taylor_swift_spotify.csv')
output_file = open("output_descriptive.txt", "w")
output_plots = PdfPages("plots_descriptive.pdf")

Identifying the number of non-null values that are in the data.
songs.info()

The print_album_names() function used to print the list of Taylor Swift album names.
def print_album_names():
  album_names = songs["album"].unique()
  num_albums = len(album_names)
  print('Taylor Swift has released', num_albums,  'albums: \n')
  output_file.write('Taylor Swift has released' + str(num_albums)+  'albums: \n')
  for album in album_names:
    print(album)
    output_file.write(str(album))


The print_avg_scores() and print_std() functions are used to calculate the average scores of the numerical values for danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, temp, time_signature, and duration_ms.
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

 
The print_max_min() function calculates the max and min function of each variable, the below is an example of the calculation of the max and min of the variable danceability.
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




Function hist_pop() to create the histogram for popularity.
def hist_pop():
  plt.figure()
  sns.histplot(songs["popularity"],bins='auto')
  plt.savefig(output_plots, format='pdf')
  plt.show()


More functions to create histograms for variables.
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

 
 

Main() function to print all of the previous functions we have defined.
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

 
Final print:
main()
files.download('output_descriptive.txt')
files.download('plots_descriptive.pdf')
 

Predictive stats:
Packages pandas, matplotlib, pyplot, seaborn, and numpy are downloaded for this script
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

songs = pd.read_csv('/drive/MyDrive/taylor_swift_spotify.csv')
output_file = open("output_predictive.txt", "w")
output_plots = PdfPages("plots_predictive.pdf")


Function lin_regression() is defined to plot the observed vs predicted data, providing a line of best fit, mean squared error, and r2 score.
def lin_regression():
  # Selecting features and target variable
  features = songs[['acousticness', 'danceability', 'energy', 'instrumentalness',
                  'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'duration_ms']]
  target = songs['popularity']

  # Scaling features
  scaler = StandardScaler()
  features_scaled = scaler.fit_transform(features)

  # Splitting the dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

  # Creating and training the linear regression model
  model = LinearRegression()
  model.fit(X_train, y_train)

  # Making predictions on the test set
  predictions = model.predict(X_test)
  mse = mean_squared_error(y_test, predictions)
  r2 = r2_score(y_test, predictions)

  # calculating line of best fit
  a, b = np.polyfit(y_test, predictions, 1)
  plt.plot(y_test, a*y_test + b, color='red', label='Best Fit Line')

  # Plotting the observed vs predicted values for visual comparison
  plt.scatter(y_test, predictions)
  plt.xlabel('Observed')
  plt.ylabel('Predicted')
  plt.title('Observed vs Predicted Popularity')
  plt.savefig(output_plots, format='pdf')
  plt.show()

  print('Mean Squared Error:', mse)
  print('R^2 Score:', r2)
  output_file.write('Mean Squared Error:'+ str(mse) + "\n")
  output_file.write('R^2 Score:'+ str(r2) + "\n")


 
The log_regression() function performs Logistic Regression with K-Fold cross-validation finding the best hyperparameters for predicting song popularity based on selected features. Providing insights into the optimal model configuration and its performance metrics.
def log_regression():
  global features
  features = songs[['acousticness', 'danceability', 'energy', 'instrumentalness',
                  'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'duration_ms']]
  target = songs['popularity']
  target = (target > target.median()).astype(int)

  scaler = StandardScaler()
  kfold = KFold(n_splits=10, random_state=42, shuffle=True)

  # Initialize the best model and parameters
  global model
  best_model = None
  best_params = {}
  best_accuracy = 0
  best_std = 0

  global X_train, X_test, y_train, y_test
  X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
  global X_train_scaled, X_test_scaled
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)



  for C in [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]:
      for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag']:
          # Set up the model with the current set of parameters
          model = LogisticRegression(C=C, solver=solver, max_iter=10000)

          # List to store accuracy for each fold
          accuracy = []

          # Perform K-Fold cross-validation
          for train_idx, test_idx in kfold.split(features):
              # Split the data
              X_train, X_test = features.values[train_idx], features.values[test_idx]
              y_train, y_test = target.values[train_idx], target.values[test_idx]

              # Scale the features
              X_train = scaler.fit_transform(X_train)
              X_test = scaler.transform(X_test)

              # Train the model
              model.fit(X_train, y_train)

              # Calculate accuracy
              score = model.score(X_test, y_test)
              accuracy.append(score)

          # Calculate the average accuracy and standard deviation
          avg_accuracy = np.mean(accuracy)
          std_accuracy = np.std(accuracy)

          # Update the best model if the current model is better
          if avg_accuracy > best_accuracy:
              best_model = model
              best_params = {'C': C, 'solver': solver}
              best_accuracy = avg_accuracy
              best_std = std_accuracy

  print("Best Model Parameters:", best_params)
  print("Best Cross-Validation Accuracy: "+ str(round(best_accuracy,2) * 100) + "%")
  print("Standard Deviation of CV Accuracy: "+ str(round(best_std,2) * 100)+ "%")

 
The coeff() function prints the coefficients and features importance.
def coeff():
  coefficients = model.coef_[0]
  # Get feature names
  feature_names = features.columns

  # Create a DataFrame to display coefficients and feature importance
  coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

  # print coefficients
  print("Coefficients:")
  output_file.write("\n Coefficients: \n")
  print(coefficients_df)
  output_file.write(str(coefficients_df))

  # print 'feature_importance'
  print("\nFeature Importance:")
  output_file.write("\n Feature Importance: \n")
  print(coefficients_df[['Feature', 'Coefficient']].sort_values(by='Coefficient', key=abs, ascending=False))
  output_file.write(str(coefficients_df[['Feature', 'Coefficient']].sort_values(by='Coefficient', key=abs, ascending=False)))



The log_plot() function plots the probability of the positive class against the actual values.

def log_plot():
  y_probs = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

  # Plotting the predicted probabilities against the actual values
  plt.figure(figsize=(10, 6))
  plt.scatter(range(len(y_test)), y_probs, c='r', label='Actual')
  plt.scatter(range(len(y_test)), y_test, alpha=0.5, edgecolor='k', label='Predicted')
  plt.title('Predicted Probabilities and Actual Values')
  plt.xlabel('Samples')
  plt.ylabel('Probability')
  plt.legend()
  plt.savefig(output_plots, format='pdf')
  plt.show()


Printing the functions using main().
def main():
  lin_regression()
  log_regression()
  coeff()
  log_plot()
  output_file.close()
  output_plots.close()


The following was printed.
main()
files.download('output_predictive.txt')
files.download('plots_predictive.pdf')
 
 

Diagnostic Stats:
Importing the pandas, matplotlib.pyplot, seaborn, and numpy files.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


Function corr() calculates the correlation and plots the heatmap
ef corr():
  # Selecting columns to find correlation
  columns = songs[['acousticness', 'danceability', 'energy', 'instrumentalness',
                  'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'duration_ms', 'popularity']]

  # Create the correlation matrix
  corr = columns.corr()

  # Set up the matplotlib figure
  f, ax = plt.subplots(figsize=(11, 9))

  # Set heatmap colors
  cmap = sns.color_palette("rocket", as_cmap=True)

  # Draw the heatmap with the mask and correct aspect ratio
  sns.heatmap(corr, cmap=cmap, vmax=.3, annot=True, center=0,
              square=True, linewidths=.5, cbar_kws={"shrink": .5})
  plt.savefig(output_plots, format='pdf')
  plt.show()

 
Function t_test_acoustic() is a t-test to compare the compare 'acousticness' between songs with high and low popularity. The songs are categorized into high and low popularity based on the median. The function calculates the t-statistic and p-value.

# Perform a t-test to compare the compare 'acousticness' between songs with high and low popularity

from scipy import stats

# Categorize songs into high and low popularity based the median

def t_test_acoustic():
  median_popularity = songs['popularity'].median()
  high_popularity = songs[songs['popularity'] >= median_popularity]
  low_popularity = songs[songs['popularity'] < median_popularity]

  # Perform the t-test
  t_stat, p_val = stats.ttest_ind(high_popularity['acousticness'], low_popularity['acousticness'])

  print("T-Statistic:", t_stat)
  print("P-Value:", p_val)
  output_file.write("T-Statistic:" + str(t_stat))
  output_file.write("P-Value:" + str(p_val))

  # When a p-value less than 0.05, it is considered statistically significant
  if p_val < 0.05:
      print("The difference in acousticness between high and low popularity songs is statistically significant.")
      output_file.write("The difference in acousticness between high and low popularity songs is statistically significant.")
  else:
      print("There're no significant difference in acousticness between high and low popularity songs.")
      output_file.write("There're no significant difference in acousticness between high and low popularity songs.")



Function scatter(), facilitates the visual exploration of the relationship between each independent variable and the 'popularity' target variable through a series of scatterplots, aiding in the identification of potential trends or patterns in the data.

def scatter():
  # Scatterplots between each independent variable and popularity
  # Create list for independent variables
  independent_variables = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                          'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'duration_ms']

  popularity = 'popularity'
  n_rows = len(independent_variables) // 2 + len(independent_variables) % 2
  fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(14, n_rows * 4))

  # Flatten the axes array for easy indexing
  axes = axes.flatten()

  # Plot each independent variable vs the target variable
  for i, var in enumerate(independent_variables):
      sns.scatterplot(x=songs[var], y=songs[popularity], ax=axes[i])
      axes[i].set_xlabel(var)
      axes[i].set_ylabel(popularity)
      axes[i].set_title(f'Scatter plot of {var} vs {popularity}')

  plt.tight_layout()
  plt.savefig(output_plots, format='pdf')
  plt.show()


main() prints the final results.

def main():
  corr()
  t_test_acoustic()
  scatter()
  output_file.close()
  output_plots.close()

main()
files.download("output_diagnostic.txt")
files.download("plots_diagnostic.pdf")
 

