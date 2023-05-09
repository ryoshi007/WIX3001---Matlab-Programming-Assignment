# WIX3001 - Matlab Programming Assignment 📚🚀

Welcome to the `WIX3001 - Matlab Programming Assignment` repository! This project is developed by **Wong Yan Jian** (Matric Number: U2102753) from the Bachelor of Data Science program at Universiti Malaya. 🎓🌟

The main goal of this assignment is to utilize a **Genetic Algorithm** to optimize a **Neural Network** for predicting the output of three databases: Wine Recognition 🍷, Students' Dropout & Academic Success 📈, and Car Evaluation 🚗. 

## Contents 📖

This repository contains the following:

- Matlab Codes 📝
- Results 📊
- Graphs 📈
- Datasets 📁

## Genetic Algorithm Enhancements 🧬

The Genetic Algorithm has been improved for better performance by incorporating the following techniques:

1. Stochastic Universal Sampling 🎲
2. Simulated Annealing 🔥
3. Adaptive Selection and Mutation Rate 🔄
4. Gaussian Mutation 🌐
5. Triggered Hypermutation 🚀

Moreover, other methods have been explored to test the code performance, such as:

- Creep Mutation 👣
- Rank-Based Selection 📊
- Selection Based on Random Permutation 🔀

## Additional Improvements (Updates) 🆕

The overall performance of genetic algorithm could be enhanced by normalizing the features using z-score normalization. This helps to standardize the range of the independent variables or features, leading to better performance of machine learning models. The code of the orignal files `run_ann_withGA` could be updated as follow:

```matlab
% load dataset
data = dlmread('student_academic_success.csv');

X = data(:, 1:end-1); % assuming all columns are attributes
Y = data(:, end); % except for the last column for labels

% Calculate the mean and standard deviation for each feature
mean_values = mean(X);
std_values = std(X);

% Normalize the features using z-score normalization
X = (X - mean_values) ./ std_values;

% rest of the code
```

## How to Use 💻

The code is designed to work with any dataset, as long as the following conditions are met:

- Values for each attribute are transformed into numerical values 🔢
- Labels are located in the last column 🏷️

## Acknowledgements 🙏

I would like to express my deepest gratitude to my lecturer, **Dr. Liew Wei Shiung**, for his guidance and support throughout this project. His expertise and knowledge have been instrumental in helping me understand the concepts and implement the algorithms effectively.

Additionally, I would like to thank the vast array of resources available on the internet, which have been invaluable in providing information, examples, and inspiration that aided in the completion of this project.

Lastly, I am grateful for the datasets found in the [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/), which allowed for the testing and evaluation of the code's performance on real-world data. This repository is a treasure trove of datasets, providing an excellent platform for students and researchers alike to work on diverse machine learning problems.

Thank you all for making this project a success! 🌟


**🎉 Have fun exploring and optimizing your Neural Networks with Genetic Algorithms! 🧠🌐**
