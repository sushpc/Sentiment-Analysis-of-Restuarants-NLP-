# Natural Language Processing

library(caret)

# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv',quote = "",stringsAsFactors = FALSE)

# Cleaning the texts

library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Fitting Logistic Regression to the Training set
classifier = glm(formula = Liked ~ .,
                 family = binomial,
                 data = training_set)
# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-692])
y_pred = ifelse(prob_pred > 0.5, 1, 0)
# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_pred = factor(y_pred , levels = c(1,0), labels = c(1,0))
y_test = factor(test_set$Liked , levels = c(1,0), labels = c(1,0))

cm = confusionMatrix(y_pred, y_test)
cm$table
tp = cm$table[1,1]  # true positive
accuracy = cm$overall[[1]]
precision = tp/(cm$table[1,1] +cm$table[1,2])
recall = tp/(cm$table[1,1]+cm$table[2,1]) 
f1_score = 2 * precision * recall / (precision + recall)

# Creating a data frame with results 
Results <- data.frame("Accuracy" = accuracy, "Precision" = round(precision,2), "Recall" = recall, "F1_score" = round(f1_score,2))
row.names(Results) <- "Logistic Regression"

# Fitting Maximum Entropy to the Training set

library(maxent)
classifier = maxent(feature_matrix = training_set[-692],
                    code_vector = training_set$Liked)
# Predicting the Test set results
y_pred = predict(classifier, test_set[-692])
y_pred = y_pred[,1]

# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_pred = factor(y_pred , levels = c(1,0), labels = c(1,0))
y_test = factor(test_set$Liked , levels = c(1,0), labels = c(1,0))

cm = confusionMatrix(y_pred, y_test)
cm$table
tp = cm$table[1,1]  # true positive
accuracy = cm$overall[[1]]
precision = tp/(cm$table[1,1] +cm$table[1,2])
recall = tp/(cm$table[1,1]+cm$table[2,1]) 
f1_score = 2 * precision * recall / (precision + recall)

# Adding the results obtained in the dataframe "Results"

Results["Maximum Entropy", ] = c(accuracy, round(precision, 2), recall, round(f1_score,2))

# Fitting K-NN to the Training set and Predicting the Test set results
library(class)
y_pred = knn(train = training_set[, -692],
             test = test_set[, -692],
             cl = training_set$Liked,
             k = 5,
             prob = TRUE)
# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_pred = factor(y_pred , levels = c(1,0), labels = c(1,0))
y_test = factor(test_set$Liked , levels = c(1,0), labels = c(1,0))

cm = confusionMatrix(y_pred, y_test)
cm$table
tp = cm$table[1,1]  # true positive
accuracy = cm$overall[[1]]
precision = tp/(cm$table[1,1] +cm$table[1,2])
recall = tp/(cm$table[1,1]+cm$table[2,1]) 
f1_score = 2 * precision * recall / (precision + recall)

# Adding the results obtained in the dataframe "Results"

Results["k-NN (k=5)", ] = c(accuracy, round(precision, 2), recall, round(f1_score,2))
Results

# Fitting SVM to the Training set
library(e1071)
classifier = svm(formula = Liked ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])
# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_pred = factor(y_pred , levels = c(1,0), labels = c(1,0))
y_test = factor(test_set$Liked , levels = c(1,0), labels = c(1,0))

cm = confusionMatrix(y_pred, y_test)
cm$table
tp = cm$table[1,1]  # true positive
accuracy = cm$overall[[1]]
precision = tp/(cm$table[1,1] +cm$table[1,2])
recall = tp/(cm$table[1,1]+cm$table[2,1]) 
f1_score = 2 * precision * recall / (precision + recall)

# Adding the results obtained in the dataframe "Results"

Results["SVM", ] = c(accuracy, round(precision, 2), recall, round(f1_score,2))
Results

# Fitting Decision Tree Classification to the Training set

library(rpart)
classifier = rpart(formula = Liked ~ .,
                   data = training_set)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692], type = 'class')

# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_pred = factor(y_pred , levels = c(1,0), labels = c(1,0))
y_test = factor(test_set$Liked , levels = c(1,0), labels = c(1,0))

cm = confusionMatrix(y_pred, y_test)
cm$table
tp = cm$table[1,1]  # true positive
accuracy = cm$overall[[1]]
precision = tp/(cm$table[1,1] +cm$table[1,2])
recall = tp/(cm$table[1,1]+cm$table[2,1]) 
f1_score = 2 * precision * recall / (precision + recall)

# Adding the results obtained in the dataframe "Results"

Results["Decision Tree", ] = c(accuracy, round(precision, 2), recall, round(f1_score,2))
Results

# Apply C50 algorithm on training set

library(C50)
classifier = C5.0(formula = Liked ~ ., data = training_set)
y_pred = predict(classifier, newdata = test_set[-692], type = 'class')
# Evaluate model prediction error using accuracy, precision-recall and F1 score

y_pred = factor(y_pred , levels = c(1,0), labels = c(1,0))
y_test = factor(test_set$Liked , levels = c(1,0), labels = c(1,0))

cm = confusionMatrix(y_pred, y_test)
cm$table
tp = cm$table[1,1]  # true positive
accuracy = cm$overall[[1]]
precision = tp/(cm$table[1,1] +cm$table[1,2])
recall = tp/(cm$table[1,1]+cm$table[2,1]) 
f1_score = 2 * precision * recall / (precision + recall)

# Adding the results obtained in the dataframe "Results"

Results["C5.0", ] = c(accuracy, round(precision, 2), recall, round(f1_score,2))


# Fitting Random Forest Classification to the Training set

library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_pred = factor(y_pred , levels = c(1,0), labels = c(1,0))
y_test = factor(test_set$Liked , levels = c(1,0), labels = c(1,0))

cm = confusionMatrix(y_pred, y_test)
cm$table
tp = cm$table[1,1]  # true positive
accuracy = cm$overall[[1]]
precision = tp/(cm$table[1,1] +cm$table[1,2])
recall = tp/(cm$table[1,1]+cm$table[2,1]) 
f1_score = 2 * precision * recall / (precision + recall)

# Adding the results obtained in the dataframe "Results"

Results["Random Forest", ] = c(accuracy, round(precision, 2), recall, round(f1_score,2))


# Fitting Naive Bayes to the Training set

library(e1071)
classifier = naiveBayes(x = training_set[-692],
                        y = training_set$Liked)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Evaluate model prediction error using accuracy, precision-recall and F1 score
y_pred = factor(y_pred , levels = c(1,0), labels = c(1,0))
y_test = factor(test_set$Liked , levels = c(1,0), labels = c(1,0))

cm = confusionMatrix(y_pred, y_test)
cm$table
tp = cm$table[1,1]  # true positive
accuracy = cm$overall[[1]]
precision = tp/(cm$table[1,1] +cm$table[1,2])
recall = tp/(cm$table[1,1]+cm$table[2,1]) 
f1_score = 2 * precision * recall / (precision + recall)

# Adding the results obtained in the dataframe "Results"

Results["Naive Bayes", ] = c(accuracy, round(precision, 2), recall, round(f1_score,2))

library(dplyr)
Results <- tibble::rownames_to_column(Results, "Model")
Results <- arrange(Results, desc(F1_score))
Results
# Plot each model measurement
ggplot(Results, aes(reorder(Model, -F1_score), F1_score)) +
  geom_col()+
  scale_fill_grey()+
  coord_flip()+
  xlab("Models")+
labs(title="Comparisson of Classification Model Performances in Binary Sentiment Analysis (Like or Dislike)")
 
write.csv(Results, "Model Comparison", row.names = FALSE)   
