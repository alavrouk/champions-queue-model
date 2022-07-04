<!-- ## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/alavrouk/LoL-data/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/alavrouk/LoL-data/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out. -->

# League of Legends - Predicting Match Outcomes with Machine Learning
### _Anton Lavrouk (Georgia Tech '24)_
### _Ethan Jiang (Vanderbilt University '24)_

## **OVERVIEW**
Before reading, a basic familiarity with the popular online game **League of Legends** is assumed. Skimming the following [link](https://mobalytics.gg/blog/absolute-beginners-guide-to-league-of-legends/) should get you up to speed. 

Using only pre-match data, this project uses machine learning to try and predict the outcome of League of Legends Champion's Queue matches. Predicting match outcome tends to be a lot easier when you are using actual match data, such as game length or takedowns per team. In fact, there is a cool implementation of this in _Counter Strike: Global Offensive_, which tries to predict your teams win percentage chance as the round happens. We wanted to challenge ourselves by trying to predict a matches outcome before it even occurs, simply from the people playing it and the champions they choose.

### A Quick Summary - League of Legends Champion's Queue
Champion's Queue intends to offer top NA & LATAM players a competitive environment closer to those found in KR and Chinese servers by

1. limiting player pool to pro and select semi-pro players and
2. lowering game latency (ping).

For professional players, the advent of champion's queue is an exciting opportunity to hone their skills in an environment that mimics the real competetive envionment. For us, it is a great opportunity to finally be able to collect data from high level league of legends matches. Before, we were limited to simply actual competetve matches, which there are unforunately not that many of per year.

### Data Collection & Usage
Champion's Queue is currently in **Summer Split 1** (as of today, 18 June 2022). Since League of Legends underwent a [significant rebalancing](https://www.leagueoflegends.com/en-us/news/game-updates/patch-12-10-notes/) right before the start of this split, our models are <ins>trained exclusively on matches from this split</ins>. 

TODO: Add something about webscraping and where we scraped from

## **ALGORITHMS**

### Clustering

- _K-Means_
- _Gaussian Mixture Models_

### Support Vector Machine (SVM)

### Neural Network

From our clustering and SVM experimentation, it turned out that the hardest matches for those classification methods to predict were the matches were both teams had a similar average winrate. In other words, the hardest ones to predict were those lying on the y=x line in the graphs above. Thus, we decided to try a neural network. Nonlinear in nature, it may be able to pick up on some subtleties from a larger amount of features. Thus, we decided to not average winrates for teams and instead used the features shown below:

![NEURAL NETWORK DATA FORMAT](/img/neuralNetDataFormat.png)

Since a neural network is capable of learning complex patterns, these extra features have the potential to boost the accuracy of the more simple models, even though we do not have that many datapoints to work with.

The following is the neural network architecture that we ultimately decided on:

```python
model = Sequential()
model.add(Dense(50, input_dim=20, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='sigmoid'))
```

with the following extra hyperparameters/specifications:

```python
opt = tf.keras.optimizers.Adam(
    learning_rate=1e-3,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(Xtrain, ytrain, epochs=150, batch_size=7, validation_data=(Xval, yval))
```

This architecture is a classic example of a binary classifier, which is why it uses the sigmoid activation function in the last layer and the binary crossentropy loss function. Another potential loss function to consider would be hinge loss. Hinge loss not only penalizes misclassified samples but also correctly classifies ones that are within a defined margin from the decision boundary. 

![HINGELOSS](/img/hingeloss.png)

Ultimately, however, we decided against hinge loss. We had a lot of points near the decision boundary, and hinge loss would penalize those points perhaps more than it should have. 

The amount of neurons in the dense layers was not tuned to perfection, but we feel like we made a decent selection in terms of model size. A larger model tends to overfit more, so we had to regularize some of the larger weights using l2 regularization and added some dropout, which overall helped reduce overfitting by a good bit. We played around with other activation functions, like leaky_relu and adding a few more sigmoids, but they did not create a large difference in performance so we stuck with the tried and true ReLU.

Below are some graphs regarding the performance of the model through the 150 epochs on both the training and validation sets. As you can see, the overfitting was kept to a minimum, as for the most parts, both plots line up. Also, the rate of descent on the loss graph seems to indicate a solid choice in learning rate.

In terms of accuracy, neural network on average performed at around **80%**.

![ACCURACY](/img/trainingAndValidationAccuracy.png)

![LOSS](/img/trainingAndValidationLoss.png)

### Random Forest

It is generally common that Random Forest and Neural Network can achieve a similar accuracy on a relatively small dataset (ours was around 1000 data points). However, this did not turn out to be the case, as random forest achieved around a **70% accuracy**. Below, you can see the first few layers of one of the decision trees that was part of the forest.

![DECISIONTREE](/img/decisionTree.png)

So why was our accuracy so low? An interesting fact about our data (formatted the same as the neural network), is that it would make sense for each feature to have the same importance in terms of the overall variance. Each winrate would play into the result of the game almost equally, since League of Legends is balanced around the fact that each role should in theory have the same impact on the result of the game, especially in an environment like champions queue, where the players are relatively evenly skilled. Combined with the fact that this is a binary classification problem, this makes the ensemble of decision trees have a difficult time accurately classifying. Below you can see the confusion matrix from our random forest.

![CONFUSIONMATRIX](/img/confusionMatrix.png)

In order to try to get rid of some features, we decided to perform PCA (principal component analysis) on our data, which left us with 15 features (arbitrary, but retained about 80% of variance) as opposed to 20, to see if the forest would react positively to a bit less features. This however, did not necessarily help that much, so was ommited from the final model. The code for that is below:

```python
pca = PCA(n_components=8)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
```

Tuning hyperparameters resulted in around 100 decision trees, and with this information, the following is the code for the random forest classifier:

```python
 rf = RandomForestClassifier(n_estimators=100)
 rf.fit(X_train, y_train)
 y_pred = rf.predict(X_test)
```
