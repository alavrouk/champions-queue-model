# League of Legends - Predicting Match Outcomes with Machine Learning
### _Anton Lavrouk (Georgia Tech '24)_
### _Ethan Jiang (Vanderbilt University '24)_

## **OVERVIEW**
Before reading, we recommend you have a basic understanding of the popular online game League of Legends on which this project is based. Skimming [this article](https://mobalytics.gg/blog/absolute-beginners-guide-to-league-of-legends/) should get you up to speed. 

At the most basic level, League of Legends pits players 5 versus 5 in a race to destroy the other team's objectives first. Each player selects a unique champion before the match begins - however, certain champions are always stronger than others given the game's current state of balancing. As a result, we theorize that both **the skill level of the player** (expressed through the player's overall winrate) and **the current viability of their champion** (expressed through the champion's overall winrate) will help make a meaningful prediction on a match's outcome.

This project uses machine learning to try and predict the outcome of League of Legends Champion's Queue matches. Predicting match outcome tends to be a lot easier when using retrospective or real-time data (e.g. game length or takedowns per team); in fact, a similar feature is implemented directly in _Counter Strike: Global Offensive_, which tries to predict your teams win percentage chance as the round happens. We wanted to challenge ourselves by trying to predict a matches outcome before it even occurs, simply from the people playing it and the champions they choose.

### A Quick Summary - League of Legends Champion's Queue
For the most competitive League of Legends players, the outcome of a match can be decided by just one or two split-second decisions. Individual reaction time is thus crucial to performance, but connection latency is perhaps just as important, especially for those in the game's [rapidly-growing professional scene](https://www.cnbc.com/2019/04/14/league-of-legends-gets-more-viewers-than-super-bowlwhats-coming-next.html).

Pro players often find it helpful to practice in ultra-low latency environments similar to those found in on-stage competition. Unfortunately, such an environment is only readily available for players in regions where server density is high (e.g. South Korea and China). To try and solve this issue, **Champion's Queue** was [introduced in early 2022](https://lolesports.com/article/champions-queue-launches-february-7/blt96b81a6e363cd602), offering select players in test regions closed access to smaller low-latency servers. The invite-only nature of Champion's Queue also helped increase the quality of potential teammates and competitors. 

For professional players, the advent of Champion's Queue is an exciting opportunity to hone their skills in an environment that mimics the real competitive environment. For us, it is a great opportunity to finally be able to collect significant data on high level League of Legends matches. Before, such data was limited to actual competition matches, which are too sparse to generate meaningful datasets.

### Data Collection & Usage
Champion's Queue is currently in **Summer Split 1** (as of today, 18 June 2022). Since League of Legends underwent a [significant rebalancing](https://www.leagueoflegends.com/en-us/news/game-updates/patch-12-10-notes/) right before the start of this split, our models are <ins>trained exclusively on matches from this split</ins>. 

In terms of data collected, the following items were scraped from [championsqueue.gg](championsqueue.gg):

- Each player's winrate
- Each champion's winrate
- All matches from champion's queue summer split one. This includes:
    - The champions picked for the match
    - The players playing the match

The scraper itself is written using the _beautifulsoup_ and _selenium_ python libraries, and it resides in the **dataGenerator.py** file.

First, the page contents is retrieved into an xml object provided by beautifulsoup, as such:

```python
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
driver = webdriver.Chrome(executable_path='driver/chromedriver.exe', options=chrome_options)
driver.get('{}?qty={}'.format("https://championsqueue.gg/players", 1346))
time.sleep(2)
html = driver.page_source
page_content = BeautifulSoup(html, 'lxml')
```

With this page content, you can extract whatever data you see fit. For example, below is the code for acquiring player winrates:

```python
players = []
for player in page_content.find_all('p', class_='name svelte-1jnscn1'):
    players.append(player)
players = np.asarray(players)
# Get rid of player team tag (to avoid issues if player changes teams)
for i in range(len(players)):
    spliced = players[i][0].split(' ')
    if len(spliced) > 1:
        del spliced[0]
        s = " "
        s = s.join(spliced)
        players[i][0] = s
players = players.reshape((players.size, 1))

winrates = []
for winrate in page_content.find_all('span', 'stat winrate svelte-1jnscn1'):
    winrates.append(winrate)
winrates = np.asarray(winrates)
winrates = winrates.reshape((winrates.size, 1))

driver.quit()
data = np.concatenate((players, winrates), 1)
```

Selenium is used when a button needs to be pressed. This happens when I need to pull match data. Unfortunately, getting win loss from this particular website is a bit of a pain. You need to click on the match element each time to get the win loss, and the driver needs to wait a few seconds in between each click to make sure everything loads. As such, the dataGenerator unfortuantely takes quite a bit of time to actually get all of the matches.

## **ALGORITHMS**

### Clustering

- _K-Means_
- _Gaussian Mixture Models_

### Support Vector Machine (SVM)

Support vector machines fit this problem very well as it is binary classification problem. Thus, the resulting hyperplane that SVM generates would be the decision boundary between a win or a loss for team one. We decided to take this opportunity to use the same data as the clustering algorithms so that we would be able to actually visualize the hyperplane. Perhaps if we used more features, we would have a higher accuracy, but obviously visualizing a 19 dimensional hyperplane is not something that science is currently capable of. However, this method yielded a respectable **72% accuracy** anyways. Below is the code of the implementation of this SVM:

```python
clf = svm.SVC(kernel='rbf', gamma=0.7, C=0.5)
clf.fit(Xtrain, ctrain)
cpred = clf.predict(Xtest)
```
The hyperparameters were tuned quite a bit, and eventually the above are the ones that we settled on. In particular, the radial basis function kernel is important to the visualization of the actual hyperplane, which is shown below:

![SVM](/img/SVM.png)

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
