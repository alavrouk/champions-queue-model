<link rel="shortcut icon" type="image/x-icon" href="favicon.ico">

# League of Legends - Predicting Match Outcomes with Machine Learning
### _Anton Lavrouk (Georgia Tech '24)_
### _Ethan Jiang (Vanderbilt University '24)_

## **OVERVIEW**
Before reading, we recommend you have a basic understanding of the popular online game League of Legends on which this project is based. Skimming [this article](https://mobalytics.gg/blog/absolute-beginners-guide-to-league-of-legends/) should get you up to speed. 

At the most basic level, League of Legends pits players 5 versus 5 in a race to destroy the other team's objectives first. Each player selects a unique champion before the match begins - however, certain champions are always stronger than others given the game's current state of balancing. As a result, we theorize that both **the skill level of the player** (expressed through the player's overall winrate) and **the current viability of their champion** (expressed through the champion's overall winrate) will help make a meaningful prediction on a match's outcome.

This project uses machine learning to try and predict the outcome of League of Legends Champion's Queue matches. Predicting match outcome tends to be a lot easier when using retrospective or real-time data (e.g. game length or takedowns per team); in fact, a similar feature is implemented directly in _Counter Strike: Global Offensive_, which tries to predict your teams win percentage chance as the round happens. **We wanted to challenge ourselves by trying to predict a matches outcome before it even occurs, simply from the people playing it and the champions they choose.**

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

To get a sense of what our data actually entailed, we decided to run clustering algorithms to both visualize the data and run rudimentary predictions. In order to do so, original data was re-expressed from 20 to 2 dimensions, as follows:

![CLUSTERINGDATA](/img/clusteringData.png)

In essence, there are two features, one for each team. Each feature equally weights the average winrate of the _players on the team_ and the average winrate of the _champions they are playing_. Thus, each feature has a range of 0-100. 

With this data in hand, we proceeded to run two clustering algorithms: **k-means** and **Gaussian mixture models**. In theory, both of those algorithms would then create two clusters representing victory or defeat (from the perspective of Team 1). 

Code snippets from both clustering models are shown below:

```python
df = DataFrame(kmData, columns=['team1wr', 'team2wr'])
kmeans = KMeans(n_clusters=2)
predictions = kmeans.fit_predict(df)

df = DataFrame(clusterData, columns=['team1wr', 'team2wr'])
gmm = GaussianMixture(n_components=2)
predictions = gmm.fit_predict(df)
```

Both models hovered around **73% accuracy** - the graphs below indicate which datapoints the models predicted correctly (green) and incorrectly (red).

![GMMGRAPH](/img/gmm.png)

![KMEANSGRAPH](/img/kmeans.png)

Unsurprisingly, both clustering algorithms struggled to classify points closer to the decision boundary. Since the majority of data is clustered around this troublesome area, we turn to other approaches more applicable to our problem.

---

### Support Vector Machine (SVM)

Binary classification problems such as win/loss prediction are well-suited for use with SVMs, as the model's resulting hyperplane naturally acts as the decision boundary between a win or a loss. For this model, we decided to use the same two-dimensional data as the clustering models so that the hyperplane could be represented graphically. 

Training on more features might have marginally increased accuracy, but would have come at the cost of simplistic visualization. However, this method yielded a respectable **72% accuracy** regardless. 

Below is a rundown of our SVM implementation:

```python
clf = svm.SVC(kernel='rbf', gamma=0.7, C=0.5)
clf.fit(Xtrain, ctrain)
cpred = clf.predict(Xtest)
```
After some hyperparameter tuning, we settled on the above values. In particular, the radial basis function kernel is important to the visualization of the actual hyperplane, which is shown below:

![SVM](/img/SVM.png)

---

### Neural Network

From our clustering and SVM experimentation, we observed that both models struggled to consistently predict matches between teams of similar average winrate, i.e. the hardest matches to predict were those lying on the y=x line in the graphs above. As neural networks are nonlinear in nature, we were interested to see if such a model would pick up on more subtleties from a larger amount of features.

Instead of simplifying player and champion winrates into a single representative value for each team (as done in the above models), we left the features separate as shown below:

![NEURAL NETWORK DATA FORMAT](/img/neuralNetDataFormat.png)

Since a neural network is capable of learning complex patterns, these extra features have the potential to boost accuracy beyond what was achieved in the simpler models above.

The following is the neural network architecture we ultimately decided on

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

Ultimately, we decided against hinge loss. We had a lot of points near the decision boundary, and hinge loss could have overly penalized those points. 

The amount of neurons in the dense layers was not tuned to perfection, but we feel like we made a decent selection in terms of model size. A larger model tends to overfit more, so we had to regularize some of the larger weights using l2 regularization and add dropout, which overall helped reduce overfitting by a good bit. We played around with other activation functions, like `leaky_relu` and adding a few more sigmoids, but they did not create a large difference in performance so we stuck with the tried and true ReLU.

Below are some graphs regarding the performance of the model through 150 epochs on both the training and validation sets. Overfitting was kept to a minimum and both plots line up for the most part. Also, the rate of descent on the loss graph seems to indicate a solid choice in learning rate.

![ACCURACY](/img/trainingAndValidationAccuracy.png)

![LOSS](/img/trainingAndValidationLoss.png)

In terms of accuracy, the neural network on average performed at around **80%**.

---

### Random Forest

It is common for random forest and neural network models to achieve a similar accuracy on a relatively small dataset (such as ours, at ~1000 data points). However, this did not turn out to be the case, as random forest achieved around a **70% accuracy**. 

Below, you can see the first few layers of one of the decision trees that was part of the forest. Note that splitting values are altered from the original 0-100 scale due to PCA.

![DECISIONTREE](/img/decisionTree.png)

Why the lower accuracy? Given the nature of our training data (formatting is the same as for neural network - see above section), it would make sense for each feature to have the same importance in terms of overall variance. Each winrate would play into the result of the game almost equally, since League of Legends is balanced around the fact that each role should (in theory) have the same impact on the result of the game, especially in a high-skill environment like Champion's Queue. 

This fact, combined with the binary nature of our classification, causes the random forest approach to be less effective for this particular instance. Taking a look at the model's confusion matrix below, we can see that the model is only predicting losses correctly 60% of the time (Actual, 0) and wins 70% of the time (Actual, 1).

![CONFUSIONMATRIX](/img/confusionMatrix.png)

To see if the model would improve from a reduced number of training features, we decided to perform principal component analysis on our data. This left us with 15 features (arbitrary, but retained about 80% of total variance) as opposed to 20.

```python
pca = PCA(n_components=8)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
```

Unfortunately, this alteration did not significantly improve accuracy and was ultimately omitted.

Tuning hyperparameters resulted in around 100 decision trees, resulting in the final classifier shown below:

```python
 rf = RandomForestClassifier(n_estimators=100)
 rf.fit(X_train, y_train)
 y_pred = rf.predict(X_test)
```
