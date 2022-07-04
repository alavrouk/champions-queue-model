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

## _Anton Lavrouk (Georgia Tech '24)_
## _Ethan Jiang (Vanderbilt University '24)_

## **OVERVIEW**
Using only pre-match data, this project uses machine learning to try and predict the outcome of League of Legends Champion's Queue matches. 

etc. etc.

### A Quick Summary - League of Legends Champion's Queue
At the most basic level, League of Legends pits players 5 versus 5 in a race to destroy the other team's objectives first. Each player selects a unique champion before the match begins - however, certain champions are always stronger than others given the game's current state of balancing. 

As a result, **both the skill level of the player and the current viability of their champion** can have a large effect on the

Champion's Queue intends to offer top NA & LATAM players a competitive environment closer to those found in KR and Chinese servers by
1. limiting player pool to pro and select semi-pro players and
2. lowering game ping.

### Data Collection & Usage
Champion's Queue is currently in **Summer Split 1** (as of today, 18 June 2022). Since League of Legends underwent a [significant rebalancing](https://www.leagueoflegends.com/en-us/news/game-updates/patch-12-10-notes/) right before the start of this split, our models are <ins>trained exclusively on matches from this split</ins>. TODO: Add something about webscraping and where we scraped from

## **ALGORITHMS**

### Clustering

- _K-Means_
- _Gaussian Mixture Models_

### Support Vector Machine (SVM)

### Neural Network

Given the almost linear decision boundary shown by the SVM and the clustering algorithms, we figured that a neural network could use its inherent nonlinearity, combined with extra features, to learn some more interesting patterns beyond the simplistic decision boundary found previously. This model used 20 features as follows:

<center>[ t1player1winrate, t1player2winrate, ..., t1champion1winrate, t1champion2winrate, ..., t2player1winrate, t2player2winrate, ..., t2champion1winrate, t2champion2winrate ]</center>

Since a neural network is capable of learning complex patterns, these extra features have the potential to boost the accuracy of the more simple models, even though we do not have that many datapoints to work with.

The following is the neural network architecture that we ultimately decided on:

```
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

```
Kevin = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        # Exponential decay rate for first moment estimates
        beta_1=0.9,
        # Exponential decay rate for second moment estimates
        beta_2=0.999,
        # Should be really small, helps avoid divide by 0
        epsilon=1e-07,
        # AMSGrad is an extension to the Adam version of gradient descent that attempts to improve the convergence
        # properties of the algorithm, avoiding large abrupt changes in the learning rate for each input variable.
        amsgrad=False,
        name='Adam',
    )
    model.compile(optimizer=Kevin,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(Xtrain, ytrain, epochs=150, batch_size=7, validation_data=(Xval, yval))
```

This architecture is a classic example of a binary classifier, which is why it uses the sigmoid activation function in the last layer and the binary crossentropy loss function. The amount of neurons in the dense layers was not tuned to perfection, but we feel like we made a decent selection in terms of model size. Obviously, a larger model tends to overfit more, so we had to regularize some of the larger weights using l2 regularization and added some dropout, which overall helped reduce overfitting by a good bit. We played around with other activation functions, like leaky_relu and adding a few more sigmoids, but they did not create a large difference in performance so we stuck with the tried and true ReLU.

### Random Forest



