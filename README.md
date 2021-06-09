# CVPR-NAS-Datasets
Here you'll find descriptions of all 6 of the datasets from the [CVPR-NAS 2021 Competition Track 3 ](https://competitions.codalab.org/competitions/29853).

# Prequisites:
* `numpy`
* `sklearn`
* `torchvision`
* `aspell` with the following languages installed: `['en', 'nl', 'de', 'es', 'fr', 'pt_PT', 'sw', 'zu', 'fi', 'sv']`

# Build The Datasets Yourself
While this will not produce exact copies of the datasets from the competition (as some are dynamically generated each time
and the train/test splits may vary), the scripts here will let you recreate each dataset. Run:

`python3 data_packager` 

to produce the `.npy` and `metadata` files that you'll recognize from the competition. The exact datasets used in the competition
will be uploaded to the competition page on CodaLab shortly.

If you're just curious what the datasets were, there are examples below of each dataset. More examples from
each dataset are included in the Jupyter notebook.

# Development Data
## Development Dataset 0: AddNIST
The first development dataset was called AddNIST. Each image contains three MNIST images, one in each of the RGB channels. The output class is `(r + g + b) - 1`, where `r`, `g`, and `b` refer to the digits represented by the three MNIST images. The three images are chosen such that `(r+g+b)-1 < 20`.

The train and validation split use Torch MNIST train images, while test split uses test images. Combinations are chosen at random, but weighted such that each of the 20 classes are evenly represented in bopth data splits. Each of the three component MNIST images are normalized according to the MNIST mean and standard deviation before they are combined into the 3 image datapoints.

Our benchmark model scored a 92.08% on this dataset, and the submission record was 95.06%, achieved by `Atech_AutoML`.

![addNIST](images/add_nist.png "AddNIST")

## Development Dataset 1: FashionMNIST
The second development dataset was just FashionMNIST. 

Our benchmark model scored a 92.87% on this dataset, and the submission record was a 94.44%, achieved by `Atech_AutoML`.

![fashionMNIST](images/fashionmnist.png "FashionMNIST")


## Development Dataset 2: Language
The third development dataset was codenamed language. Here, we loaded the Aspell language dictionary for 10 lanauges that all use the Latin alphabet: English, Dutch, German, Spanish, French, Portuguese, Swedish, Zulu, Swahili, and Finnish. We then filtered these words to only those that use 6 letters total. Of these six letter words, all that used diacritics (letters such as é or ü), y, or z were removed, meaning there were 24 possible letters within each word. These words were then combined into random groups of 4, and one hot encoded. This creates a 24x24 matrix, which consistutes the input image. The six letter words were divided into train and test groups to prevent train/test leakage, meaning that there were no words shared across the train and test word sets used to generate the train and set images.

The image class refers to the original language that the four words come from.

Our benchmark model scored an 87.00% on this dataset, and the submission record was 89.71%, achieved by `yonga`. We were surprised by how well models could learn this data, given how random it looks to the human eye.

![language](images/language.png "Language")

# Evaluation Data
## Evaluation Dataset 0: MultNIST
The first evaluation dataset was codenamed MultNIST. The process of this is very similar to AddNIST, except the output class is `(r*g*b) % 10`; the last digit of the product of `r`, `g`, and `b`. All other processing is identical to that of AddNIST.

Our benchmark model scored a 91.55% on this dataset, and the submission record was 95.45%, achieved by `Atech_AutoML`.

![multNist](images/multnist.png "MultNIST")

## Evaluation Dataset 1: CIFARTile
The second evaluation dataset was codenamed CIFARTile. This takes images from CIFAR-10 and tiles them into a 2x2 grid. The label for the grid is the total number of discrete classes in the tiling _minus 1_ (to ensure that the classes are `0, 1, 2, 3`]. So for example, a tile of `[horse, horse, frog, cat]` has three discrete classes and thus has a label of 2. The train and validation data splits get images from the train set of CIFAR-10, while the test data split gets images from the test set. 

For each grid, a total number of classes `nclasses` is chosen between 1 and 4. If `nclasses` is:

1: 1 class is selected at random and all four images in the tile are from that one class. 

2: 2 classes are selected at random, and there are two images of each class in the tile, placed randomly into the tiling. 

3: 3 classes are selected at random. There are two images of the first class and one of the second and third, with the images placed randomly into the tiling. 

4: 4 classes are selected at random. There is one image of each class, placed randomly into the tiling.


Each individual image in the tile is processed as per the recommended CIFAR-10 augmentation and normalization policy used in the PyTorch documentation: a 32 pixel random crop with padding 4, a random horizontal flip, and a normalization around the global channel mean and standard deviation.

Our benchmark model scored a 45.56% on this dataset, and the submission record was 73.08%, achieved by `SRCB_VC_Lab`.

![CIFARTile](images/cifartile.png "CIFARTile")

## Evaluation Dataset 2: Gutenberg
The third evaluation dataset was codenamed Gutenberg. Here, the following texts were downloaded from [Project Gutenberg](https://www.gutenberg.org/) from six different authors:

* `Thomas Aquinas`: Summa I-II, Summa Theologica, Part III, On Prayer and the Contemplative Life
* `Confucius`: The Sayings of Confucius, The Wisdom of Confucius
* `Hawthorne`: The Scarlet Letter, The House of the Seven Gables
* `Plato`: The Republic, Symposium, Laws
* `Shakespeare`: Romeo and Juliet, Macbeth, Merchant of Venice, King Lear, Twelfth Night
* `Tolstoy`: War and Peace, Anna Karenina

Each text was an English translation of the source material, with authors chosen to represent a wide variety of cultures, time periods, and languages. From each text, basic text preprocessing is performed; removing punctuation, mapping diacritics to their base letters, and removing common 'structure' words (things like "chapter", "scene" or "prologue". The texts were then split into sequences of words. From these word sequences, consecutive sequences of three words that were between 3 and 6 letters long were extracted, called "phrases". Phrases that appeared in multiple authors' corpuses were removed. Each word in each phrase were padded with underscores if they were shorted than 6 letters. These phrases were then one-hotted, and the label corresponds to the original author that wrote the phrase. For example, for Shakespeare, you might have something like `such__sweet_sorrow` or `lady__doth__protest`.


Our benchmark model scored a 40.98% on this dataset, and the submission record was 50.85%, achieved by `Atech_AutoML`.

![CIFARTile](images/gutenberg.png "Gutenberg")

# Competition Results By Dataset
![DevelPerformance](images/devel_graph.png "Devel Performance")
![EvaluationPerformance](images/eval_graph.png "Evaluation Performance")

