*(Originally posted on Quora as an answer to [How do I learn deep learning in 2 months?](https://www.quora.com/How-do-I-learn-deep-learning-in-2-months/answer/Vivek-Kumar-893))*

If you have coding experience with an engineering background or relevant knowledge in mathematics and computer science, in just two months you can become proficient in deep learning. Hard to believe? Here's a four-step process that makes it possible. 


For more inspiration check out the following video by Andrew Ng
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/n1ViNeWhC24/hqdefault.jpg)](http://www.youtube.com/watch?v=n1ViNeWhC24)

## Step 0: Learn Machine Learning Basics

*(Optional, but highly recommended)*

Start with Andrew Ng's Class on machine learning
[https://www.coursera.org/learn/machine-learning](https://www.coursera.org/learn/machine-learning).
His course provides an introduction to the various Machine Learning algorithms currently out there and, more importantly, the general procedures and methods for machine learning, including data preprocessing, hyper-parameter tuning, and more.

I would also recommend reading the [NIPS 2015 Deep Learning Tutorial](http://www.iro.umontreal.ca/~bengioy/talks/DL-Tutorial-NIPS2015.pdf) by Geoff Hinton, Yoshua Bengio, and Yann LeCun, which offers an introduction at a slightly lower level.

## Step 1: Dig into Deep Learning

My personal learning preference is to watch lecture videos, and there are several excellent courses online. Here are few classes I especially like and can recommend: 


* [Deep Learning at Oxford
2015](http://www.cs.ox.ac.uk/teaching/courses/2014-2015/ml/) Taught by Nando de
Freitas who expertly explains the basics, without overcomplicating it. Start
with Lecture 9 if you are already familiar with Neural Networks and want to go
deep. He uses Torch framework in his examples. ([Videos on
Youtube](https://m.youtube.com/playlist?list=PLE6Wd9FR--EfW8dtjAuPoTuPcqmOV53Fu))
* [Neural Networks for Machine
Learning](https://www.coursera.org/learn/neural-networks): Geoffrey Hinton’s
class on Coursera. Hinton is an excellent researcher who demonstrated the use of
generalized [backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
algorithm and was crucial to the development of [deep
learning](https://en.wikipedia.org/wiki/Deep_learning). I have utmost respect
for him, but I found the delivery of this course bit unorganized. Furthermore,
coursera messes up with the placement of quizzes. Still worth a look, though. 
* [Neural Networks
Class](http://info.usherbrooke.ca/hlarochelle/neural_networks/content.html) by
Hugo Larochelle: Another excellent course
* [Yaser Abu-Mostafa's Machine Learning
Course](https://work.caltech.edu/telecourse.html): More theory, if you are
interested.

If you are more into books as a primary learning tool, here are some excellent resources. 

* [Neural Networks and Deep Learning
Book](http://neuralnetworksanddeeplearning.com/) by [Michael
Nielsen](http://michaelnielsen.org/): Online book that offers several interactive
JavaScript elements to play with.
* [Deep Learning Book](http://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua
Bengio and Aaron Courville: A bit denser, but never the less a great resource

## Step 10: Pick a focus area and go deeper

The next step is to identify what you are passionate about and go deeper. The field
is vast, so this list is in no way a comprehensive one.


### 1. Computer vision

Deep Learning has transformed this area. Stanford’s CS231N course by Andrej
Karpathy's course is the best course I have come across; [CS231n Convolutional
Neural Networks for Visual Recognition](http://cs231n.github.io/). It teaches
you the basics up to convnets, as well as helping you to set up GPU instance
in AWS. Also, take the time to check out [Getting Started in Computer
Vision](https://sites.google.com/site/mostafasibrahim/research/articles/how-to-start)
by [Mostafa S. Ibrahim](https://sites.google.com/site/mostafasibrahim/)

### 2. Natural Language Processing (NLP)

Used for machine translation, question and answering and sentiment analysis. To
master this field, an in-depth understanding of both algorithms and the
underlying computational properties of natural languages is a must. [CS 224N /
Ling 284](http://web.stanford.edu/class/cs224n/) by [Christopher
Manning](http://nlp.stanford.edu/~manning/) is a great course to get started.
[CS224d: Deep Learning for Natural Language
Processing](http://cs224d.stanford.edu/), another Stanford class by David Socher
(founder of [MetaMind](https://www.metamind.io/))is an excellent course
to progress to, as it goes over all the latest Deep Learning research related to NLP. For more
details see [How do I learn Natural Language
Processing?](https://www.quora.com/How-do-I-learn-Natural-Language-Processing/answer/Vivek-Kumar-893?srid=J2jU)

### 3. Memory Network (RNN-LSTM)

Recent work in combining attention mechanism in LSTM Recurrent Neural networks
with external writable memory has led to some impressive work in building
systems that can understand, store and retrieve information in a question &
answering style. This research area got its start in Dr. Yann Lecun’s Facebook
AI lab at NYU. The original paper is available on arxiv: [Memory
Networks](http://arxiv.org/abs/1410.3916). There are then a number of research variants,
datasets, benchmarks, etc. that have stemmed from this work to aid further learning. For example,
Metamind's [Dynamic Memory Networks for Natural Language
Processing](http://arxiv.org/abs/1506.07285) is a great resource

### 4. Deep Reinforcement Learning

Deep Reinforcement Learning was made famous by AlphaGo, the Go-playing system that [recently
defeated](http://www.nytimes.com/2016/03/16/world/asia/korea-alphago-vs-lee-sedol-go.html?__hstc=13887208.2c86f1d755a00edda38e8cb1d7fb3483.1473023471841.1473023471841.1473023471844.2&__hssc=13887208.1.1473023471844&__hsfp=1720600770)
the strongest Go players in history. David Silver's (Google Deepmind) [Video
Lectures on RL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html), and
Professor [Rich Stutton's book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html) are
a great place to start. For a gentler introduction to LSTM see Christopher’s post
on [Understanding LSTM networks
](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) and Andrej Karpathy’s
[The Unreasonable Effectiveness of Recurrent Neural
Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### 5. Generative Models

While discriminatory models try to detect, identify and separate things, they
end up looking for features which differentiate and do not understand data at a
fundamental level. Apart from the short-term applications, generative models
provide the potential to automatically learn natural features; categories or
dimensions or something else entirely. Out of the three commonly used generative
models — [Generative Adversarial Networks
(GANs)](http://arxiv.org/abs/1406.2661), [Variational Autoencoders
(VAEs)](https://arxiv.org/abs/1312.6114) and Autoregressive models (such as
[PixelRNN](http://arxiv.org/abs/1601.06759)), GAN's are most popular. To dig
deeper read

* [Original GAN paper by Ian Goodfellow et al.](http://arxiv.org/abs/1406.2661)
* The [Laplacian Adversarial Networks (LAPGAN)
Paper](http://papers.nips.cc/paper/5773-deep-generative-image-models-using-a-laplacian-pyramid-of-adversarial-networks)
(LAPGAN) which fixed the stability issue
* [The Deep Convolutional Generative Adversarial Networks (DCGAN)
paper](http://arxiv.org/abs/1511.06434) and [DCGAN
Code](https://github.com/Newmu/dcgan_code) which can be used to learn a
hierarchy of features without any supervision. Also, check out [DCGAN used for
Image Superresolution](https://github.com/david-gpu/srez)

## Step 11: Build Something

Reading and watching lessons is great, but doing is the real key to becoming an expert. Try to create something which interests you
and matches your skill level. Here are a few suggestions to get you thinking:

* As is tradition, start with classifying the [MNIST
dataset](http://yann.lecun.com/exdb/mnist/)
* Try face detection and classification on [ImageNet](http://image-net.org/index).
If you are up to it, do the [ImageNet Challenge
2016](http://image-net.org/challenges/LSVRC/2016/).
* Perform a Twitter sentiment analysis using
[RNNs](https://cs224d.stanford.edu/reports/YuanYe.pdf) or
[CNNs](http://casa.disi.unitn.it/~moschitt/since2013/2015_SIGIR_Severyn_TwitterSentimentAnalysis.pdf)
* Teach neural networks to reproduce the artistic style of famous painters ([A
Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576v1))
* [Compose Music With Recurrent Neural
Networks](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/)
* [Play ping-pong using Deep Reinforcement
Learning](http://karpathy.github.io/2016/05/31/rl/)
* Use [Neural Networks to Rate a
selfie](http://karpathy.github.io/2015/10/25/selfie/)
* Automatically [color Black & White pictures using Deep
Learning](https://twitter.com/ColorizeBot)

For more inspiration, take a look at CS231n [Winter
2016](http://cs231n.stanford.edu/reports2016.html) and [Winter
2015](http://cs231n.stanford.edu/reports.html) projects. Also, keep an eye on the
Kaggle and HackerRank competitions for fun stuff and the opportunities to
compete and learn.

## Continue Learning

Learning never truly ends. Here are some pointers to help you with continuous learning

* Read some excellent blogs. Both [Christopher Olah's blog](https://christopherolah.wordpress.com/) & 
[Andrew Karpathy's Blog](http://karpathy.github.io/) do a great job of explaining basic concepts
and recent breakthroughs
* Follow influencers on Twitter. Here are a few to get started @drfeifei, @ylecun,
@karpathy, @AndrewYNg, @Kdnuggets, @OpenAI, @googleresearch. (see: [Who to
follow on Twitter for machine learning information
?](https://www.quora.com/Who-should-I-follow-on-Twitter-to-get-useful-and-reliable-machine-learning-information/answer/Vivek-Kumar-893)
)
* Joining the [Google+ Deep Learning Community](https://plus.google.com/communities/112866381580457264725), by Yann Lecunn, is a good way to keeping in touch with innovations in deep learning as
well as communicating with other deep learning professionals and enthusiasts.

See
[ChristosChristofidis/awesome-deep-learning](https://github.com/ChristosChristofidis/awesome-deep-learning),
a curated list of awesome Deep Learning tutorials, projects and communities for
more fun.
