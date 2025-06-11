# Shadorbs

This is a set of programs that take videos of people playing Slay the Spire
and outputs an agent that can play the game based on what it learned from those videos.
The name is a portmanteau of shadow and Jorbs
since I used his videos as the training data.
For more information, read `thesis.pdf`.
Below is a description of the contents of this repo.

## video2data/

This directory pertains to analysing the video input and extracting data points
that can then be used to train a machine learning model.

### vid2dat.py

This is the script that does all the video analysis.
It takes one argument, which is the name of the video to analyse.
Videos should be put in a directory together with `vid2dat.py`
called `vids`, which is where it looks for the given argument.
The script is single-core, so it should be parallelized
with something like `xargs` or `parallel`.
Use as many processes as you have cores or slightly less.  
Since this project has a limited scope
and has only used videos from Jorbs,
this script is very inflexible in regards to the video input it can process:

- The video must be from V2.0 or later.
  (assuming there are no gameplay changes after V2.2).
- The character needs to be the Ironclad.
- It needs to be 1080p.
- The color channels need to be BGR.
  You can convert each frame to BGR with something like
  `frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)`,
  but it will significantly increase the computation time.
- The username needs to be a specific length
  for the health to be in the place that the script looks.

### bgs/

This directory contains a set of images
which `vid2dat.py` uses for its video analysis.

### data/

This directory contains all the data points
that were used for the experiments described in `thesis.pdf`.
This is also the directory that `vid2dat.py` outputs to,
so backup this directory if you want to keep the premade data.

## data2model/

This directory pertains to training a machine learning model
based on the data points from the previous section.

### dat2mod.ipynb

This is Jupyter notebook with a setup
that uses PyTorch to train a neural network with the data points from `vid2dat.py`.

### best0

This is the best model I was able to train
based on its accuracy on an evaluation set.

### lookup

This file is just for convenience
so you can easily look up which card an ID corresponds to.

## model2plays/

This directory contains modifications for ForgottenArbiter's [Spirecomm](https://github.com/ForgottenArbiter/spirecomm).
These modifications will make the Spirecomm AI use the neural network from the previous section
when deciding which card to play.
To apply the modifications, copy the contents of this directory into the root directory of Spirecomm,
replacing `main.py`.

### main.py

`main.py` is just changed to include some lines of commented code that you can use to
quickly change which AI to use,
and it will only play the Ironclad instead of rotating between characters
since this project is only implemented for the Ironclad.

### main-rl.py

Use this file instead of `main.py`
to use the reinforcement learning actor.
Either rename this file to `main.py`
or reconfigure CommunicationMod to execute `main-rl.py` instead.

### shadorbs.py

This is an agent based on Spirecomm
that uses the model from the previous section
to decide which card to play.
You need to copy the model to a file at 'shadorbs/model'
in the same directory as this file.

### rl.py

This is an agent that uses reinforcement learning
(specifically Actor Advantage-Critic)
to train itself just by playing the game.
It is not recommended to use this agent without [SuperFastMode](https://github.com/Skrelpoid/SuperFastMode).
