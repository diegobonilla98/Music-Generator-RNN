# Music-Generator-RNN
AI tries to write music from experience


This took a very long time to finish but it's absolutely worth it.

At first I wanted to make predictions from songs (classical style),  but when I tried to load the .wav file into numpy array turned out to be more than 7 million values (22kHz of data, 5+min song). This flagged a MemoryError almost in every step. Also, the number of combinations notes were almost as big as the input, so having a neural network with millions of output probabilities is no good.

I figured out that I need to reduce the amount of data severally and also make the number of possible outputs less than 50.
I finally came up with "digitalizing" myself some songs with the help of a piano so I could control the maximum notes played, the tempo, the key and other parameters. In the folder "music" there are 9 files with numbers from 0 to 24 (#keys in two piano octaves, from C3 to B4) which let me play almost all the songs I know making sure to play one note at the time. This took several time that's why there are only 9 of them. (edit: I actually did 9 more but you get the idea)

Since I had so many little data, I started the model with 256 Long Short-Term Memory (to keep a melody) and then added regularization (dropout of 20%) to make sure the output song doesn't sound much like the input data. The output is pretty straight forward and consists in a dense layer with softmax activation. (~0.6 accuracy, ~1.0 loss)

After generating a fixed number of notes using the model, I applied sound effects like reverb to make it sound better and filter the high pitches and save it as a wav file.

The results were AMAZING and some of them sounded very good. Other sounded (or had parts) like the songs I picked for the data, but I'm pretty sure is because there are very few songs in the dataset.


EDIT: I extended the functionality with a sloppy piano so the output audio is better. Some songs are really pleasant to listen to. Some other not that much...
