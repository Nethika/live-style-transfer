# Style Transfer Demo

## Using OpenCV and models trained with Torch.


## To run:

Copy `models` folder from [AI coloring](https://github.com/Nethika/ai_coloring)

```
python video.py --models models
```

The Demo will change styles every 30 seconds.

We have 14 styles.

Press the Key `Q` to stop the Demo.

## fast-neural-style

Perceptual Losses for Real-Time Style Transfer and Super-Resolution 

https://cs.stanford.edu/people/jcjohns/eccv16/ Justin Johnson, Alexandre Alahi, Li Fei-Fei Presented at ECCV 2016

The paper builds on A Neural Algorithm of Artistic Style by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge by training feedforward neural networks that apply artistic styles to images. After training, our feedforward networks can stylize images hundreds of times faster than the optimization-based method presented by Gatys et al.

