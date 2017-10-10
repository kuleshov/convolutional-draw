# draw

TensorFlow implementation of [Convolutional DRAW](https://arxiv.org/pdf/1604.08772.pdf) on the MNIST generation task.

Based on the implementation of regular DRAW by Eric Jang.

## Usage

`python draw.py --data_dir=/tmp/draw` downloads the binarized MNIST dataset to /tmp/draw/mnist and trains the DRAW model with attention enabled for both reading and writing. After training, output data is written to `/tmp/draw/draw_data.npy`

You can visualize the results by running the script `python plot_data.py <prefix> <output_data>`

For example, 

`python myattn /tmp/draw/draw_data.npy`

To run training without attention, do:

`python draw.py --working_dir=/tmp/draw --read_attn=False --write_attn=False`

## Restoring from Pre-trained Model

Instead of training from scratch, you can load pre-trained weights by uncommenting the following line in `draw.py` and editing the path to your checkpoint file as needed. Save electricity! 

```python
saver.restore(sess, "/tmp/draw/drawmodel.ckpt")
```

## Useful Resources

- https://github.com/vivanov879/draw
- https://github.com/jbornschein/draw
- https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW (wish I had found this earlier)
- [Video Lecture on Variational Autoencoders and Image Generation]( https://www.youtube.com/watch?v=P78QYjWh5sM&list=PLE6Wd9FR--EfW8dtjAuPoTuPcqmOV53Fu&index=3)

