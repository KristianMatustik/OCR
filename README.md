Simple school project for digit recognition. 

I implemented a simple NN from scratch (inspired by https://www.youtube.com/watch?v=hfMk-kjRv4c) and a method for cropping the digits out of the image. I trained the NN for handwritten digits using the MNIST data set, on which it managed to get over 96 % accuracy on testing data. The cropping made it perform really well even with reduced NN size (turns 28x28 image (8bit color) into 14x14 as it is implemented currently), however, makes it useless when introducing noise the image, since it crops the image only by completely empty pixels. One single pixel can thus severly alter the cropped image, off centering the digit and giving way different, usually wrong result. This could be probably fixed by training the NN on data with noise (the cropping would then be mostly useless and just complication though), or implementing more sophisticated cropping mechanism, that could identify the digit correctly within a noisy data.

The app provides super simple GUI to test on own images. Simply draw with right click on the left dark panel to create the input image. The right panel then shows the cropped image and below is displayed table with the 10 digits and their respective probabilities. The image can then be erased by right click - dragging mouse or completely scraped by space bar.



Code and GUI is kinda mess, hopefully will get some time and motivation in the future to celan it up and improve it.
