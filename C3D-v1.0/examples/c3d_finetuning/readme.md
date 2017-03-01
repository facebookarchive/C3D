Finetuning C3D Pre-trained model on UCF101
=================================================

We provide a simple example of finetuning C3D model (pre-trained on sport1m) on UCF101.

Prepare data
----------------------

* Download ucf101

* Extract frames (you can also use .avi by modifying list files and prototxt files, but this example uses frames)

* Modify train_01.lst and test_01.lst so that the paths point to the directories containing extracted frames. We note that this example uses the first public train/test splits provided by UCF101.

Start finetuning
------------------------
Simply run the following script:

    sh ucf101_finetuning.sh


Test your finetuned model
------------------------
When the training is done, to test the finetuned model, run:

    sh ucf101_testing.sh

The testing results are clip accuracy, which you can further aggregate to compute video-level accuracy.
