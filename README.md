# UDA-repo
UDA reproduction in Cifar-10

# Trainning

python main.py --cuda

load pretrained model:

python main.py --cuda --model_path (your model path)

# Testing

I did not write the code to test individually, I just do it as the validation during training.

# Result

It is difficult for my network to reach the result in paper, because they use a large mini-batch and 400k steps, and I can not do like that.

So I use 5k supervised pictures, 45k unsupervised pictures, a large $\lambda$ equaling to 10, and a small mini-batch, and a different learning rate.

And I use the trick mentioned in paper, Softmax Temperature, which they only use in texture classification. It means  the loss will not involve those unsupervised pictures whose max probability is lower than a threshold. I use the same threshold as TSA. I am not sure about if it works, but as there two points making the unsupervised loss to zero, one is predicting absolutely correct(the prediction of correct class is 100%, and others are 0%),the other ones is predicting the result like random choice(all the prediction is 10%). Using Softmax Temperature can avoid this awful result.

My result is awful, 75.52% at last.

