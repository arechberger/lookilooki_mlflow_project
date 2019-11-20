import argparse
import sys
from exp.nb_06 import *
import luke_model
import luke_data
from callbacks import AvgStatsMlFlowCallback

# Command-line arguments
parser = argparse.ArgumentParser(description='model_lookilooki Example')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
#todo add dropout
parser.add_argument('--enable-cuda', type=str, choices=['True', 'False'], default='True',
                    help='enables or disables CUDA training')
parser.add_argument('--experiment_name', type=str, default='mlflow_demo1',
                    help='mlflow experiment name')
parser.add_argument('--tracking_uri', type=str, default='http://lookilooki.root.sx:8000',
                    help='mlflow tracking_uri')
parser.add_argument('--run_name', type=str, default='first_try',
                    help='mlflow run_name')

args = parser.parse_args()


enable_cuda_flag = True if args.enable_cuda == 'True' else False

args.cuda = enable_cuda_flag and torch.cuda.is_available()

x_train,y_train,x_valid,y_valid = luke_data.get_data()
x_train,x_valid = normalize_to(x_train,x_valid)
train_ds,valid_ds = Dataset(x_train, y_train),Dataset(x_valid, y_valid)

c = 1
bs = args.batch_size
data = DataBunch(*get_dls(train_ds, valid_ds, bs), c)


model = luke_model.Luke(cn_dropout=0.05)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
learn = Learner(model, optimizer, loss_func, data)

params = dict(
    bs = bs,
    cn_dropout=0.05,
    lr=args.lr,
    momentum=args.momentum,
    epochs=args.epochs
)

mlflowcallback = partial(
    AvgStatsMlFlowCallback,
    accuracy,
    tracking_uri=args.tracking_uri,
    experiment_name=args.experiment_name,
    run_name=args.run_name,
    params=params,
)
cb_funcs = [mlflowcallback]

if enable_cuda_flag:
    cb_funcs.append(CudaCallback)

run = Runner(cb_funcs=cb_funcs)
run.fit(args.epochs, learn)

print("done")



