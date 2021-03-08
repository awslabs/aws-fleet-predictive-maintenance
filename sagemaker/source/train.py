import os
import argparse
import json
import multiprocessing

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, accuracy_score

from dl_utils.dataset import PMDataset_torch
from dl_utils.stratified_sampler import StratifiedSampler
from dl_utils.network import Network

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='FPM args')
    parser.add_argument('--train_input_filename', type=str, default="../data/processed/train_dataset.csv",
                        help='input path of the data, default: "../data/processed/train_dataset.csv"')
    parser.add_argument('--test_input_filename', type=str, default="../data/processed/test_dataset.csv",
                        help='input path of the data, default: "../data/processed/test_dataset.csv"')
    parser.add_argument('--output_path', type=str, default="../output",
                        help='output to store model artefacts, default: "../output"')

    parser.add_argument('--sensor_headers', type=str, default=json.dumps(["voltage", "current"]),
                        help='sensors headers in the dataset, default: {}'.format(json.dumps(["voltage", "current"])))
    parser.add_argument('--target_column', type=str, default="target",
                        help='name of the target in the dataset, default: target')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, default: 0.001')
    parser.add_argument('--epochs', type=int, default=200,
                        help='epochs, default: 200')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size, default: 128')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='drop out, default: 0.0')
    parser.add_argument('--fc_hidden_units', type=str, default="[256, 128]",
                       help="Hidden units, default \"[256, 128]\"")
    parser.add_argument('--conv_channels', type=str, default="[2, 8, 2]",
                       help="Conv channels, default \"[2, 8, 2]\"")

    args = parser.parse_args()

    return args

def run_epoch(net, dl, optimizer, critereon, is_train):
    if is_train:
        net.train()
    else:
        net.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_auc = 0.0
    N = 0
    for data, label in dl:
        if use_cuda:
            data = data.to(device).float()
            label = label.to(device).long()
        else:
            data = data.float()
            label = label.long()

        out = net(data)
        loss = critereon(out, label)
        optimizer.zero_grad()
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        predict = out.cpu().detach().numpy()[:, 1]
        total_acc += accuracy_score(label.cpu().detach().numpy(), predict>0.5)
        total_auc += roc_auc_score(label.cpu().detach().numpy(), predict)
        N += 1

    return total_loss/N, total_acc/N, total_auc/N

def run():
    torch.backends.cudnn.enabled = False
    if "SM_HPS" in os.environ:
        is_sm_mode = True
    else:
        is_sm_mode = False
        
    args = parse_args()
    
    assert len(json.loads(args.sensor_headers)) == json.loads(args.conv_channels)[-1], "The last conv filter must be equal the the number of sensor_headers"
        
    if 'SM_CHANNEL_TRAIN' in os.environ:
        train_path = os.path.join(
            os.environ['SM_CHANNEL_TRAIN'], 
            os.path.basename(args.train_input_filename))
    else:
        train_path = args.train_input_filename
        
    if 'SM_CHANNEL_TRAIN' in os.environ:
        test_path = os.path.join(
            os.environ['SM_CHANNEL_TEST'], 
            os.path.basename(args.test_input_filename))
    else:
        test_path = args.test_input_filename
        
    if 'SM_OUTPUT_DATA_DIR' in os.environ:
        output_path = os.environ['SM_OUTPUT_DATA_DIR']
        output_path = os.path.join(output_path, "output")
    else:
        output_path = args.output_path
        
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    train_ds = PMDataset_torch(
        train_path,
        target_column=args.target_column,
        standardize=True,
        sensor_headers=json.loads(args.sensor_headers))
    test_ds = PMDataset_torch(
        test_path,
        target_column=args.target_column,
        standardize=True,
        sensor_headers=json.loads(args.sensor_headers))
        
    batch_size = args.batch_size
    class_labels = torch.tensor(train_ds.labels)
    ss = StratifiedSampler(class_labels, batch_size)
    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        batch_size,
        num_workers=multiprocessing.cpu_count()-1, 
        shuffle=False, 
        sampler=ss)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=True)
    
    net = Network(num_features=len(json.loads(args.sensor_headers)),
                  fc_hidden_units=json.loads(args.fc_hidden_units),
                  conv_channels=json.loads(args.conv_channels),
                  dropout_strength=args.dropout)
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    critereon = torch.nn.CrossEntropyLoss()
    for e in range(args.epochs):
        train_loss, train_acc, train_auc = run_epoch(net, train_dl, optimizer, 
                                                     critereon, is_train=True)
        test_loss, test_acc, test_auc = run_epoch(net, test_dl, optimizer,
                                                  critereon, is_train=False)
        
        if is_sm_mode:
            print("Epoch: {}".format(e))
            print("Train loss: {:0.4f}".format(train_loss))
            print("Train acc: {:0.4f}".format(train_acc))
            print("Train auc: {:0.4f}".format(train_auc))
            print("Test loss: {:0.4f}".format(test_loss))
            print("Test acc: {:0.4f}".format(test_acc))
            print("Test auc: {:0.4f}".format(test_auc))
        else:
            print("{} train loss: {:0.4f} acc {:0.4f} auc {:0.4f}|".format(e, train_loss, train_acc, train_auc), end="")
            print("test loss {:0.4f} acc {:0.4f} auc {:0.4f}".format(test_loss, test_acc, test_auc))
        
        if e % 20 == 0:
            torch.save(
                {"net": net.state_dict(),
                 "sensor_headers": json.loads(args.sensor_headers),
                 "fc_hidden_units": json.loads(args.fc_hidden_units),
                 "conv_channels": json.loads(args.conv_channels)}, 
                os.path.join(output_path, "net.pth"))
            
if __name__ == "__main__":
    run()
