import numpy as np
import argparse
import os
# import imp
import re
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

from utils import utils
from utils.readers import DecompensationReader
from utils.preprocessing import Discretizer, Normalizer
from utils import metrics
from utils import common_utils
from model import StageAwareTransformer

def parse_arguments(parser):
    parser.add_argument('--test_mode', type=int, default=0, help='Test SA-CRNN on MIMIC-III dataset')
    parser.add_argument('--data_path', type=str,  default='./data/',metavar='<data_path>', help='The path to the MIMIC-III data directory')
    parser.add_argument('--file_name', type=str, default='SAT', metavar='<data_path>', help='File name to save model')
    parser.add_argument('--small_part', type=int, default=0, help='Use part of training data')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learing rate')

    parser.add_argument('--input_dim', type=int, default=76, help='Dimension of visit record data')
    parser.add_argument('--rnn_dim', type=int, default=384, help='Dimension of hidden units in RNN')
    parser.add_argument('--output_dim', type=int, default=1, help='Dimension of prediction target')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--dropconnect_rate', type=float, default=0.5, help='Dropout rate in RNN')
    parser.add_argument('--dropres_rate', type=float, default=0.3, help='Dropout rate in residue connection')
    parser.add_argument('--K', type=int, default=10, help='Value of hyper-parameter K')
    parser.add_argument('--chunk_level', type=int, default=3, help='Value of hyper-parameter K')

    #####################
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer encoder layers')
    parser.add_argument('--ff_dim', type=int, default=512, help='Feed-forward hidden dimension')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length for positional encoding')
    #######################
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    print(torch.cuda.is_available())
    if args.test_mode == 1:
        print('Preparing test data ... ')

        train_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
            args.data_path, 'train'), listfile=os.path.join(args.data_path, 'train_listfile.csv'), small_part=True)
        discretizer = Discretizer(timestep=1.0, store_masks=True,
                                impute_strategy='previous', start_time='zero')

        discretizer_header = discretizer.transform(train_data_loader._data["X"][0])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)
        normalizer_state = 'decomp_normalizer'
        normalizer_state = os.path.join(os.path.dirname(args.data_path), normalizer_state)
        normalizer.load_params(normalizer_state)

        test_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data_path, 'test'),
                                                                        listfile=os.path.join(args.data_path, 'test_listfile.csv'), small_part=args.small_part)
        test_data_gen = utils.BatchGenDeepSupervision(test_data_loader, discretizer,
                                                    normalizer, args.batch_size,
                                                    shuffle=False, return_names=True)

        print('Constructing model ... ')
        device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
        print("available device: {}".format(device))

        model = StageAwareTransformer(
            input_dim=76 + 17,
            hidden_dim=384,
            conv_size=10,
            output_dim=1,
            levels=3,
            dropconnect=0.3,
            dropout=0.3,
            dropres=0.3,
            num_heads=8,
            num_layers=2,
            ff_dim=512,
            max_len=512,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        checkpoint = torch.load('./saved_weights/StageAwareTransformer')
        save_chunk = checkpoint['chunk']
        print("last saved model is in chunk {}".format(save_chunk))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.eval()
        with torch.no_grad():
            cur_test_loss = []
            test_true = []
            test_pred = []
            
            for each_batch in range(test_data_gen.steps):
                test_data = next(test_data_gen)
                test_name = test_data['names']
                test_data = test_data['data']

                test_x = torch.tensor(test_data[0][0], dtype=torch.float32).to(device)
                test_mask = torch.tensor(test_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
                test_y = torch.tensor(test_data[1], dtype=torch.float32).to(device)
                tmp = torch.zeros(test_x.size(0),17, dtype=torch.float32).to(device)
                test_interval = torch.zeros((test_x.size(0),test_x.size(1),17), dtype=torch.float32).to(device)

                for i in range(test_x.size(1)):
                    cur_ind = test_x[:,i,-17:]
                    tmp+=(cur_ind == 0).float()
                    test_interval[:, i, :] = cur_ind * tmp
                    tmp[cur_ind==1] = 0  
                
                if test_mask.size()[1] > 400:
                    test_x = test_x[:, :400, :]
                    test_mask = test_mask[:, :400, :]
                    test_y = test_y[:, :400, :]
                    test_interval = test_interval[:, :400, :]
                
                test_x = torch.cat((test_x, test_interval), dim=-1)
                test_time = torch.ones((test_x.size(0), test_x.size(1)), dtype=torch.float32).to(device)
                
                test_output, test_dis = model(test_x, test_time, device)
                masked_test_output = test_output * test_mask

                test_loss = test_y * torch.log(masked_test_output + 1e-7) + (1 - test_y) * torch.log(1 - masked_test_output + 1e-7)
                test_loss = torch.sum(test_loss, dim=1) / torch.sum(test_mask, dim=1)
                test_loss = torch.neg(torch.sum(test_loss))
                cur_test_loss.append(test_loss.cpu().detach().numpy()) 
                
                for m, t, p in zip(test_mask.cpu().numpy().flatten(), test_y.cpu().numpy().flatten(), test_output.cpu().detach().numpy().flatten()):
                    if np.equal(m, 1):
                        test_true.append(t)
                        test_pred.append(p)
            
            print('Test loss = %.4f'%(np.mean(np.array(cur_test_loss))))
            print('\n')
            test_pred = np.array(test_pred)
            test_pred = np.stack([1 - test_pred, test_pred], axis=1)
            test_ret = metrics.print_metrics_binary(test_true, test_pred)
            
            from sklearn.metrics import roc_curve, auc
            import matplotlib.pyplot as plt
            
            test_pred_prob = test_pred[:, 1]
            fpr, tpr, _ = roc_curve(test_true, test_pred_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(5, 5))
            plt.fill_between(fpr, tpr, alpha=0.15, color='darkorange')
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'Transformer (AUROC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=1.5,
                     linestyle='--', label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=13)
            plt.legend(loc='lower right', fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig('roc_curve.pdf', format='pdf', dpi=300)
            print(f'AUROC = {roc_auc:.4f}')


    else:
        ''' Prepare training data'''
        print('Preparing training data ... ')
        train_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
            args.data_path, 'train'), listfile=os.path.join(args.data_path, 'train_listfile.csv'), small_part=args.small_part)
        val_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(
            args.data_path, 'train'), listfile=os.path.join(args.data_path, 'val_listfile.csv'), small_part=args.small_part)
        discretizer = Discretizer(timestep=1.0, store_masks=True,
                                impute_strategy='previous', start_time='zero')

        discretizer_header = discretizer.transform(train_data_loader._data["X"][0])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)
        normalizer_state = 'decomp_normalizer'
        normalizer_state = os.path.join(os.path.dirname(args.data_path), normalizer_state)
        normalizer.load_params(normalizer_state)

        train_data_gen = utils.BatchGenDeepSupervision(train_data_loader, discretizer,
                                                        normalizer, args.batch_size, shuffle=True, return_names=True)
        val_data_gen = utils.BatchGenDeepSupervision(val_data_loader, discretizer,
                                                    normalizer, args.batch_size, shuffle=False, return_names=True)

        '''Model structure'''
        print('Constructing model ... ')
        device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
        print("available device: {}".format(device))

        model = StageAwareTransformer(
            input_dim=args.input_dim + 17,
            hidden_dim=args.rnn_dim,
            conv_size=args.K,
            output_dim=args.output_dim,
            levels=args.chunk_level,
            dropconnect=args.dropconnect_rate,
            dropout=args.dropout_rate,
            dropres=args.dropres_rate,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            max_len=args.max_len,
        ).to(device)



        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        '''Train phase'''
        print('Start training ... ')

        train_loss = []
        val_loss = []
        batch_loss = []
        max_auprc = 0

        file_name = './saved_weights/'+args.file_name
        for each_chunk in range(args.epochs):
            cur_batch_loss = []
            model.train()
            for each_batch in range(train_data_gen.steps):
                batch_data = next(train_data_gen)
                batch_name = batch_data['names']
                batch_data = batch_data['data']

                batch_x = torch.tensor(batch_data[0][0], dtype=torch.float32).to(device)
                batch_mask = torch.tensor(batch_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
                batch_y = torch.tensor(batch_data[1], dtype=torch.float32).to(device)
                tmp = torch.zeros(batch_x.size(0),17, dtype=torch.float32).to(device)
                batch_interval = torch.zeros((batch_x.size(0),batch_x.size(1),17), dtype=torch.float32).to(device)
                
                for i in range(batch_x.size(1)):
                    cur_ind = batch_x[:,i,-17:]
                    tmp+=(cur_ind == 0).float()
                    batch_interval[:, i, :] = cur_ind * tmp
                    tmp[cur_ind==1] = 0        
                
                if batch_mask.size()[1] > 400:
                    batch_x = batch_x[:, :400, :]
                    batch_mask = batch_mask[:, :400, :]
                    batch_y = batch_y[:, :400, :]
                    batch_interval = batch_interval[:, :400, :]

                batch_x = torch.cat((batch_x, batch_interval), dim=-1)
                batch_time = torch.ones((batch_x.size(0), batch_x.size(1)), dtype=torch.float32).to(device)

                optimizer.zero_grad()
                cur_output, _ = model(batch_x, batch_time, device)
                masked_output = cur_output * batch_mask 
                loss = batch_y * torch.log(masked_output + 1e-7) + (1 - batch_y) * torch.log(1 - masked_output + 1e-7)
                loss = torch.sum(loss, dim=1) / torch.sum(batch_mask, dim=1)
                loss = torch.neg(torch.sum(loss))
                cur_batch_loss.append(loss.cpu().detach().numpy())

                loss.backward()
                optimizer.step()
                
                if each_batch % 50 == 0:
                    print('Chunk %d, Batch %d: Loss = %.4f'%(each_chunk, each_batch, cur_batch_loss[-1]))

            batch_loss.append(cur_batch_loss)
            train_loss.append(np.mean(np.array(cur_batch_loss)))
            
            print("\n==>Predicting on validation")
            with torch.no_grad():
                model.eval()
                cur_val_loss = []
                valid_true = []
                valid_pred = []
                for each_batch in range(val_data_gen.steps):
                    valid_data = next(val_data_gen)
                    valid_name = valid_data['names']
                    valid_data = valid_data['data']
                    
                    valid_x = torch.tensor(valid_data[0][0], dtype=torch.float32).to(device)
                    valid_mask = torch.tensor(valid_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
                    valid_y = torch.tensor(valid_data[1], dtype=torch.float32).to(device)
                    tmp = torch.zeros(valid_x.size(0),17, dtype=torch.float32).to(device)
                    valid_interval = torch.zeros((valid_x.size(0),valid_x.size(1),17), dtype=torch.float32).to(device)
                    
                    for i in range(valid_x.size(1)):
                        cur_ind = valid_x[:,i,-17:]
                        tmp+=(cur_ind == 0).float()
                        valid_interval[:, i, :] = cur_ind * tmp
                        tmp[cur_ind==1] = 0  
                    
                    if valid_mask.size()[1] > 400:
                        valid_x = valid_x[:, :400, :]
                        valid_mask = valid_mask[:, :400, :]
                        valid_y = valid_y[:, :400, :]
                        valid_interval = valid_interval[:, :400, :]
                    
                    valid_x = torch.cat((valid_x, valid_interval), dim=-1)
                    valid_time = torch.ones((valid_x.size(0), valid_x.size(1)), dtype=torch.float32).to(device)
                    
                    valid_output, valid_dis = model(valid_x, valid_time, device)
                    masked_valid_output = valid_output * valid_mask

                    valid_loss = valid_y * torch.log(masked_valid_output + 1e-7) + (1 - valid_y) * torch.log(1 - masked_valid_output + 1e-7)
                    valid_loss = torch.sum(valid_loss, dim=1) / torch.sum(valid_mask, dim=1)
                    valid_loss = torch.neg(torch.sum(valid_loss))
                    cur_val_loss.append(valid_loss.cpu().detach().numpy())

                    for m, t, p in zip(valid_mask.cpu().numpy().flatten(), valid_y.cpu().numpy().flatten(), valid_output.cpu().detach().numpy().flatten()):
                        if np.equal(m, 1):
                            valid_true.append(t)
                            valid_pred.append(p)

                val_loss.append(np.mean(np.array(cur_val_loss)))
                print('Valid loss = %.4f'%(val_loss[-1]))
                print('\n')
                valid_pred = np.array(valid_pred)
                valid_pred = np.stack([1 - valid_pred, valid_pred], axis=1)
                ret = metrics.print_metrics_binary(valid_true, valid_pred)
                print()

                cur_auprc = ret['auprc']
                if cur_auprc > max_auprc:
                    max_auprc = cur_auprc
                    state = {
                        'net': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'chunk': each_chunk
                    }
                    torch.save(state, file_name)
                    print('\n------------ Save best model ------------\n')


        '''Evaluate phase'''
        print('Testing model ... ')

        checkpoint = torch.load(file_name)
        save_chunk = checkpoint['chunk']
        print("last saved model is in chunk {}".format(save_chunk))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.eval()

        test_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data_path, 'test'),
                                                                        listfile=os.path.join(args.data_path, 'test_listfile.csv'), small_part=args.small_part)
        test_data_gen = utils.BatchGenDeepSupervision(test_data_loader, discretizer,
                                                    normalizer, args.batch_size,
                                                    shuffle=False, return_names=True)

        with torch.no_grad():
            torch.manual_seed(RANDOM_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(RANDOM_SEED)
        
            cur_test_loss = []
            test_true = []
            test_pred = []
            
            for each_batch in range(test_data_gen.steps):
                test_data = next(test_data_gen)
                test_name = test_data['names']
                test_data = test_data['data']

                test_x = torch.tensor(test_data[0][0], dtype=torch.float32).to(device)
                test_mask = torch.tensor(test_data[0][1], dtype=torch.float32).unsqueeze(-1).to(device)
                test_y = torch.tensor(test_data[1], dtype=torch.float32).to(device)
                tmp = torch.zeros(test_x.size(0),17, dtype=torch.float32).to(device)
                test_interval = torch.zeros((test_x.size(0),test_x.size(1),17), dtype=torch.float32).to(device)

                for i in range(test_x.size(1)):
                    cur_ind = test_x[:,i,-17:]
                    tmp+=(cur_ind == 0).float()
                    test_interval[:, i, :] = cur_ind * tmp
                    tmp[cur_ind==1] = 0  
                
                if test_mask.size()[1] > 400:
                    test_x = test_x[:, :400, :]
                    test_mask = test_mask[:, :400, :]
                    test_y = test_y[:, :400, :]
                    test_interval = test_interval[:, :400, :]
                
                test_x = torch.cat((test_x, test_interval), dim=-1)
                test_time = torch.ones((test_x.size(0), test_x.size(1)), dtype=torch.float32).to(device)
                
                test_output, test_dis = model(test_x, test_time, device)
                masked_test_output = test_output * test_mask

                test_loss = test_y * torch.log(masked_test_output + 1e-7) + (1 - test_y) * torch.log(1 - masked_test_output + 1e-7)
                test_loss = torch.sum(test_loss, dim=1) / torch.sum(test_mask, dim=1)
                test_loss = torch.neg(torch.sum(test_loss))
                cur_test_loss.append(test_loss.cpu().detach().numpy()) 
                
                for m, t, p in zip(test_mask.cpu().numpy().flatten(), test_y.cpu().numpy().flatten(), test_output.cpu().detach().numpy().flatten()):
                    if np.equal(m, 1):
                        test_true.append(t)
                        test_pred.append(p)
            
            print('Test loss = %.4f'%(np.mean(np.array(cur_test_loss))))
            print('\n')
            test_pred = np.array(test_pred)
            test_pred = np.stack([1 - test_pred, test_pred], axis=1)
            test_ret = metrics.print_metrics_binary(test_true, test_pred)
