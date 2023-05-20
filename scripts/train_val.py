import pandas as pd
import torch
from dataset.AccountingDataset import AcountingDataset
from models.BERT import BERT
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import random
import transformers
from utils.utils import AverageMeter
import os


def main():

    # Parameters
    data_excel_path = 'E:\WS\AccountingClassifier\data\General_ledger.xlsx'
    max_length = 100
    validate_percentage = 0.1  # This amount of data is used for validation
    num_epochs = 200
    validate_iteration_frequency = 200
    train_average_meter_reset_frequency = 200
    save_epoch_frequency = 10
    save_dir = '../ckpt'
    batch_size = 320


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    # Datasets and loaders setup

    # First, go through and filter data because there are empty fields. We ignore any data with empty fields.
    data_valid_indices = []
    valid_descriptions_dict = {}
    categories_and_count_dict = {}

    df = pd.read_excel(data_excel_path)
    #print(df)

    for i in range(0, len(df. index)):
        row = df.iloc[i]  # a series
        category = row['Category']
        if category not in categories_and_count_dict:
            categories_and_count_dict[category] = 1
        else:
            categories_and_count_dict[category] += 1
        if not pd.isna(category):
            description = row['Memo/Description']
            if description not in valid_descriptions_dict:
                data_valid_indices.append(i)
                valid_descriptions_dict[description] = 1


    random.shuffle(data_valid_indices)
    cut_off_index = int(len(data_valid_indices) * (1-validate_percentage))
    valid_indices_train = data_valid_indices[:cut_off_index]
    valid_indices_val = data_valid_indices[cut_off_index:]

    valid_categories = [k.strip() for k in categories_and_count_dict.keys() if not pd.isna(k)]
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = AcountingDataset(df, valid_indices_train, tokenizer, valid_categories, max_length=max_length)
    print("length of train dataset: %d" % len(train_dataset))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = AcountingDataset(df, valid_indices_val, tokenizer, valid_categories, max_length=max_length)
    print("length of validation dataset: %d" % len(val_dataset))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


    # Model setup
    model = BERT(len(valid_categories)).cuda()
    for param in model.bert_model.parameters():  # Finetune the last linear layer
        param.requires_grad = False

    # Loss functions
    criterion_BCE = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    args = {}
    args['model'] = model
    args['train_dataloader'] = train_dataloader
    args['val_dataloader'] = val_dataloader
    args['criterion'] = criterion_BCE
    args['optimizer'] = optimizer
    args['total_iterations'] = 0
    args['train_iterations'] = 0
    args['val_iterations'] = 0
    args['epoch'] = 0
    args['train_avg_BCE_loss'] = AverageMeter()
    args['val_avg_BCE_loss'] = AverageMeter()
    args['validate_iteration_frequency'] = validate_iteration_frequency
    args['train_average_meter_reset_frequency'] = train_average_meter_reset_frequency
    args['val_epoch_total_num_samples'] = 0
    args['val_epoch_total_num_correct_samples'] = 0


    for epoch in range(num_epochs):
        args = train_epoch(args)
        args = val_epoch(args)
        args['epoch'] += 1
        if args['epoch'] != 0 and args['epoch'] % save_epoch_frequency == 0:
            save_path = os.path.join(save_dir, 'epoch_%05d.pth' % args['epoch'])
            torch.save({
                'epoch': epoch,
                'total_iteration': args['total_iterations'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)


def iterate_common(data_dict, args, mode='train'):

    ids = data_dict['ids'].cuda()  # BS, max_length
    mask = data_dict['mask'].cuda()  # BS, max_length
    token_type_ids = data_dict['token_type_ids'].cuda()  # BS, max_length
    target = data_dict['target'].cuda()  # BS, num_classes

    optimizer = args['optimizer']
    model = args['model']
    criterion = args['criterion']

    if mode == 'train':
        optimizer.zero_grad()

    output = model(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids)  # BS, num_classes
    target = target.type_as(output)

    loss = criterion(output, target)
    if mode == 'train':
        args['train_avg_BCE_loss'].update(loss.item())
    else:
        args['val_avg_BCE_loss'].update(loss.item())

    if mode == 'train':
        loss.backward()
        optimizer.step()
        args['train_iterations'] += 1
        args['total_iterations'] += 1
    else:
        args['val_iterations'] += 1


    output = torch.argmax(output, dim=1)  # BS, 1
    target = torch.argmax(target, dim=1)
    num_correct = torch.sum(output==target)
    num_samples = ids.size(0)
    accuracy_batch = num_correct.item() / num_samples

    if mode == 'val':
        args['val_epoch_total_num_samples'] += num_samples
        args['val_epoch_total_num_correct_samples'] += num_correct.item()

    text_to_print = '[Epoch %d]' % args['epoch']
    if mode == 'train':
        text_to_print += '[Train]'
    else:
        text_to_print += '[Val]'
    if mode == 'train':
        text_to_print += '[%d/%d]' % (args['train_iterations'], len(args['train_dataloader']))
    else:
        text_to_print += '[%d/%d]' % (args['val_iterations'], len(args['val_dataloader']))
    text_to_print += '[Loss_BCE %.6f]' % loss.item()
    if mode == 'train':
        text_to_print += '[Average_Loss_BCE %.6f]' % args['train_avg_BCE_loss'].avg()
    else:
        text_to_print += '[Average_Loss_BCE %.6f]' % args['val_avg_BCE_loss'].avg()
    text_to_print += '[batch accuracy: %.3f]' % accuracy_batch

    print(text_to_print)
    return args


def train_epoch(args):

    model = args['model']
    train_dataloader = args['train_dataloader']
    epoch = args['epoch']
    validate_iteration_frequency = args['validate_iteration_frequency']
    train_average_meter_reset_frequency = args['train_average_meter_reset_frequency']

    args['train_iterations'] = 0
    args['train_avg_BCE_loss'] = AverageMeter()

    model.train()
    print("Starting training epoch %d" % epoch)
    for idx, data_dict in enumerate(train_dataloader):
        args = iterate_common(data_dict, args, mode='train')
        if args['train_iterations'] != 0 and args['train_iterations'] % train_average_meter_reset_frequency == 0:
            args['train_avg_BCE_loss'] = AverageMeter()
        if args['train_iterations'] != 0 and args['train_iterations'] % validate_iteration_frequency == 0:
            val_epoch(args)

    return args


def val_epoch(args):

    model = args['model']
    val_dataloader = args['val_dataloader']
    epoch = args['epoch']

    args['val_iterations'] = 0
    args['val_epoch_total_num_samples'] = 0
    args['val_epoch_total_num_correct_samples'] = 0
    args['val_avg_BCE_loss'] = AverageMeter()

    model.eval()
    print("Starting Evaluation epoch %d" % epoch)
    for idx, data_dict in enumerate(val_dataloader):
        args = iterate_common(data_dict, args, mode='val')
    model.train()

    print("Evaluation for epoch %d complete. Total accuracy: %.3f" %
          (epoch, args['val_epoch_total_num_correct_samples'] / args['val_epoch_total_num_samples']))

    return args


if __name__ == '__main__':
    main()
