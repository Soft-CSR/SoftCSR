# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import torch
import argparse
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
from datasets import RecWithContrastiveLearningDataset

from trainers import SoftCSRTrainer
from models import  OfflineItemSimilarity, OnlineItemSimilarity, SASRecGRU4per
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed,get_local_time

def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")

def main():
    parser = argparse.ArgumentParser()
    # system
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')

    #parser.add_argument('--do_eval', type=int, default='0')
    parser.add_argument("--gpu_id", type=str, default="2", help="gpu_id")

    # data augmentation args
    parser.add_argument('--noise_ratio', default=0.0, type=float, \
                        help="percentage of negative interactions in a sequence - robustness analysis")
    parser.add_argument('--training_data_ratio', default=1.0, type=float, \
                        help="percentage of training samples used for training - robustness analysis")
    parser.add_argument('--augment_threshold', default=-1, type=int, \
                        help="control augmentations on short and long sequences.\
                        default:-1, means all augmentations types are allowed for all sequences.")
    parser.add_argument('--similarity_model_name', default='ItemCF_IUF', type=str, \
                        help="Method to generate item similarity score. choices: \
                        Random, ItemCF, ItemCF_IUF(Inverse user frequency), Item2Vec, LightGCN")
    parser.add_argument("--augmentation_warm_up_epoches", type=float, default=160, \
                        help="number of epochs to switch from \
                        memory-based similarity model to \
                        hybrid similarity model.")
    parser.add_argument('--base_augment_type', default='random', type=str, \
                        help="default data augmentation types. Chosen from: \
                        mask, crop, reorder, substitute, insert, random, \
                        combinatorial_enumerate (for multi-view).")

    parser.add_argument('--augment_type_for_short', default='SIMRC', type=str, \
                        help="data augmentation types for short sequences. Chosen from:SIMRC.")

    parser.add_argument("--tao", type=float, default=0.2, help="crop ratio for crop operator")
    parser.add_argument("--gamma", type=float, default=0.7, help="mask ratio for mask operator")
    parser.add_argument("--beta", type=float, default=0.2, help="reorder ratio for reorder operator") 
    parser.add_argument("--substitute_rate", type=float, default=0.1, \
                        help="substitute ratio for substitute operator")
    parser.add_argument("--insert_rate", type=float, default=0.4, \
                        help="insert ratio for insert operator")
    parser.add_argument("--max_insert_num_per_pos", type=int, default=1, \
                        help="maximum insert items per position for insert operator - not studied")

    ## contrastive learning task args
    parser.add_argument('--temperature', default= 1.0, type=float,
                        help='softmax temperature (default:  1.0) - not studied.')
    parser.add_argument('--n_views', default=2, type=int, metavar='N',
                        help='Number of augmented data for each sequence - not studied.')

    # model args
    parser.add_argument("--model_name", default='SoftCSR', type=str)
    parser.add_argument("--base_model_name",'-b', default='SASRecGRU4per', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--hidden_dropout_prob_gru", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--loss_type', default="CE", type=str)
    parser.add_argument("--embedding_size", type=int, default=64, help="embedding_size of SASRec model")

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=1, type=int)  # 1
    parser.add_argument("--rec_weight", type=float, default=1.0, \
                        help="weight of contrastive learning task")
    # outer minimize
    parser.add_argument("--alpha", type=float, default=0.1, help="weight of contrastive learning task")
    parser.add_argument("--alpha_1", type=float, default=0.001, help="weight of contrastive learning task")
    parser.add_argument("--alpha_2", type=float, default=0.001, help="weight of contrastive learning task")
    # inner maximize
    parser.add_argument("--beta_0", type=float, default=0.0001, \
                        help="weight of contrastive learning task replace beta in paper")
    parser.add_argument("--beta_1", type=float, default=0.001, \
                        help="weight of contrastive learning task")
    parser.add_argument("--beta_2", type=float, default=0.001, \
                        help="weight of contrastive learning task")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--adv_step", type=int, default=5, help="train step for the perturbation iterations")
    parser.add_argument("--eta", type=float, default=0.5, help="eta in method_sequence ")
    parser.add_argument("--norm_type", default='l2', type=str, help='normal type')
    # perturbation
    parser.add_argument("--epsilon_sequence", type=float, default=0.1, help="epsilon in method_sequence ")
    parser.add_argument("--epsilon_item", type=float, default=0.1, help="epsilon in method_item ")
    parser.add_argument("--epsilon_gru", type=float, default=0.1, help="epsilon in method_gru ")

    # add perturbation via method sequence item gru
    parser.add_argument("--method_sequence", default='No', type=str, help='whether add perturbation via method 3-1')
    parser.add_argument("--method_item", default='No', type=str, help='whether add perturbation via method 3-2 ')
    parser.add_argument("--method_gru_theta_update", default='Yes', type=str, help='whether add perturbation via method 3-3-plus')
    parser.add_argument('--clip', type=float, default=1.0,help='gradient clipping')

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + '.txt'

    user_seq, max_item, valid_rating_matrix, test_rating_matrix = \
        get_user_seqs(args.data_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    nowtime = get_local_time()
    # save model args

    args_str = f'{args.model_name}-{args.base_model_name}-{args.data_name}-{nowtime}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    
    show_args_info(args)

    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # -----------   pre-computation for item similarity   ------------ #
    args.similarity_model_path = os.path.join(args.data_dir,\
                            args.data_name+'_'+args.similarity_model_name+'_similarity.pkl')

    offline_similarity_model = OfflineItemSimilarity(data_file=args.data_file,
                            similarity_path=args.similarity_model_path,
                            model_name=args.similarity_model_name,
                            dataset_name=args.data_name)

    args.offline_similarity_model = offline_similarity_model

    # -----------   online based on shared item embedding for item similarity --------- #
    online_similarity_model = OnlineItemSimilarity(item_size=args.item_size)
    args.online_similarity_model = online_similarity_model

    # training data for node classification
    train_dataset = RecWithContrastiveLearningDataset(args, 
                                    user_seq[:int(len(user_seq)*args.training_data_ratio)], \
                                    data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)


    model = SASRecGRU4per(args=args)

    trainer = SoftCSRTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        print(f'Train SoftCSR')
        early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on NDCG@20
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        trainer.args.train_matrix = test_rating_matrix
        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
    return result_info


if __name__ == '__main__':
    main()
