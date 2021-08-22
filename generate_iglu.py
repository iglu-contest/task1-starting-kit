from argparse import Namespace
import sys
sys.path.append('./python')
from bleu import compute_bleu
from train_and_eval import generate, multinomial_generate, multinomial_generate_seq2seq
from data_loader import CwCDataset
from utils import *
from vocab import load_vocab
import os
import argparse
import torch
import pickle
import pprint
import json
from glob import glob

# from vocab import Vocabulary
# 1. a function named creat_dataloader
# 2. a class Model():
# self.init(): this function will create a model
# self.generate(data=DataIGLU, output='output.txt'): this function will generte the predctions for test set


#################################################
####   You can skip this section   #############
#################################################

def format_prev_utterances(prev_utterances, encoder_vocab):
    prev_utterances = list(map(lambda x: list(
        map(lambda y: encoder_vocab.idx2word[y.item()], x)), prev_utterances))
    prev_utterances = list(map(lambda x: " ".join(x), prev_utterances))[0]
    return prev_utterances


def wordify_ground_truth_utterance(ground_truth_utterance, decoder_vocab):
    """
            Maps from a 2d tensor to a list of tokens and removes eos symbol
    """
    return list(map(lambda x: list(map(lambda y: decoder_vocab.idx2word[y.item()], x)), ground_truth_utterance))[0][:-1]


def format_ground_truth_utterance(ground_truth_utterance, decoder_vocab):
    ground_truth_utterance = wordify_ground_truth_utterance(
        ground_truth_utterance, decoder_vocab)
    ground_truth_utterance = " ".join(ground_truth_utterance)
    return ground_truth_utterance


def format_generated_utterance(generated_utterance):
    generated_utterance = list(map(lambda x: " ".join(x), generated_utterance))
    return generated_utterance


def load_saved_config(model_path=''):
    config_params = None
    config_file = os.path.join(model_path, "config.txt")
    if os.path.isfile(config_file):
        _, config_params = get_config_params(config_file)
    return config_params


def read_iglu_result(iglu_file):
    with open(iglu_file, 'r') as f:
        iglu_content = f.readlines()
    iglu_kv = {}
    for line in iglu_content:
        if len(line.split('@@@')) != 2:
            continue
        (time_stamp, pred) = line.split('@@@')
        iglu_kv[time_stamp] = pred
    return iglu_kv


def build_ref_pred_pair(ref_dict, pred_dict):
    ref_list, pred_list = [], []
    for k, v in ref_dict.items():
        ref_list.append([v])
        if k in pred_dict:
            pred_list.append(pred_dict[k])
        else:
            pred_list.append(' ')
    return ref_list, pred_list
#########################################################
#########################################################

# !!!! this is the first function you have to implement; it will return your own dataloader
# !!!! the testing set will have the same structure and format as the validation and training data
####


def create_dataloader(data_path, split):
    model_path = './saved_model/1626589670356'
    config_params = load_saved_config(model_path)
    gold_config_path = os.path.join(data_path, 'gold-configurations')

    # with open(config_params["encoder_vocab_path"], 'rb') as f:
    # 		encoder_vocab = pickle.load(f)
    # with open(config_params["decoder_vocab_path"], 'rb') as f:
    # 		decoder_vocab = pickle.load(f)

    encoder_vocab = load_vocab('./vocabulary/encoder_vocab.pkl')
    decoder_vocab = load_vocab('./vocabulary/decoder_vocab.pkl')

    test_dataset = CwCDataset(
        model=config_params["model"], split=split, lower=True, dump_dataset=False,
        data_dir=data_path, gold_configs_dir=gold_config_path, vocab_dir=config_params[
            "vocab_dir"],
        encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab, load_dataset=False, transform=None
    )
    test_dataset.set_args(num_prev_utterances=config_params["num_prev_utterances"], blocks_max_weight=config_params["blocks_max_weight"], use_builder_actions=config_params['use_builder_actions'], include_empty_channel=config_params['include_empty_channel'], use_condensed_action_repr=config_params['use_condensed_action_repr'],
                          action_type_sensitive=config_params['action_type_sensitive'], feasible_next_placements=config_params['feasible_next_placements'],  spatial_info_window_size=config_params["spatial_info_window_size"], counters_extra_feasibility_check=config_params["counters_extra_feasibility_check"], use_existing_blocks_counter=config_params["use_existing_blocks_counter"])
    test_dl = test_dataset.get_data_loader(
        batch_size=1, shuffle=False, num_workers=0)
    return test_dl


# This is a class wrapper you have to implement;
# We will create a test entity by running 'model=Model()'
# We will obtain the predictions by running 'model.generate(test_set, output_path)'

class Model():
    def __init__(self):
        # !!!! this function does not accept any extra input
        # you should explicitly feed your model path in your submission here
        self.model_path = "saved_model/1626589670356"
        # self.model_path="/datadrive/model/utterances_and_block_region_counters/20210718/utterances_and_block_region_counters_trainer-1626589670355/1626589670356/"

        # this dict is used to store all your hyper-parameters, you can load them from a file in your submission
        self.config_params = load_saved_config(self.model_path)
        self.load_model(self.model_path)

        # with open(self.config_params["encoder_vocab_path"], 'rb') as f:
        # 	self.encoder_vocab = pickle.load(f)
        # with open(self.config_params["decoder_vocab_path"], 'rb') as f:
        # 	self.decoder_vocab = pickle.load(f)

        self.encoder_vocab = load_vocab('./vocabulary/encoder_vocab.pkl')
        self.decoder_vocab = load_vocab('./vocabulary/decoder_vocab.pkl')

        print("Model has been loaded")

    def load_model(self, model_path=''):
        # you can load the trained models or vocabs in this function
        # you need to make sure files to be loaded do exist in the path

        self.model_type = self.config_params["model"]

        model_files = glob(model_path+"/*-best.pkl")

        self.models = {}
        for model_file in model_files:
            with open(model_file, 'rb') as f:
                if not torch.cuda.is_available():
                    model = torch.load(f, map_location="cpu")
                else:
                    model = torch.load(f)
                if "flatten_parameters" in dir(model):
                    # print(dir(model))
                    model.flatten_parameters()  # TODO: flatten for all sub-modules recursively
                if "encoder" in model_file:
                    self.models["encoder"] = model
                elif "decoder" in model_file:
                    self.models["decoder"] = model

    def generate(self, test_set, output_path):
        beam_size = 10
        max_decoding_length = 30
        gamma = 0.8
        init_args = Namespace(set_decoder_hidden=self.config_params['set_decoder_hidden'],
                              concatenate_decoder_inputs=self.config_params['concatenate_decoder_inputs'],
                              concatenate_decoder_hidden=self.config_params['concatenate_decoder_hidden'],
                              decoder_input_concat_size=self.config_params['decoder_input_concat_size'],
                              decoder_hidden_concat_size=self.config_params['decoder_hidden_concat_size'],
                              advance_decoder_t0=self.config_params['advance_decoder_t0'])

        # beam search decoding
        generated_utterances_, to_print = generate(
            self.models["encoder"], self.models["decoder"],
            test_set, self.decoder_vocab,
            beam_size=beam_size, max_length=max_decoding_length, args=init_args,
            development_mode=False, gamma=gamma
        )

        def format(output_obj):
            prev_utterances = format_prev_utterances(
                output_obj["prev_utterances"], self.encoder_vocab)
            ground_truth_utterance = format_ground_truth_utterance(
                output_obj["ground_truth_utterance"], self.decoder_vocab)
            generated_utterance = format_generated_utterance(
                output_obj["generated_utterance"])

            return {
                "prev_utterances": prev_utterances,
                "ground_truth_utterance": ground_truth_utterance,
                "generated_utterance": generated_utterance,
                "json_id": output_obj["json_id"],
                "sample_id": output_obj["sample_id"],
                "time_stamp": output_obj["time_stamp"]
            }

        with open(output_path, 'w') as f2:
            for dia in generated_utterances_:
                generated_one = format(dia)['generated_utterance'][0]
                f2.write(dia['time_stamp'] +
                         '       @@@       ' + generated_one + '\n')

        # predict_path = output_path.replace('.txt', '_pred.txt')
        # ref_path = output_path.replace('.txt', '_ref.txt')
        # predict_file = read_iglu_result(predict_path)
        # ref_file = read_iglu_result(ref_path)
        # reference_corpus, pred_corpus = build_ref_pred_pair(ref_file, predict_file)

        # bleu_1_results = compute_bleu(reference_corpus, pred_corpus, max_order=1, smooth=False)
        # bleu_2_results = compute_bleu(reference_corpus, pred_corpus, max_order=2, smooth=False)
        # bleu_3_results = compute_bleu(reference_corpus, pred_corpus, max_order=3, smooth=False)
        # bleu_4_results = compute_bleu(reference_corpus, pred_corpus, max_order=4, smooth=False)

        # print(bleu_4_results[0])


def main():
    initialize_rngs(2021)

    data_path = '/datadrive/uiuc_warmup/'
    split = 'test'
    model = Model()

    test_dataset = create_dataloader(data_path, split)

    output_path = os.path.join('.', split+'.txt')
    model.generate(test_dataset, output_path)


if __name__ == '__main__':
    main()
