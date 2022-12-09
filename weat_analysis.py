import os
from os.path import join
import json
import argparse
from copy import deepcopy 

import responsibly
from responsibly.we import weat
from gensim import downloader
from gensim.models import KeyedVectors

import spacy
nlp_spacy = spacy.load("en_core_web_trf")
from subject_verb_object_extract_test import findSVOs, nlp

parser = argparse.ArgumentParser(description='weat_analysis')
parser.add_argument('--orig_att', default='dark')
parser.add_argument('--dataset', default='conceptual-captions')
parser.add_argument('--with_pvalue', action='store_true', default=False)

args = parser.parse_args()

if __name__ == "__main__":
    attribute = args.orig_att
    dataset = args.dataset
    modes = ["mod", "orig"]

    if not os.path.exists(f"captions/val2014_{attribute}_{dataset}_verbs.json") or not os.path.exists(f"captions/val2014_{attribute}_{dataset}_subjects.json"): 
        ######## Loading Captions #########
        print("######## Loading Captions #########")
        cap_dir = "captions"

        cap_dict = {}
        for mod in modes:
            file = f"val2014_{attribute}_{mod}_{dataset}.json"
            # print(file)
            with open(join(cap_dir, file)) as json_file:
                caps = json.load(json_file)
                caps = list(caps.values())
            
            if attribute not in cap_dict.keys():
                cap_dict[attribute] = {}
            
            if dataset not in cap_dict[attribute]:
                cap_dict[attribute][dataset] = {}
            
            cap_dict[attribute][dataset][mod] = caps
        
        ######## Extract Subjects and Verbs from captions ########
        print("######## Extract Subjects and Verbs from captions ########")
        svo_dict2 = deepcopy(cap_dict)

        for mod in cap_dict[attribute][dataset].keys():
            caps = cap_dict[attribute][dataset][mod]
            
            svos2 = []
            for cap in caps:
                tokens2 = nlp(cap)
                svo2 = findSVOs(tokens2)
                svos2.append(svo2)
            svo_dict2[attribute][dataset][mod] = svos2

        with open(f"captions/val2014_{attribute}_{dataset}_svos.json", "w") as outfile:
            json.dump(svo_dict2, outfile)

        verb_dict = deepcopy(cap_dict)
        for i, mod in enumerate(cap_dict[attribute][dataset].keys()):
            caps = cap_dict[attribute][dataset][mod]

            verbs = []
            for cap in caps:
                doc_cap = nlp_spacy(cap)
                for token in doc_cap:
                    if token.pos_ == "VERB": verbs.append(token.lemma_)
            
            verb_dict[attribute][dataset][mod] = verbs
        
        with open(f"captions/val2014_{attribute}_{dataset}_verbs.json", "w") as outfile:
            json.dump(verb_dict, outfile)

        subject_dict = deepcopy(svo_dict2)
        for i, mod in enumerate(svo_dict2[attribute][dataset].keys()):
            svos = svo_dict2[attribute][dataset][mod]
            subjects = [svo[0][0] for svo in svos if len(svo) != 0]
            subjects = " ".join(subjects).split(" ")

            common_subj_list = ["person", "man", "woman", "boy", "girl", "people",
                                "group", "in", "shirt", "black", "left", "photo",
                                "layer", "ice", "the", "area",
                                "I", "a", "i", "this", "of", "my", "what", "4k",
                                "t", "-", ",", ".", "on", "his"]
            subjects = [subject for subject in subjects if subject not in common_subj_list]
            subject_dict[attribute][dataset][mod] = subjects
        
        with open(f"captions/val2014_{attribute}_{dataset}_subjects.json", "w") as outfile:
            json.dump(subject_dict, outfile)

    ######## WEAT Calculation #########
    print("######## WEAT Calculation #########")
    with open(f"captions/val2014_{attribute}_{dataset}_verbs.json") as json_file:
        verb_dict = json.load(json_file)

    with open(f"captions/val2014_{attribute}_{dataset}_subjects.json") as json_file:
        subject_dict = json.load(json_file)

    w2v_path = downloader.load('word2vec-google-news-300', return_path=True)
    print(w2v_path)

    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    print("Word2vec embeddings loaded")

    dark_attributes = { 'name': 'dark_attributes',
                        'words': ["black", "brown", "dark", "colored", "african", "minority", "immigrant", "foreigner"]}
    light_attributes = {'name': 'light_attributes',
                        'words': ["white", "pale", "light", "caucasian", "european", "majority", "citizen", "national"]}

    light_verb_words = list(set([verb for verb in verb_dict["dark"]["conceptual-captions"]["mod"] if verb in w2v_model.vocab]))
    light_verbs = { 'name': 'light_verbs',
                    'words': light_verb_words}
    dark_verb_words = list(set([verb for verb in verb_dict["dark"]["conceptual-captions"]["orig"] if verb in w2v_model.vocab]))
    dark_verb_words = dark_verb_words + ["work"]*(len(light_verb_words) - len(dark_verb_words))
    dark_verbs = {  'name': 'dark_verbs',
                    'words': dark_verb_words}

    light_subject_words = list(set([subject for subject in subject_dict["dark"]["conceptual-captions"]["mod"] if subject in w2v_model.vocab]))
    light_subjects = {  'name': 'light_subjects',
                        'words': light_subject_words}
    dark_subject_words = list(set([subject for subject in subject_dict["dark"]["conceptual-captions"]["orig"] if subject in w2v_model.vocab]))
    dark_subject_words = dark_subject_words + [dark_subject_words[0]]*(len(light_subject_words) - len(dark_subject_words))
    dark_subjects = {   'name': 'dark_subjects',
                        'words': dark_subject_words}


    weat_result = weat.calc_single_weat(w2v_model,
                                        first_target = dark_verbs,
                                        second_target = light_verbs,
                                        first_attribute  = dark_attributes,
                                        second_attribute = light_attributes,
                                        with_pvalue = args.with_pvalue)
    print(weat_result)

    weat_result = weat.calc_single_weat(w2v_model,
                                        first_target = dark_subjects,
                                        second_target = light_subjects,
                                        first_attribute  = dark_attributes,
                                        second_attribute = light_attributes,
                                        with_pvalue = args.with_pvalue)
    print(weat_result)

    weat_result = weat.calc_weat_pleasant_unpleasant_attribute(w2v_model,
                                                    first_target = dark_verbs,
                                                    second_target = light_verbs,
                                                    with_pvalue = args.with_pvalue)
    print(weat_result)

    weat_result = weat.calc_weat_pleasant_unpleasant_attribute(w2v_model,
                                                    first_target = dark_subjects,
                                                    second_target = light_subjects,
                                                    with_pvalue = args.with_pvalue)
    print(weat_result)

