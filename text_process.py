import spacy

from tqdm import tqdm
import codecs as cs
from os.path import join as pjoin

nlp = spacy.load('en_core_web_sm')
def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list

def process_humanml3d(corpus):
    text_save_path = './dataset/pose_data_raw/texts'
    desc_all = corpus
    for i in tqdm(range(len(desc_all))):
        caption = desc_all.iloc[i]['caption']
        start = desc_all.iloc[i]['from']
        end = desc_all.iloc[i]['to']
        name = desc_all.iloc[i]['new_joint_name']
        word_list, pose_list = process_text(caption)
        tokens = ' '.join(['%s/%s'%(word_list[i], pose_list[i]) for i in range(len(word_list))])
        with cs.open(pjoin(text_save_path, name.replace('npy', 'txt')), 'a+') as f:
            f.write('%s#%s#%s#%s\n'%(caption, tokens, start, end))

def process_kitml(corpus):
    text_save_path = './dataset/kit_mocap_dataset/texts'
    desc_all = corpus
    for i in tqdm(range(len(desc_all))):
        caption = desc_all.iloc[i]['desc']
        start = 0.0
        end = 0.0
        name = desc_all.iloc[i]['data_id']
        word_list, pose_list = process_text(caption)
        tokens = ' '.join(['%s/%s' % (word_list[i], pose_list[i]) for i in range(len(word_list))])
        with cs.open(pjoin(text_save_path, name + '.txt'), 'a+') as f:
            f.write('%s#%s#%s#%s\n' % (caption, tokens, start, end))

if __name__ == "__main__":
    corpus = pd.read_csv('./dataset/kit_mocap_dataset/desc_final.csv')
    process_humanml3d(corpus)