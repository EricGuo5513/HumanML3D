import codecs as cs


def process_text(sentence):
    sentence = sentence.replace("-", "")
    doc = nlp(sentence)
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == "NOUN" or token.pos_ == "VERB") and (word != "left"):
            yield token.lemma_
        else:
            yield word


def process_humanml3d(corpus):
    text_save_path = "./dataset/pose_data_raw/texts"
    desc_all = corpus
    for i in range(len(desc_all)):
        caption = desc_all.iloc[i]["caption"]
        start = desc_all.iloc[i]["from"]
        end = desc_all.iloc[i]["to"]
        name = desc_all.iloc[i]["new_joint_name"]
        word_list, pose_list = process_text(caption)
        tokens = " ".join([f"{word}/{pose}" for word, pose in word_list])
        with cs.open(text_save_path / name.with_suffix(".txt"), "a+") as f:
            f.write(f"{caption}#{tokens}#{start}#{end}\n")
