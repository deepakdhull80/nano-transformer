import string

def isalphanum(sentence):
    return all([all([c in string.ascii_letters for c in list(w)]) for w in sentence.split()])
