from collections import defaultdict

class BytePairEncodingTokenizer:
    def __init__(self, corpus, iterations=300):
        self.corpus = corpus
        self.vocab = defaultdict(int)
        self.iterations = iterations

    def set_word_frequencies(self):
        self.word_frequencies = defaultdict(int)
        
        for word in self.corpus.split():
          self.word_frequencies[word] += 1

    def set_pair_frequencies(self):
        self.pair_frequencies = defaultdict(int)

        for word in self.word_frequencies.keys():
          chars = self.split_word_according_to_vocab(word)
          for char in range(len(chars) - 1):
            pair = ''.join(str(x) for x in chars[char:char+2])
            self.pair_frequencies[pair] += self.word_frequencies[word]

    def set_base_vocab(self):
        for word in self.word_frequencies.keys():
          splitted_word_by_end_char = word.split("</w>")
          for char in splitted_word_by_end_char[0]:
            self.vocab[f"{char}"] += self.word_frequencies[word]
          self.vocab["</w>"] += self.word_frequencies[word]

    def merge_vocab(self):
        max_key = max(self.pair_frequencies, key=self.pair_frequencies.get)
        char_1, char_2 = self.split_word_according_to_vocab(max_key)

        new_vocab = defaultdict(int)
        for word in self.vocab.keys():
          if word == char_1:
            new_vocab[word] = self.vocab[word] - self.pair_frequencies[max_key]
          elif word == char_2:
            new_vocab[word] = self.vocab[word] - self.pair_frequencies[max_key]
          else:
            new_vocab[word] = self.vocab[word]
        
        new_vocab[max_key] = self.pair_frequencies[max_key]

        cleaned_vocab = defaultdict(int)
        for vocab in new_vocab.keys():
          if new_vocab[vocab] > 0:
            cleaned_vocab[vocab] = new_vocab[vocab]

        self.vocab = cleaned_vocab

    # replace according to longest match
    def split_word_according_to_vocab(self, word):
      sorted_dict = dict(sorted(self.vocab.items(), key=lambda item: -len(item[0])))

      word_chars = list(word)
      replaced = 0
      for vocab in sorted_dict.keys():
        if vocab in word:
          word = word.replace(vocab, "|" + str(replaced) + "|")
        replaced += 1

      word_tokenized = []
      for thing in word.split("|"):
         if thing.isnumeric():
            word_tokenized.append(list(sorted_dict.keys())[int(thing)])
      return word_tokenized

    def train(self):
      print("Training Byte Pair Encoding Tokenizer")
      self.set_word_frequencies()
      self.set_base_vocab()
      print("Starting training...")
      for i in range(self.iterations):
        print(f"Training iteration {i+1}/{self.iterations}")
        self.set_pair_frequencies()
        self.merge_vocab()
      print("Training finished! Vocab size: ", len(self.vocab))

    def get_vocab_index_for_token(self, word):
       i = 0
       for value in sorted(self.vocab.keys(), key=lambda x: -len(x)):
          if value == word:
            return i
          i += 1

    def get_token_for_vocab_index(self, index):
      i = 0
      for value in sorted(self.vocab.keys(), key=lambda x: -len(x)):
        if i == index:
          return value
        i += 1 
    def encode(self, word):
      encoded = []
      for vocab in self.split_word_according_to_vocab(word):
         encoded.append(self.get_vocab_index_for_token(vocab))
      return encoded
    
    def decode(self, word):
      decoded = []
      for vocab in word:
        decoded.append(self.get_token_for_vocab_index(vocab))
      return "".join(decoded)

