import os
import math
import random
import nltk
import numpy as np
import numpy.random as rand
import os.path
import argparse
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors


"""
Python implementation of the Glove training algorithm from the article by Pennington, Socher and Manning (2014).

This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019, 2021 by Johan Boye.
"""
class Glove:

    # Mapping from words to IDs.
    word2id = defaultdict(lambda: None)

    # Mapping from IDs to words.
    id2word = defaultdict(lambda: None)

    # Mapping from focus words to neighbours to counts (called X 
    # to be consistent with the notation in the Glove paper).
    X = defaultdict(lambda: defaultdict(int))

    # Mapping from word IDs to (focus) word vectors. (called w_vector 
    # to be consistent with the notation in the Glove paper).
    w_vector = defaultdict(lambda: None)

    # Mapping from word IDs to (context) word vectors (called w_tilde_vector
    # to be consistent with the notation in the Glove paper)
    w_tilde_vector = defaultdict(lambda: None)

    # The ID of the latest encountered new word.
    latest_new_word = -1

    # Dimension of word vectors.
    dimension = 50

    # Left context window size.
    left_window_size = 2

    # Right context window size.
    right_window_size = 2

    # The local context window.
    window = []

    # The ID of the current focus word.
    focus_word_id = -1

    # The current token number.
    current_token_number = 0

    # Cutoff for gradient descent.
    epsilon = 0.01

    # Initial learning rate.
    learning_rate = 0.05

    # The number of times we can tolerate that loss increases
    patience = 5
    
    # Temporary file used for storing the model
    temp_file = "temp__.txt"
    
    # Initializes the local context window
    def __init__( self, ignore, left_window_size, right_window_size ) :
        self.window = [-1 for i in range(left_window_size + right_window_size)]
        self.left_window_size = left_window_size
        self.right_window_size = right_window_size
        # Re-start training from pretrained vectors 
        if not ignore and os.path.exists(self.temp_file):
            self.read_temp_file( self.temp_file )
        


    #--------------------------------------------------------------------------
    #
    #  Methods for processing all files and computing all counts
    #


    # Initializes the necessary information for a word.

    def init_word( self, word ) :

        self.latest_new_word += 1

        # This word has never been encountered before. Init all necessary
        # data structures.
        self.id2word[self.latest_new_word] = word
        self.word2id[word] = self.latest_new_word

        # Initialize arrays with random numbers in [-0.5,0.5].
        w = rand.rand(self.dimension)-0.5
        self.w_vector[self.latest_new_word] = w
        w_tilde = rand.rand(self.dimension)-0.5
        self.w_tilde_vector[self.latest_new_word] = w_tilde
        return self.latest_new_word



    # Slides in a new word in the local context window
    #
    # The local context is a list of length left_window_size+right_window_size.
    # Suppose the left window size and the right window size are both 2.
    # Consider a sequence
    #
    # ... this  is  a  piece  of  text ...
    #               ^
    #           Focus word
    #
    # Then the local context is a list [id(this),id(is),id(piece),id(of)],
    # where id(this) is the wordId for 'this', etc.
    #
    # Now if we slide the window one step, we get
    #
    # ... is  a  piece  of  text ...
    #              ^
    #         New focus word
    #
    # and the new context window is [id(is),id(a),id(of),id(text)].
    
    def slide_window( self, idx ) :

        # YOUR CODE HERE

        pass


    # Update counts based on the local context window
    def update_counts( self ) :
        for idx in self.window :
            if idx > 0 :
                context_words = self.X[self.focus_word_id]
                if context_words == None :
                    context_words = defaultdict(int)
                    self.X[self.focus_word_id] = context_words
                count = context_words[idx]
                if count == None :
                    count = 0
                context_words[idx] = count+1


    # Handles one token in the text
    def process_token( self, word ) :

        # First check if the word has been encountered before.
        # Init all data structures if it hasn't.
        idx = self.word2id[word]
        if idx is None :
            idx = self.init_word( word )

        # Special cases for the very first words
        if ( self.current_token_number == 0 ) :
            self.focus_word_id = idx
        elif self.current_token_number <= self.right_window_size :
            self.window[self.right_window_size-self.current_token_number] = idx
        else :
            self.update_counts()
            self.slide_window( idx )


    # This function recursively processes all files in a directory
    def process_files( self, file_or_dir ) :
        if os.path.isdir( file_or_dir ) :
            for root,dirs,files in os.walk( file_or_dir ) :
                for file in files :
                    self.process_files( os.path.join(root, file ))
        else :
            stream = open( file_or_dir, mode='r', encoding='utf-8', errors='ignore' )
            text = stream.read()
            try :
                tokens = nltk.word_tokenize(text) 
            except LookupError :
                nltk.download('punkt')
                tokens = nltk.word_tokenize(text)
            for token in tokens:
                self.process_token(token.lower())
                self.current_token_number += 1
                if self.current_token_number % 10000 == 0 :
                    print( 'Processed ' + str(self.current_token_number) + ' tokens' )

        
    #
    #  Methods for processing all files and computing all counts
    #
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #   Loss function, gradient descent, etc.
    #

    # The mysterious "f" function from the article
    def f( self, count ) :
        if count<100 :
            ratio = count/100.0
            return math.pow( ratio, 0.75 )
        return 1.0
    

    # The loss function
    def loss( self ) :

        # YOUR CODE HERE
        
        return 0.0
    

    # Compute the gradient of a given pair of datapoints
    def compute_gradient(self, i, j) :

        # YOUR CODE HERE

        # Returns the gradient of the loss function w.r.t w_vector[i]
        # and w.r.t. w_tilde_vector[j]
        # return wi_vector_grad, wj_tilde_vector_grad
        return None, None

    
    # Stochastic gradient descent
    def train( self ) :
        iterations = 0

        # YOUR CODE HERE
        
        while ( False ) : # Replace this condition

            # YOUR CODE HERE
            
            if iterations%1000000 == 0 :
                self.write_word_vectors_to_file( self.outputfile )
                self.write_temp_file( self.temp_file )
                self.learning_rate *= 0.99
                

    #
    #  End of loss function, gradient descent, etc.
    #
    #-------------------------------------------------------

    #-------------------------------------------------------
    #
    #  I/O
    #

    # Writes the vectors to file. These are the vectors you would
    # export and use in another application.
    def write_word_vectors_to_file( self, filename ) :
        with open(filename, 'w') as f:
            for idx in self.id2word.keys() :
                f.write('{} '.format( self.id2word[idx] ))
                for i in self.w_vector[idx] :
                    f.write('{} '.format( i ))
                f.write( '\n' )
        f.close()


    # Saves the state of the computation to file, so that
    # training can be resumed later.
    def write_temp_file( self, filename ) :
        with open(filename, 'w') as f:
            f.write('{} '.format( self.learning_rate ))
            f.write( '\n' )
            for idx in self.id2word.keys() :
                f.write('{} '.format( self.id2word[idx] ))
                for i in list(self.w_vector[idx]) :
                    f.write('{} '.format( i ))
                for i in list(self.w_tilde_vector[idx]) :
                    f.write('{} '.format( i ))
        f.close()


    # Reads the partially trained model from file, so
    # that training can be resumed.
    def read_temp_file(self, fname):
        i = 0
        with open(fname) as f:
            self.learning_rate = float(f.readline())
            for line in f:
                data = line.split()
                w = data[0]
                vec = np.array([float(x) for x in data[1:self.dimension+1]])
                self.id2word[i] = w
                self.word2id[w] = i
                self.w_vector[i] = vec
                vec = np.array([float(x) for x in data[self.dimension+1:]])
                self.w_tilde_vector[i] = vec
                i += 1
        f.close()
        self.dimension = len( self.w_vector[0] )

       
def main() :

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Glove trainer')
    parser.add_argument('--file', '-f', type=str,  default='../RandomIndexing/data', help='The files used in the training.')
    parser.add_argument('--output', '-o', type=str, default='vectors.txt', help='The file where the vectors are stored.')
    parser.add_argument('--left_window_size', '-lws', type=int, default='2', help='Left context window size')
    parser.add_argument('--right_window_size', '-rws', type=int, default='2', help='Right context window size')

    arguments = parser.parse_args()  
    
    glove = Glove(False, arguments.left_window_size, arguments.right_window_size)
    glove.outputfile = arguments.output
    glove.process_files( arguments.file )
    glove.train()

        
if __name__ == '__main__' :
    main()    

