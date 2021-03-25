from pathlib import Path
from parse_dataset import Dataset
import argparse
import sys
import json


class Parser: 
    SH, LA, RA = 0,1,2

    def conllu(self, source):
        buffer = []
        for line in source:
            line = line.rstrip()    # strip off the trailing newline
            if not line.startswith("#"):
                if not line:
                    yield buffer
                    buffer = []
                else:
                    columns = line.split("\t")
                    if columns[0].isdigit():    # skip range tokens
                        buffer.append(columns)

    def trees(self, source):
        """
        Reads trees from an input source.

        Args: source: An iterable, such as a file pointer.

        Yields: Triples of the form `words`, `tags`, heads where: `words`
        is the list of words of the tree (including the pseudo-word
        <ROOT> at position 0), `tags` is the list of corresponding
        part-of-speech tags, and `heads` is the list of head indices
        (one head index per word in the tree).
        """
        for rows in self.conllu(source):
            words = ["<ROOT>"] + [row[1] for row in rows]
            tags = ["<ROOT>"] + [row[3] for row in rows]
            tree = [0] + [int(row[6]) for row in rows]
            relations = ["root"] + [row[7] for row in rows]
            yield words, tags, tree, relations


    def step_by_step(self,string) :
        """
        Parses a string and builds a dependency tree. In each step,
        the user needs to input the move to be made.
        """
        w = ("<ROOT> " + string).split()
        i, stack, pred_tree = 0, [], [0]*len(w) # Input configuration
        while True :
            print( "----------------" )
            print( "Buffer: ", w[i:] )
            print( "Stack: ", [w[s] for s in stack] )
            print( "Predicted tree: ", pred_tree )
            try :
                m = int(input( "Move (SH=0, LA=1, RA=2): " ))
                if m not in self.valid_moves(i,stack,pred_tree) :
                    print( "Illegal move" )
                    continue
            except KeyboardInterrupt:
                sys.exit()
            except :
                print( "Illegal move" )

                continue
            i, stack, pred_tree = self.move(i,stack,pred_tree,m)
            if i == len(w) and stack == [0] :
                # Terminal configuration
                print( "----------------" )
                print( "Final predicted tree: ", pred_tree )
                return

    def create_dataset(self, source) :
        """
        Creates a dataset from all parser configurations encountered
        during parsing of the training dataset.
        (Not used in assignment 1).
        """
        ds = Dataset()
        with open(source) as f:
            for w,tags,tree,relations in self.trees(f): 
                i, stack, pred_tree = 0, [], [0]*len(tree) # Input configuration
                m = self.compute_correct_move(i,stack,pred_tree,tree)
                while m != None :
                    ds.add_datapoint(w, tags, i, stack, m, train)
                    i,stack,pred_tree = self.move(i,stack,pred_tree,m)
                    m = self.compute_correct_move(i,stack,pred_tree,tree)
        return ds
   


    def valid_moves(self, i, stack, pred_tree):
        """Returns the valid moves for the specified parser
        configuration.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
        
        Returns:
            The list of valid moves for the specified parser
                configuration.
        """
        #SH, LA, RA = 0,1,2
        #print("ääääääää")
        #print("buffer idx ", i)
        #print("stack: ", stack)

        moves = []

        # YOUR CODE HERE
        numwords = len(pred_tree)
        if i < numwords:
            moves.append(0)
        if len(stack) > 1:
            moves.append(2)
        if len(stack) > 2:
            #if stack[-2] = 
            moves.append(1) # so we don't put dependency to ROOT
        
        return moves
        

    def move(self, i, stack, pred_tree, move):
        """
        Executes a single move.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            move: The move that the parser should make.
        
        Returns:
            The new parser configuration, represented as a triple
            containing the index of the new first unprocessed word,
            stack, and partial dependency tree.
        """

        # YOUR CODE HERE
        # SH, LA, RA = 0,1,2
    
        if move == 0: # SH
            stack.append(i)
            i += 1
        elif move == 1: # LA
            right = stack.pop()
            left = stack.pop()
            pred_tree[left] = right
            stack.append(right)
        elif move == 2: # RA
            right = stack.pop()
            pred_tree[right] = stack[-1]

        return i, stack, pred_tree


    def compute_correct_moves(self, tree):
        """
        Computes the sequence of moves (transformations) the parser 
        must perform in order to produce the input tree.
        """

        #print("TREE: ", tree)
        i, stack, pred_tree = 0, [], [] # Input configuration
        # init pred_tree to 0s
        for _ in range(len(tree)):
            pred_tree.append(0)

        moves = []
        m = self.compute_correct_move(i,stack,pred_tree,tree)
        while m != None :
            moves.append(m)
            i,stack,pred_tree = self.move(i,stack,pred_tree,m)
            m = self.compute_correct_move(i,stack,pred_tree,tree)
        #print("PREDICTED TREE: ", pred_tree)
        #print("MOVES: ", moves)
        #if pred_tree != tree:
            #print("NOT EQUAL!!!!!")
            #sys.exit()
        #print("-------")
        return moves


    def compute_correct_move(self, i,stack,pred_tree,correct_tree) :
        """
        Given a parser configuration (i,stack,pred_tree), and 
        the correct final tree, this method computes the  correct 
        move to do in that configuration.
    
        See the textbook, chapter 15, page 11. 
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            correct_tree: The correct dependency tree.
        
        Returns:
            The correct move for the specified parser
            configuration, or `None` if no move is possible.
        """
        assert len(pred_tree) == len(correct_tree)

        # YOUR CODE HERE
        #SH, LA, RA = 0,1,2
        valids = self.valid_moves(i, stack, pred_tree)
        N = len(pred_tree)

        if len(stack) >= 2:
            left = stack[-2]
            right = stack[-1]

            if 1 in valids and correct_tree[left] == right:
                return 1

            if 2 in valids and correct_tree[right] == left:
                nodep = True
                for idx in range(i, N):
                    if correct_tree[idx] == right:
                        nodep = False
                if nodep:
                    return 2

        if i < N and 0 in valids:
            return 0

        return None
   

  
filename = Path("en-ud-train-projective.conllu")
correct_filename = "correct_moves_en-ud-dev.conllu"


if __name__ == '__main__':
    global TREE_NUM
    TREE_NUM = 0

    parser = argparse.ArgumentParser(description='Transition-based dependency parser')
    parser.add_argument('-s', '--step_by_step', type=str, help='step-by-step parsing of a string')
    parser.add_argument('-m', '--compute_correct_moves', type=str, default=filename, help='compute the correct moves given a correct tree')
    args = parser.parse_args()

    p = Parser()
    if args.step_by_step:
        p.step_by_step( args.step_by_step )

    elif args.compute_correct_moves:
        with open(correct_filename) as f:
            corr_moves = [json.loads(line.rstrip()) for line in f]
            
            
        with open(args.compute_correct_moves) as source:
            for w,tags,tree,relations in p.trees(source) :
                #print("TREE_NUM=",TREE_NUM)
                out = p.compute_correct_moves(tree)
                print(out)
                if corr_moves[TREE_NUM] != out:
                    print("NOT EQUAL!")
                    sys.exit()
                TREE_NUM += 1





