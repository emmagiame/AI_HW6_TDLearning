"""
TODO: Convert from Supervised Learning to Temporal Difference Learning

Steps to implement TD Learning:   

1. remove heuristic utility class #DONE

2. Implement TD Update Rule (Leonie)
   - Replace supervised learning with TD(0): V(s) ← V(s) + α[R + γV(s') - V(s)]
   - V(s) = current state value (neural network prediction)
   - R = immediate reward from transition
   - γ = discount factor (typically 0.9-0.99)
   - V(s') = next state value
   - α = learning rate
   - TD learning should learn from actual game outcomes and state transitions

3. Store State Transitions (Emma)
   - Instead of (state, heuristic_value) pairs, store:
     (current_state, action, reward, next_state, done)

4. Define Reward Function (Emma)
   - +1.0 for winning
   - -1.0 for losing
   - +0.1 for collecting food
   - -0.01 for each turn (encourages faster wins)
   - 0 otherwise

5. Modify Training Logic (Emma)
   - Change trainOnExample() to use TD targets instead of supervised targets
   - trainOnTDExample(state, reward, next_state, done):
       V_current = predict(state)
       if done:
           target = reward
       else:
           V_next = predict(next_state)
           target = reward + gamma * V_next
       td_error = target - V_current
       backward(state, target)

6. Update getMove() Method (Leonie)
   - Store previous state before making a move
   - After opponent's turn, calculate reward
   - Train on transition: (prev_state, move, reward, current_state)

7. Modify registerWin() Method (Leonie)
   - Train on final transition with terminal reward
   - Process entire episode's transitions
   - Update network weights based on accumulated TD errors

8. Add Episode Memory (Leonie)
   - Store sequence of states and actions during each game:
     self.episodeHistory = []  # List of (state, action, reward) tuples

9. Add Exploration Strategy (Emma)
    - Implement ε-greedy exploration:
      * With probability ε, choose random move
      * With probability (1-ε), choose best move according to network
      * Decay ε over time (exploration → exploitation)
      
      
      Categories of states:
      1. Worker on food
      2. worker with food on anthill or tunnel
      3. attacker on board
      4. attacker on enemies side of board
      5. queen health is less than full
      5. anthill health is less than full
      6. enemy attacker is within 2 steps of queen
      7. enemy ant is within 2 steps of anthill
      8. food more than 8
      9. food is more than or equal to 2
      10. food is greater than enemies food
      11. number of workers is equal to 1 
      12. worker within 1 steps of food
      13. worker with food within 1 step of anthill or tunnel
      14. enemy attacker on board
      
"""

import random
import sys
import numpy as np
import os
import json
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *
import heapq


##
# NeuralNetwork
# Description: A multi-layer neural network that learns to evaluate game states.
# learns the utility function from HW2_AI through supervised learning.
#
##
#TODO: #2 Convert from Supervised Learning to Temporal Difference Learning
# TD(0): V(s) ← V(s) + α[R + γV(s') - V(s)]
class NeuralNetwork:
    def __init__(self, inputSize=13, hiddenSize1=32, hiddenSize2=16, outputSize=1, learningRate=0.5):
        self.learningRate = learningRate
        
        # Initialize weights with small random values
        # Using Xavier initialization for better convergence
        self.W1 = np.random.randn(inputSize, hiddenSize1) * np.sqrt(2.0 / inputSize)
        self.b1 = np.zeros((1, hiddenSize1))
        
        self.W2 = np.random.randn(hiddenSize1, hiddenSize2) * np.sqrt(2.0 / hiddenSize1)
        self.b2 = np.zeros((1, hiddenSize2))
        
        self.W3 = np.random.randn(hiddenSize2, outputSize) * np.sqrt(2.0 / hiddenSize2)
        self.b3 = np.zeros((1, outputSize))
        
        # Store layer outputs for backpropagation
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.a3 = None
    
    ##
    # ReLU activation function
    ##
    def relu(self, x):
        return np.maximum(0, x)
    
    ##
    # Derivative of ReLU activation function
    ##
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    ##
    # Sigmoid activation function
    ##
    def sigmoid(self, x):
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    ##
    # Derivative of Sigmoid activation function
    ##
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    ##
    # Forward propagation through the network.
    ##
    def forward(self, X):
        # Layer 1: Input -> Hidden1 (ReLU)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Layer 2: Hidden1 -> Hidden2 (ReLU)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        # Layer 3: Hidden2 -> Output (Sigmoid)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        
        return self.a3[0, 0]  # Return scalar output
    
    ##
    # Back propagation to update weights.
    ##
    def backward(self, X, y):
        m = X.shape[0]
        
        # Output layer error (sigmoid derivative)
        dz3 = self.a3 - y
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Hidden layer 2 error
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer 1 error
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W3 -= self.learningRate * dW3
        self.b3 -= self.learningRate * db3
        self.W2 -= self.learningRate * dW2
        self.b2 -= self.learningRate * db2
        self.W1 -= self.learningRate * dW1
        self.b1 -= self.learningRate * db1
    
    ##
    # Train network on a single example.
    ##
    def trainOnExample(self, X, y):
        # Forward pass
        prediction = self.forward(X)
        
        # Backward pass
        self.backward(X, np.array([[y]]))
        
        # Calculate and return error
        error = (prediction - y) ** 2
        return error
    
    ##
    # Make a prediction on input without training.
    ##
    def predict(self, X):
        return self.forward(X)
    
    ##
    # Return all weights as a dictionary for hard-coding.
    ##
    def getWeights(self):
        return {
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist(),
            'W3': self.W3.tolist(),
            'b3': self.b3.tolist(),
        }
    
    ##
    # Load weights from a dictionary (for hard-coded weights).
    ##
    def setWeights(self, weights):
        self.W1 = np.array(weights['W1'])
        self.b1 = np.array(weights['b1'])
        self.W2 = np.array(weights['W2'])
        self.b2 = np.array(weights['b2'])
        self.W3 = np.array(weights['W3'])
        self.b3 = np.array(weights['b3'])


##
# StateEncoder
# Description: Converts game state to normalized feature vector for neural network input.
# Maps game state information to [0, 1] range for network accessibility.
##
#TODO: #3 Store State Features for TD Learning, (current_state, action, reward, next_state, done)
class StateEncoder:
    
    @staticmethod
    def encodeState(currentState, playerId):
        #Encode a game state as a normalized feature vector.
        ##
        
        features = []
        
        # Get player inventory
        myInv = currentState.inventories[playerId]
        enemyId = 1 - playerId  # In 2-player game
        enemyInv = currentState.inventories[enemyId]
        neutralInv = currentState.inventories[2]
        
        # Feature 0: Food collected (normalize to 0-1, max 11)
        food_collected = min(myInv.foodCount, 11) / 11.0
        features.append(food_collected)
        
        # Get all ants and constructions
        myWorkers = getAntList(currentState, playerId, (WORKER,))
        myAnts = getAntList(currentState, playerId)
        mySoldiers = getAntList(currentState, playerId, (SOLDIER,))
        enemyWorkers = getAntList(currentState, enemyId, (WORKER,))
        
        # Feature 1: Number of workers (normalize to 0-1, max 3)
        num_workers = min(len(myWorkers), 3) / 3.0
        features.append(num_workers)
        
        # Feature 2: Number of soldier ants (normalize, max 1 soldier typically)
        num_soldiers = min(len(mySoldiers), 1) / 1.0
        features.append(num_soldiers)
        
        # Feature 3: Queen health (normalize 0-20)
        queen = getCurrPlayerInventory(currentState).getQueen()
        queen_health = (queen.health / 20.0) if queen else 0
        features.append(queen_health)
        
        # Feature 4: Enemy queen health
        enemy_ants = getAntList(currentState, enemyId, (QUEEN,))
        if enemy_ants:
            enemy_queen = enemy_ants[0]
            enemy_queen_health = (enemy_queen.health / 20.0) if enemy_queen else 0
        else:
            enemy_queen_health = 0
        features.append(enemy_queen_health)
        
        # Feature 5: Average worker carrying status
        carrying_ratio = (sum(1 for w in myWorkers if w.carrying) / len(myWorkers)) if myWorkers else 0
        features.append(carrying_ratio)
        
        # Feature 6: Closest food distance (normalized)
        foods = getConstrList(currentState, 2, (FOOD,))
        if foods and myWorkers:
            closest_food_dist = min(
                stepsToReach(currentState, w.coords, f.coords)
                for w in myWorkers for f in foods
            )
            # Normalize: 0 = at food, 1 = far away (max board distance ~20)
            closest_food_norm = min(closest_food_dist / 20.0, 1.0)
        else:
            closest_food_norm = 1.0
        features.append(closest_food_norm)
        
        # Feature 7: Worker to food distance ratio
        if foods and myWorkers:
            avg_dist = np.mean([
                min(stepsToReach(currentState, w.coords, f.coords) for f in foods)
                for w in myWorkers
            ])
            dist_ratio = min(avg_dist / 15.0, 1.0)
        else:
            dist_ratio = 1.0
        features.append(dist_ratio)
        
        # Feature 8: Home proximity of carrying workers
        home_spots = getConstrList(currentState, playerId, (ANTHILL, TUNNEL))
        if home_spots and myWorkers:
            carrying_workers = [w for w in myWorkers if w.carrying]
            if carrying_workers:
                avg_home_dist = np.mean([
                    min(stepsToReach(currentState, w.coords, h.coords) for h in home_spots)
                    for w in carrying_workers
                ])
                home_proximity = min(avg_home_dist / 15.0, 1.0)
            else:
                home_proximity = 1.0
        else:
            home_proximity = 0.5
        features.append(home_proximity)
        
        # Feature 9: Anthill intact
        anthill = getCurrPlayerInventory(currentState).getAnthill()
        anthill_intact = 1.0 if anthill else 0.0
        features.append(anthill_intact)
        
        # Feature 10: Tunnel intact
        tunnel = getCurrPlayerInventory(currentState).getTunnels()
        tunnel_intact = 1.0 if tunnel else 0.0
        features.append(tunnel_intact)
        
        # Feature 11: Enemy worker count (normalize 0-3)
        enemy_worker_count = min(len(enemyWorkers), 3) / 3.0
        features.append(enemy_worker_count)
        
        # Feature 12: Food on board (normalize, typically 0-4)
        food_on_board = min(len(foods), 4) / 4.0
        features.append(food_on_board)
        
        # Convert to numpy array and reshape for network input
        features_array = np.array(features, dtype=np.float32).reshape(1, -1)
        return features_array


##
# AIPlayer
# Description: The neural network-based AI player for HW5 Part B.
# Uses a trained neural network to evaluate game states instead of heuristics.
##
class AIPlayer(Player):
    
    ##
    # Initialize the neural network AI player.
    ##
    def __init__(self, inputPlayerId): 
        super(AIPlayer, self).__init__(inputPlayerId, "ANN_PartB_AI")
        
        # Initialize neural network
        self.network = NeuralNetwork(inputSize=13, hiddenSize1=32, hiddenSize2=16, outputSize=1, learningRate=0.5)
        
        # Flag to indicate if we're using hard-coded weights
        self.useHardcodedWeights = False
        
        # Storage for training data (to follow the hint about randomization)
        self.trainingBuffer = []
        
        # Hard-coded weights (will be set after training).
        # The full trained weights are shipped in `trained_weights.json`.
        # To avoid repeatedly calling the original heuristic after the
        # network is good enough we will load and use these weights.
        self.hardcodedWeights = None
        self.hardcodedWeightsPath = os.path.join(os.path.dirname(__file__), '..', 'trained_weights.json')
        # Threshold to decide when to switch to hard-coded weights.
        self.hardcodeThreshold = 0.01
    
    ##
    # Set hard-coded weights after training.
    ## 
    def setHardcodedWeights(self, weights):        
        self.network.setWeights(weights)
        self.useHardcodedWeights = True
    
    ##
    # Setup phase placement logic (same as HW2_AI).
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        if currentState.phase == SETUP_PHASE_1:
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    x = random.randint(0, 9)
                    y = random.randint(0, 3)
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    x = random.randint(0, 9)
                    y = random.randint(6, 9)
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
        
    #TODO: #5 Define reward function and store transitions for TD Learning
    
    ##
    # Use neural network to evaluate moves.
    ##
    def getMove(self, currentState):
        moves = listAllLegalMoves(currentState)
        
        if not moves:
            return Move(END)
        
        # Collect training data from current state (only while training)
        if not self.useHardcodedWeights:
            self.addToTrainingBuffer(currentState)
        
        bestMove = None
        bestUtility = -1
        
        for move in moves:
            nextState = getNextState(currentState, move)
            
            # Use network if we have switched to hard-coded weights,
            # otherwise fall back to the heuristic (used while training).
            if self.useHardcodedWeights:
                try:
                    features = StateEncoder.encodeState(nextState, currentState.whoseTurn)
                    utility = float(self.network.predict(features))
                except Exception:
                    utility = 0.0
            else:
                # This ensures good gameplay while collecting training data
                #TODO: #1 Replace heuristic with network prediction for TD learning
                utility = HeuristicUtility.evaluate(nextState, currentState.whoseTurn)
            
            # Track best move
            if utility > bestUtility:
                bestUtility = utility
                bestMove = move
        
        return bestMove if bestMove else Move(END)
    
    ##
    # Same as HW2_AI random attack selection.
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]
    
    ##
    # Called when the game ends - use for training.
    ##
    def registerWin(self, hasWon):
        # Train network on buffered game states if available
        if self.trainingBuffer:
            # Perform training silently (no console output)
            try:
                buffer_size = max(1, len(self.trainingBuffer))
                epochs = min(400, max(50, buffer_size // 5))
                totalError = self.trainOnBuffer(epochs=epochs)

                # If error is sufficiently low, load hard-coded weights
                # and switch to using the neural network exclusively.
                if totalError is not None and totalError < self.hardcodeThreshold:
                    try:
                        # Prefer embedded weights if available, otherwise load file
                        if self.hardcodedWeights is None and os.path.exists(self.hardcodedWeightsPath):
                            with open(self.hardcodedWeightsPath, 'r') as fh:
                                weights = json.load(fh)
                                self.setHardcodedWeights(weights)
                        elif self.hardcodedWeights is not None:
                            self.setHardcodedWeights(self.hardcodedWeights)

                        # Clear buffer and stop further training
                        self.trainingBuffer = []
                    except Exception:
                        pass
            except Exception:
                # swallow exceptions to avoid console output; training failure shouldn't stop game flow
                pass
    
    ##
    # Train the network on a single game state.
    # Uses heuristic utility as the target.
    ##
    #TODO
    def trainOnGameState(self, gameState):
        try:
            # If we've switched to hard-coded weights, do not call the heuristic.
            if self.useHardcodedWeights:
                return 0.0
            # Encode state
            features = StateEncoder.encodeState(gameState, self.playerId)
            
            # Get target value from heuristic
            #TODO: #1 Replace heuristic with network prediction for TD learning
            target = HeuristicUtility.evaluate(gameState, self.playerId)
            
            # Train network on this example
            error = self.network.trainOnExample(features, target)
            
            return error
        except:
            return 0.0
    
    ##
    # Add a game state to the training buffer.
    ##
    def addToTrainingBuffer(self, gameState):
        try:
            # Do not add more training data once we switched to hard-coded weights
            if self.useHardcodedWeights:
                return
            #TODO: #1 Replace heuristic with network prediction for TD learning
            target = HeuristicUtility.evaluate(gameState, self.playerId)
            features = StateEncoder.encodeState(gameState, self.playerId)
            self.trainingBuffer.append((features, target))
        except:
            pass
    
    ##
    # Train on buffered states in random order.
    ##
    def trainOnBuffer(self, epochs=100):
        if not self.trainingBuffer:
            return 0.0
        
        totalError = 0.0
        
        for epoch in range(epochs):
            # Randomize order
            random.shuffle(self.trainingBuffer)
            
            # Train on all examples
            epochError = 0.0
            for features, target in self.trainingBuffer:
                try:
                    error = self.network.trainOnExample(features, target)
                    epochError += error
                except:
                    pass
            
            totalError = epochError / len(self.trainingBuffer)
        
        return totalError
