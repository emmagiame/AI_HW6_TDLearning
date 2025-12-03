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

3. Store State Transitions (Emma) #DONE
   - Instead of (state, heuristic_value) pairs, store:
     (current_state, action, reward, next_state, done)

4. Define Reward Function (Emma) #DONE
   - +1.0 for winning
   - -1.0 for losing
   - +0.1 for collecting food
   - -0.01 for each turn (encourages faster wins)
   - 0 otherwise

5. Modify Training Logic (Emma) #DONE
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

6. Update getMove() Method (Leonie) #DONE
   - Store previous state before making a move
   - After opponent's turn, calculate reward
   - Train on transition: (prev_state, move, reward, current_state)

7. Modify registerWin() Method (Leonie) #DONE
   - Train on final transition with terminal reward
   - Process entire episode's transitions
   - Update network weights based on accumulated TD errors

8. Add Episode Memory (Leonie) #DONE
   - Store sequence of states and actions during each game:
     self.episodeHistory = []  # List of (state, action, reward) tuples

9. Add Exploration Strategy (Emma) #DONE
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
# Description: A multi-layer neural network that learns to evaluate game states
# using Temporal Difference (TD) learning. The network predicts state values V(s)
# and is trained via TD(0) updates: V(s) ← V(s) + α[R + γV(s') - V(s)].
##
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
# Description: Converts game state to a normalized 13-feature vector for neural network input.
# All features are normalized to [0, 1] range for consistent network training.
#
# Features encoded:
#   0: Food collected (normalized to max 11)
#   1: Number of workers (normalized to max 3)
#   2: Number of soldiers (normalized to max 1)
#   3: Queen health (normalized 0-20)
#   4: Enemy queen health (normalized 0-20)
#   5: Anthill health (normalized 0-3)
#   6: Minimum distance from workers to food (normalized)
#   7: Minimum distance from carrying workers to deposit (normalized)
#   8: Workers carrying food (normalized)
#   9: Enemy food count (normalized to max 11)
#   10: Food advantage (normalized -11 to +11)
#   11: Enemy soldiers on our side (normalized)
#   12: Enemy threat to queen (distance-based, normalized)
##
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
# Description: An AI agent that uses TD(0) learning to play the Antics game.
# The agent learns state values through self-play, using a neural network as
# a function approximator for V(s).
#
# Key components:
#   - ε-greedy exploration: Balances exploration vs exploitation (ε decays over episodes)
#   - Transition storage: Records (s, a, r, s', done) tuples during gameplay
#   - Reward shaping: 14+ reward categories encourage strategic behavior
#   - TD(0) training: Updates value estimates using V(s) ← V(s) + α[R + γV(s') - V(s)]
#
# Learning parameters:
#   - γ (gamma) = 0.9: Discount factor for future rewards
#   - ε (epsilon) = 1.0 → 0.05: Exploration rate with 0.995 decay per episode
#   - α (learning rate) = 0.1: Step size for gradient descent
##
class AIPlayer(Player):
    
    ##
    # Initialize the neural network AI player.
    ##
    def __init__(self, inputPlayerId): 
        super(AIPlayer, self).__init__(inputPlayerId, "AI_HW6")
        
        # Initialize neural network
        self.network = NeuralNetwork(inputSize=13, hiddenSize1=32, hiddenSize2=16, outputSize=1, learningRate=0.5)

        # Store tuples of (state_features, action, reward, next_state_features, done)
        self.transitions = []

        self.gamma = 0.9  # discount factor
        self.turnPenalty = -0.005  # smaller step penalty (was -0.01)
        self.alpha = 0.1  # learning rate - increased for faster learning (was 0.01)
        self.V = {}
        self.load_value_table()

        self.epsilonMin = 0.05  # Minimum exploration rate
        self.epsilonDecay = 0.995  # Decay rate per episode
        self.epsilon = 1.0  # Default: full exploration (will be overwritten if saved)
        self.load_epsilon()  # Load saved epsilon if it exists
        
        self.prevState = None
        self.prevAction = None

        # Episode memory (sequence of transitions)
        self.episodeHistory = []  # list of (stateFeatures, action, reward, nextStateFeatures, done)
        
        # Game counter for periodic saving (every 50 games instead of every game)
        self.gameCount = 0
        self.saveInterval = 50
    
    
    ##
    # Setup phase placement logic (same as always).
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
        
    
    ##
    # Move selection
    # ε-greedy over V(s') predictions using the V table (tabular TD learning)
    ##
    def getMove(self, currentState):
        # consider all legal moves
        moves = listAllLegalMoves(currentState)
        
        if not moves:
            return Move(END)

        # ε-greedy: explore or exploit
        if random.random() < self.epsilon:
            # Explore: choose random move
            bestMove = random.choice(moves)
        else:
            # Exploit: choose best move based on V table lookup (fast!)
            bestMove = None
            bestUtility = -float('inf')
            
            for move in moves:
                nextState = getNextState(currentState, move)
                
                # Use V table lookup instead of neural network (much faster)
                stateKey = self.stateCategory(nextState)
                utility = self.V.get(stateKey, 0.0)
                
                if utility > bestUtility:
                    bestUtility = utility
                    bestMove = move
               
            # Fallback in case no best move found     
            if bestMove is None:
                bestMove = random.choice(moves)

        # adds transition from previous move
        if self.prevState is not None and self.prevAction is not None:
            # now in current state after taking prevAction from prevState
            self.addTransition(
                prevState=self.prevState,
                action=self.prevAction,
                nextState=currentState,
                done=False,  # game not done yet
                terminalReward=None
            )

            # Calculate reward - just turn penalty, let terminal rewards guide learning
            reward = self.turnPenalty
            self.tdUpdate(self.prevState, reward, currentState)

        # what state we were in before making this move
        self.prevState = currentState # Save state before move
        self.prevAction = bestMove # Save chosen move
        
        return bestMove 


    ##
    # Same as always: random attack from available enemies
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]
    
    ##
    # Called at end of episode. Add final terminal transition, train on all transitions,
    # optionally: perform an episode pass, decay epsilon, and clear episode memory.
    ##
    def registerWin(self, hasWon):
        finalReward = 1.0 if hasWon else -1.0
        
        # BACKWARD PROPAGATION: Update all states in episode history
        # Start from terminal reward and propagate backward
        futureValue = finalReward
        
        # Get all state keys visited this episode (trans[0] is already the state key)
        # Process in reverse order to propagate values backward
        episode_states = []
        for trans in self.transitions:
            if trans[0] is not None:  # trans[0] is already the state_key (from stateCategory)
                episode_states.append(trans[0])
        
        # Add the final state if we have one
        if self.prevState is not None:
            episode_states.append(self.stateCategory(self.prevState))
        
        # Backward pass: propagate value from terminal state back through episode
        for state_key in reversed(episode_states):
            old_value = self.V.get(state_key, 0.0)
            td_error = futureValue - old_value
            self.V[state_key] = old_value + self.alpha * td_error
            # Next iteration's future value is this state's updated value (discounted)
            futureValue = self.gamma * self.V[state_key]

        # decay exploration rate
        self.epsilon = max(self.epsilonMin, self.epsilon * self.epsilonDecay)

        # reset episode memory and prev state/action
        self.transitions = []
        self.episodeHistory = []
        self.prevState = None
        self.prevAction = None
        
        # Only save periodically to avoid slow disk I/O every game
        self.gameCount += 1
        if self.gameCount % self.saveInterval == 0:
            self.save_value_table()
            self.save_epsilon()  # Save epsilon to preserve exploration progress
    

    # ===============================
    # TD Learning scaffolding
    # ===============================
    
    ## 
    # compute reward for TD learning.
    ##
    def computeReward(self, prevState, action, nextState, done, terminalReward=None):
        # terminal reward for win/loss
        if done and terminalReward is not None:
            return terminalReward
        
        reward = 0.0
        # inventory info
        myInvPrev = prevState.inventories[self.playerId]
        myInvNext = nextState.inventories[self.playerId]
        foods = getConstrList(prevState, 2, (FOOD,))
        
        # construction info
        anthillPrev = myInvPrev.getAnthill()
        anthillNext = myInvNext.getAnthill()
        tunnelPrev = myInvPrev.getTunnels()
        tunnelNext = myInvNext.getTunnels()
        homeCoords = []
        if anthillNext:
            homeCoords.append(anthillNext.coords)
        if tunnelNext:
            homeCoords.extend([t.coords for t in tunnelNext])
            
        # enemy info
        enemyId = 1 - self.playerId
        enemySide = range(5,10) if self.playerId == 0 else range(0,5)
        enemyInvPrev = prevState.inventories[enemyId]
        enemyInvNext = nextState.inventories[enemyId]
        enemyQueenPrev = getAntList(prevState, enemyId, (QUEEN,))
        enemyQueenNext = getAntList(nextState, enemyId, (QUEEN,))
        if enemyQueenPrev:
            enemyQueenPrev = enemyQueenPrev[0]
        if enemyQueenNext:
            enemyQueenNext = enemyQueenNext[0]
        enemyAttackersPrev = getAntList(prevState, enemyId, (SOLDIER,))
        enemyAttackersNext = getAntList(nextState, enemyId, (SOLDIER,))
        
        # Ant info
        myWorkersPrev = getAntList(prevState, self.playerId, (WORKER,))
        myWorkersNext = getAntList(nextState, self.playerId, (WORKER,))
        workerOnFoodPrev = sum(1 for w in myWorkersPrev if not w.carrying and any(w.coords == f.coords for f in foods))
        workerOnFoodNext = sum(1 for w in myWorkersNext if not w.carrying and any(w.coords == f.coords for f in foods))
        workerDeliveringPrev = sum(1 for w in myWorkersPrev if w.carrying and w.coords in homeCoords)
        workerDeliveringNext = sum(1 for w in myWorkersNext if w.carrying and w.coords in homeCoords)
        workersNearFoodPrev = sum(1 for w in myWorkersPrev if not w.carrying and any(approxDist(w.coords, f.coords) <= 1 for f in foods))
        workersNearFoodNext = sum(1 for w in myWorkersNext if not w.carrying and any(approxDist(w.coords, f.coords) <= 1 for f in foods))
        workersNearHomePrev = sum(1 for w in myWorkersPrev if w.carrying and any(approxDist(w.coords, h) <= 1 for h in homeCoords))
        workersNearHomeNext = sum(1 for w in myWorkersNext if w.carrying and any(approxDist(w.coords, h) <= 1 for h in homeCoords))
        workerCountPrev = len(myWorkersPrev)
        workerCountNext = len(myWorkersNext)
        myAttackersPrev = getAntList(prevState, self.playerId, (SOLDIER,))
        myAttackersNext = getAntList(nextState, self.playerId, (SOLDIER,))
        attackersOnEnemySidePrev = sum(1 for a in myAttackersPrev if a.coords[1] in enemySide)
        attackersOnEnemySideNext = sum(1 for a in myAttackersNext if a.coords[1] in enemySide)
        myQueenPrev = myInvPrev.getQueen()
        myQueenNext = myInvNext.getQueen()
        
        # other info
        carryingPrev = sum(1 for w in myWorkersPrev if w.carrying) # if previous worker was carrying
        carryingNext = sum(1 for w in myWorkersNext if w.carrying) # if next worker is carrying/ will be carrying
        threatsNearQueenPrev = sum(1 for a in enemyAttackersPrev if approxDist(a.coords, myQueenPrev.coords) <= 2) if myQueenPrev else 0
        threatsNearQueenNext = sum(1 for a in enemyAttackersNext if myQueenNext and approxDist(a.coords, myQueenNext.coords) <= 2) 
        threatsNearAnthillPrev = sum(1 for a in enemyAttackersPrev if approxDist(a.coords, anthillPrev.coords) <= 2) if anthillPrev else 0
        threatsNearAnthillNext = sum(1 for a in enemyAttackersNext if anthillNext and approxDist(a.coords, anthillNext.coords) <= 2) 
    
        # 1. worker on food but not carrying
        if workerOnFoodNext > workerOnFoodPrev:
            reward += 0.05 * (workerOnFoodNext - workerOnFoodPrev)
        
        # 2. worker with food on anthill or tunnel
        if workerDeliveringNext > workerDeliveringPrev:
            reward += 0.1 * (workerDeliveringNext - workerDeliveringPrev)
        
        # 3. attacker on board
        if len(myAttackersNext) > len(myAttackersPrev):
            reward += 0.03 # small reward for having attackers
        
        # 4. attacker on enemies side of board
        if attackersOnEnemySideNext > attackersOnEnemySidePrev:
            reward += 0.04 # reward for pushing attackers forward
        
        # 5. queen health is less than full
        if myQueenPrev and myQueenNext:
            if myQueenNext.health < myQueenPrev.health:
                damage = myQueenPrev.health - myQueenNext.health
                reward -= 0.02 * damage # small penalty per damage point
        
        # 6. anthill health is less than full
        if anthillPrev and anthillNext:
            if anthillNext.captureHealth < anthillPrev.captureHealth:
                damage = anthillPrev.captureHealth - anthillNext.captureHealth
                reward -= 0.02 * damage # small penalty per damage point
        
        # 7. enemy attacker is within 2 steps of queen
        if myQueenNext and enemyAttackersNext:
            if threatsNearQueenNext > threatsNearQueenPrev:
                reward -= 0.08 # penalty for more threats near queen
        
        # 8. enemy ant is within 2 steps of anthill
        if anthillNext and enemyAttackersNext:
            if threatsNearAnthillNext > threatsNearAnthillPrev:
                reward -= 0.06 # penalty for more threats near anthill
        
        # 9. food more than 8
        if myInvPrev.foodCount < 8 and myInvNext.foodCount >= 8:
            reward += 0.2 # big reward for reaching food milestone
        
        # 10. food is more than or equal to 2
        if myInvPrev.foodCount < 2 and myInvNext.foodCount >= 2:
            reward += 0.05 # reward for reaching food milestone
        
        # 11. food is greater than enemies food
        if myInvNext.foodCount > enemyInvNext.foodCount and myInvPrev.foodCount <= enemyInvPrev.foodCount:
            reward += 0.03 # reward for having more food than enemy
        if myInvNext.foodCount < enemyInvNext.foodCount and myInvPrev.foodCount >= enemyInvPrev.foodCount:
            reward -= 0.03 # penalty for having less food than enemy
        
        # 12. number of workers is equal to 1
        if len(myWorkersPrev) != 1 and len(myWorkersNext) == 1:
            reward += 0.02 # small reward for having exactly 1 worker
        if len(myWorkersPrev) == 1 and len(myWorkersNext) != 1:
            reward -= 0.02 # small penalty for losing exact 1 worker status
            
        # 13. worker within 1 steps of food
        if foods and myWorkersNext:
            if workersNearFoodNext > workersNearFoodPrev:
                reward += 0.02 # small reward for workers approaching food
        
        # 14. worker with food within 1 step of anthill or tunnel
        if homeCoords and myWorkersNext:
            if workersNearHomeNext > workersNearHomePrev:
                reward += 0.03 # small reward for workers approaching home while carrying food
        
        # 15. enemy attacker on board
        if len(enemyAttackersNext) > len(enemyAttackersPrev):
            reward -= 0.02 # small penalty for enemy attackers being on board
        
        ### Additional reward shaping ###
        
        # Food delivery reward
        if myInvNext.foodCount > myInvPrev.foodCount:
            reward += 0.15 # more valuable than just picking up food
            
        # Food pick-up reward
        if carryingNext > carryingPrev:
            reward += 0.05 * (carryingNext - carryingPrev)
            
        # Enemy queen damage reward
        if enemyQueenPrev and enemyQueenNext:
            if enemyQueenNext.health < enemyQueenPrev.health:
                damage = enemyQueenPrev.health - enemyQueenNext.health
                reward += 0.02 * damage # small reward per damage point
        
        # Big reward for killing enemy queen
        if enemyQueenPrev and not enemyQueenNext:
            reward += 0.5 # big reward for winning fight but still less than terminal win reward
        
        # Anthill loss penalty
        if anthillPrev and not anthillNext:
            reward -= 0.3 # big penalty for losing anthill
            
        # Tunnel loss penalty
        if tunnelPrev and not tunnelNext:
            reward -= 0.2 # penalty for losing tunnel
        
        # Worker loss penalty
        if workerCountNext < workerCountPrev:
            reward -= 0.05 * (workerCountPrev - workerCountNext)
            
        # Small penalty for too many workers (waste of resources)
        if workerCountNext > 3:
            reward -= 0.01
        
        return reward

    ##
    # Train on a single TD example.
    ##
    def trainOnTDExample(self, stateFeatures, reward, nextStateFeatures, done):
        try:
            v_s = float(self.network.predict(stateFeatures))
            target = reward if done else reward + self.gamma * float(self.network.predict(nextStateFeatures))
            # Backprop toward TD target
            self.network.trainOnExample(stateFeatures, target)
            return (v_s - target) ** 2
        except Exception:
            return 0.0

    ##
    # Store a transition for later training.
    ##
    def addTransition(self, prevState, action, nextState, done=False, terminalReward=None):
        try:
            sf = self.stateCategory(prevState)
        except Exception:
            sf = None

        nsf = None
        if nextState is not None:
            try:
                nsf = self.stateCategory(nextState)
            except Exception:
                nsf = None

        # compute the reward now (terminalReward passed when registering final reward)
        r = 0.0
        try:
            r = self.computeReward(prevState, action, nextState, done, terminalReward)
        except Exception:
            r = 0.0

        # Append to batch transitions if we have encodings (used by trainFromTransitions)
        if sf is not None:
            self.transitions.append((sf, action, r, nsf, done))

        # Append the raw/encoded transition to episode history for end-of-episode processing
        self.episodeHistory.append((sf, action, r, nsf, done))

    ##
    # Train from all stored transitions.
    ##
    def trainFromTransitions(self):
        if not self.transitions:
            return

        random.shuffle(self.transitions)
        try:
            for (sf, action, r, nsf, done) in self.transitions:
                # trainOnTDExample uses network.predict internally
                self.trainOnTDExample(sf, r, nsf, done)
        finally:
            # Clear transitions after training to start fresh for next episode
            self.transitions = []


    ##
    # TD Learning Update Rule
    # updates the value of each state based on its reward, the learning rate, the discount factor, and the next state's value
    ##
    def tdUpdate(self, oldState, reward, newState):

        old_key = self.stateCategory(oldState)
        new_key = self.stateCategory(newState)

        old_value = self.V.get(old_key, 0.0)
        new_value = self.V.get(new_key, 0.0)

        td_target = reward + self.gamma * new_value
        td_error = td_target - old_value

        self.V[old_key] = old_value + self.alpha * td_error


    # ##
    # # assigns states into categories to limit space needed to store state values
    # # category is assigned based on economy, army strength, and queen's danger of state
    # # returns a tuple used to index the TD value table
    # ##
    def stateCategory(self, state):
        ## Economy

        # food category based on number of food player has
        food = state.inventories[self.playerId].foodCount
        if food <= 2:
            food_cat = 0
        elif food <= 5:
            food_cat = 1
        else:
            food_cat = 2

        # worker category based on number of workers
        workers = [a for a in getAntList(state, self.playerId) if a.type == WORKER]
        num_workers = len(workers)
        if num_workers == 0:
            worker_cat = 0
        elif num_workers == 1:
            worker_cat = 1
        else:
            worker_cat = 2

        # worker distance category based on distance from carrying worker to anthill or tunnel / non-carrying worker to food
        workers = [a for a in getAntList(state, self.playerId) if a.type == WORKER]
        foodObjs = self.getCurrPlayerFood(state)
        tunnel = getCurrPlayerInventory(state).getTunnels()[0]
        anthill = getCurrPlayerInventory(state).getAnthill()
        distances = []
        for worker in workers:
            if worker.carrying:
                # Carrying worker should go HOME (tunnel/anthill)
                distances.append(
                    min(approxDist(worker.coords, tunnel.coords), approxDist(worker.coords, anthill.coords)))
            else:
                # Non-carrying worker should go to FOOD
                distances.append(min(approxDist(worker.coords, foodObjs[0].coords),
                                     approxDist(worker.coords, foodObjs[1].coords)))
        average_distance = sum(distances) / len(distances) if len(distances) > 0 else 0
        if average_distance < 3:
            worker_dist_cat = 0 # close or on target
        elif average_distance < 5:
            worker_dist_cat = 1 # moving towards target
        else:
            worker_dist_cat = 2 # far away from target
        
        # Carrying status - are workers carrying food?
        carrying_count = sum(1 for w in workers if w.carrying)
        if num_workers == 0:
            carry_cat = 0
        elif carrying_count == 0:
            carry_cat = 0  # no one carrying
        elif carrying_count == num_workers:
            carry_cat = 2  # all carrying
        else:
            carry_cat = 1  # some carrying

        ## Army Strength

        # difference in army strength
        my_combat = len([a for a in getAntList(state, self.playerId) if a.type in (SOLDIER, DRONE)])
        opp_combat = len([a for a in getAntList(state, 1 - self.playerId) if a.type in (SOLDIER, DRONE)])
        diff = my_combat - opp_combat
        if diff <= -2:
            army_diff_cat = 0  # behind
        elif diff <= 1:
            army_diff_cat = 1  # equal
        else:
            army_diff_cat = 2  # ahead

        # army strength
        if my_combat <= 1:
            army_cat = 0  # behind
        elif diff <= 4:
            army_cat = 1  # equal
        else:
            army_cat = 2  # ahead

        ## Queen

        # queen danger based on closest attacking ant from opponent
        my_queen = getAntList(state, self.playerId, (QUEEN,))[0]
        enemy_soldiers = getAntList(state, 1 - self.playerId, (SOLDIER,))
        if enemy_soldiers:
            dists = [approxDist(my_queen.coords, s.coords) for s in enemy_soldiers]
            min_dist = min(dists)
            if min_dist <= 3:
                q_d_cat = 2  # very close
            elif min_dist <= 6:
                q_d_cat = 1  # approaching
            else:
                q_d_cat = 0  # none in sight
        else:
            q_d_cat = 0

        # queen's health
        if my_queen.health <= 4:
            q_h_cat = 2 # dying
        elif my_queen.health <= 7:
            q_h_cat = 1 # badly injured
        else:
            q_h_cat = 0 # healthy


        # Return a compact categorical representation
        # 8 categories: food(3) * worker(3) * dist(3) * carry(3) * army_diff(3) * army(3) * q_danger(3) * q_health(3) = 6561 states
        return (food_cat * 2187 + worker_cat * 729 + worker_dist_cat * 243 + 
                carry_cat * 81 + army_diff_cat * 27 + army_cat * 9 + q_d_cat * 3 + q_h_cat)


    ##
    # Return: a list of the food objects on my side of the board
    def getCurrPlayerFood(self, currentState):
        food = getConstrList(currentState, 2, (FOOD,))
        myFood = []
        if (currentState.inventories[0].player == currentState.whoseTurn):
            myFood.append(food[2])
            myFood.append(food[3])
        else:
            myFood.append(food[0])
            myFood.append(food[1])
        return myFood

    def save_value_table(self, filename="value_table.json"):
        # Use absolute path relative to this script's parent directory (src/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, "..", filename)
        with open(filepath, "w") as f:
            json.dump(self.V, f)

    def load_value_table(self, filename="value_table.json"):
        # Use absolute path relative to this script's parent directory (src/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, "..", filename)
        try:
            with open(filepath, "r") as f:
                content = f.read().strip()

                # If file exists but is empty → start fresh
                if content == "":
                    print("Value table file is empty. Starting fresh.")
                    self.V = {}
                    return

                # Otherwise try to decode JSON content
                self.V = json.loads(content)

        except FileNotFoundError:
            print("No saved value table found. Starting fresh.")
            self.V = {}

        except json.JSONDecodeError:
            print("Value table file is invalid or corrupted. Starting fresh.")
            self.V = {}

    def save_epsilon(self, filename="epsilon.json"):
        # Save epsilon to preserve exploration progress across sessions
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, "..", filename)
        with open(filepath, "w") as f:
            json.dump({"epsilon": self.epsilon}, f)

    def load_epsilon(self, filename="epsilon.json"):
        # Load epsilon to resume training from where we left off
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, "..", filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                self.epsilon = data.get("epsilon", 1.0)
                print(f"Loaded epsilon: {self.epsilon:.4f}")
        except (FileNotFoundError, json.JSONDecodeError):
            print("No saved epsilon found. Starting with full exploration.")
            self.epsilon = 1.0