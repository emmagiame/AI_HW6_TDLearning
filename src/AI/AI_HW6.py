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
#   - γ (gamma) = 0.95: Discount factor for future rewards
#   - ε (epsilon) = 1.0 → 0.05: Exploration rate with 0.995 decay per episode
#   - α (learning rate) = 0.5: Step size for gradient descent
##
class AIPlayer(Player):
    
    ##
    # Initialize the neural network AI player.
    ##
    def __init__(self, inputPlayerId): 
        super(AIPlayer, self).__init__(inputPlayerId, "ANN_PartB_AI")
        
        # Initialize neural network
        self.network = NeuralNetwork(inputSize=13, hiddenSize1=32, hiddenSize2=16, outputSize=1, learningRate=0.5)

        # Store tuples of (state_features, action, reward, next_state_features, done)
        self.transitions = []

        self.gamma = 0.95  # discount factor
        self.turnPenalty = -0.01  # small step penalty to encourage faster wins
        
        self.epsilon = 1.0  # Start with full exploration
        self.epsilonMin = 0.05  # Minimum exploration rate
        self.epsilonDecay = 0.995  # Decay rate per episode
        
        self.prevState = None
        self.prevAction = None
    
    
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
        
    
    ##
    # Move selection
    # For TD: replace heuristic fallback with ε-greedy over V(s') predictions
    ##
    def getMove(self, currentState):
        # If prevState exists, call: addTransition(prevState, prevAction, currentState, done=False)
        # This records the transition from last move after environment applied it
        if self.prevState is not None and self.prevAction is not None:
            # now in current state after taking prevAction from prevState
            self.addTransition(
                prevState=self.prevState,
                action=self.prevAction,
                nextState=currentState,
                done=False, # game not done yet
                terminalReward=None
            )
        
        # consider all legal moves
        moves = listAllLegalMoves(currentState)
        
        if not moves:
            return Move(END)
        
        # TODO(step 9): Implement ε-greedy move selection
        # if random.random() < self.epsilon:
        #     bestMove = random.choice(moves)
        # else:
        #     # Use network to evaluate moves (see current logic below)
        
        # ε-greedy: explore or exploit
        if random.random() < self.epsilon:
            # Explore: choose random move
            bestMove = random.choice(moves)
        else:
            # Exploit: choose best move based on network evaluation
            bestMove = None
            bestUtility = -float('inf')
            
            for move in moves:
                nextState = getNextState(currentState, move)
                
                try:
                    features = StateEncoder.encodeState(nextState, currentState.whoseTurn)
                    utility = float(self.network.predict(features))
                except Exception:
                    utility = 0.0
                
                if utility > bestUtility:
                    bestUtility = utility
                    bestMove = move
               
            # Fallback in case no best move found     
            if bestMove is None:
                bestMove = random.choice(moves)
        
        # what state we were in before making this move
        self.prevState = currentState # Save state before move
        self.prevAction = bestMove # Save chosen move
        
        return bestMove 
    
    ##
    # Same as HW2_AI random attack selection.
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]
    
    ##
    # Episode end: legacy supervised training (temporary)
    # For TD: finalize last transition with terminal reward, then train on all transitions
    ##
    def registerWin(self, hasWon):
        # Record final transition with terminal reward and done=True
        if self.prevState is not None:
            finalReward = 1.0 if hasWon else -1.0
            # Create terminal transition (nextState can be None or current since it does not matter)
            self.addTransition(
                prevState=self.prevState, 
                action=self.prevAction, 
                nextState=None,
                done=True, 
                terminalReward=finalReward
            )
        
        # train on collected transitions
        self.trainFromTransitions()
        
        # decay epsilon
        self.epsilon = max(self.epsilonMin, self.epsilon * self.epsilonDecay)
        
        # Reset previous state/action for next episode
        self.prevState = None
        self.prevAction = None
    

    # ===============================
    # TD Learning scaffolding
    # ===============================
    def computeReward(self, prevState, action, nextState, done, terminalReward=None):
        
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

    def trainOnTDExample(self, stateFeatures, reward, nextStateFeatures, done):
        try:
            v_s = float(self.network.predict(stateFeatures))
            target = reward if done else reward + self.gamma * float(self.network.predict(nextStateFeatures))
            # Backprop toward TD target
            self.network.trainOnExample(stateFeatures, target)
            return (v_s - target) ** 2
        except Exception:
            return 0.0

    def addTransition(self, prevState, action, nextState, done=False, terminalReward=None):
        # Call this in registerWin with done=True and terminalReward=+1/-1
        try:
            sf = StateEncoder.encodeState(prevState, self.playerId)
            nsf = StateEncoder.encodeState(nextState, self.playerId) if nextState else None
            # Reward computed from state diff; final outcome handled in registerWin
            r = self.computeReward(prevState, action, nextState, done, terminalReward)
            self.transitions.append((sf, action, r, nsf, done))
        except Exception:
            pass

    def trainFromTransitions(self):
        # shuffle transitions for better convergence
        random.shuffle(self.transitions)
        
        try:
            for (sf, action, r, nsf, done) in self.transitions:
                self.trainOnTDExample(sf, r, nsf, done)
        finally:
            # Clear transitions after training
            self.transitions = []
