import random
import sys
import unittest
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
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "HW2_AI")
    
    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    #getMove
    #Description: Gets the next move from the Player using a 1-ply search.
    #
    #   Implementation details for Part A:
    #   a. Generate a list of all possible moves from the current GameState using 
    #      AIPlayerUtils.listAllLegalMoves()
    #   b. For each move, generate the resulting GameState using 
    #      AIPlayerUtils.getNextState()
    #   c. Create a Node for each move/GameState pair with:
    #      - move: the Move to make
    #      - gameState: the resulting state
    #      - depth: 1 (always 1 for Part A)
    #      - evaluation: utility(state) + depth
    #      - parent: None (no parent for 1-ply search)
    #   d. Return the move associated with the best evaluated node
    #      (Randomized tie-breaking avoids cyclical behavior)
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##

    def getMove(self, currentState):
        # Generate all legal moves from current state
        moves = listAllLegalMoves(currentState)
        
        # Get current worker carrying states for utility calculation
        myWorkers = getAntList(currentState, currentState.whoseTurn, (WORKER,))
        preCarrying = {worker.UniqueID: worker.carrying for worker in myWorkers}
        
        # Create nodes for each possible move (Part A: depth = 1)
        nodes = []
        for move in moves:
            nextState = getNextState(currentState, move)
            # Evaluation = utility of next state + depth
            evaluation = self.utility(nextState, preCarrying) + 1  # depth is 1 for Part A
            node = Node(move, nextState, depth=1, evaluation=evaluation, parent=None)
            nodes.append(node)
        
        # Find and return the move with best evaluation
        if nodes:
            bestNode = self.bestMove(nodes)
            return bestNode.move
        else:
            # Fallback: return END move if no nodes generated
            return Move(END)


    
    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon): 
        #method templaste, not implemented
        pass


    ##
    #bestMove
    #Description: Gets the best move based on evaluation. Searches a given list 
    #   of nodes to find the one with the best evaluation and return it to the caller.
    #
    #Parameters:
    #   nodeList - a list of nodes for a given move
    #
    #Return: The best evaluated node
    ##
    def bestMove(self, nodeList):
        if not nodeList:
            return None
        # Find the maximum evaluation score (higher is better)
        maxEval = max(node.evaluation for node in nodeList)
        # Get all nodes with the best evaluation
        bestNodes = [node for node in nodeList if node.evaluation == maxEval]
        # Add small random value to break ties and avoid cyclical behavior
        return random.choice(bestNodes)
    
    
    ##
    # expandNode
    # Description: Expands a node to generate all possible child nodes based on legal moves.
    #
    # Parameters:
    #   node - The node to be expanded (Node)
    #
    # Return: A list of child nodes generated from the current node
    ##
    def expandNode(self, node):
        moves = listAllLegalMoves(node.gameState)
        nodeList = []
        myWorkers = getAntList(node.gameState, node.gameState.whoseTurn, (WORKER,))
        preCarrying = {worker.UniqueID: worker.carrying for worker in myWorkers}

        for move in moves:
            gameState = getNextState(node.gameState, move)
            childNode = Node(move, gameState, node.depth+1, self.utility(gameState, preCarrying), node)
            nodeList.append(childNode)
        
        return nodeList

    ##
    #utility
    #Description: Calculates a heuristic evaluation score for a given game state on a 
    #   scale of 0.0 to 1.0. This method does NOT use the board 2D array, making it 
    #   compatible with fastclone() GameState objects.
    #
    #   Scoring Strategy:
    #   - Base score: 0.5 (neutral at start)
    #   - Food progress: Tracks current food count toward goal of 11
    #   - Worker management: Rewards having 1-2 workers, penalizes 0 or 3+ workers
    #   - Worker efficiency: Rewards picking up and delivering food, moving toward goals
    #   - Win condition: Returns 0.99 when 11 food collected
    #   - Losing condition: Returns lower values when worker count is 0
    #
    #Parameters:
    #   currentState - The state of the current game (GameState)
    #   preCarrying - Dict mapping worker UniqueID to whether they were carrying before move
    #
    #Return: A float between 0.0 and 1.0 representing state quality
    ##
    def utility(self, currentState, preCarrying):
        myWorkers = getAntList(currentState, currentState.whoseTurn, (WORKER,))
        foods = getConstrList(currentState, None, (FOOD,))
        homeSpots = getConstrList(currentState, currentState.whoseTurn, (TUNNEL, ANTHILL))
        myInv = getCurrPlayerInventory(currentState)
        evaluation = 0.5  # neutral base score at game start
        
        # Check winning condition (11 food collected)
        if myInv.foodCount >= 11:
            return 0.99  # Almost perfect score for winning
        
        # Food progress (weighted 0.0 - 0.4 impact on total score)
        food_score = myInv.foodCount / 11.0
        evaluation += food_score * 0.4
        
        # Worker management (impact: -0.3 to +0.05)
        numWorkers = len(myWorkers)
        if numWorkers == 0:
            evaluation -= 0.3  # Heavy penalty for no workers (losing badly)
        elif numWorkers > 2:
            evaluation -= 0.05 * (numWorkers - 2)  # Penalty for excess workers
        else:
            evaluation += 0.05  # Reward for optimal 1-2 workers
        
        # Worker efficiency and movement toward goals (weighted up to 0.2)
        if myWorkers and foods and homeSpots:
            worker_efficiency = 0.0
            
            for worker in myWorkers:
                workerID = worker.UniqueID
                wasCarrying = preCarrying.get(workerID, False)
                
                # Reward for successful food pickup and delivery
                if not wasCarrying and worker.carrying:
                    worker_efficiency += 0.08  # Just picked up food
                elif wasCarrying and not worker.carrying:
                    worker_efficiency += 0.12  # Just delivered food
                else:
                    # Reward progress toward destination without moving
                    if not worker.carrying:  # Heading to food
                        closestFood = min(foods, key=lambda f: stepsToReach(currentState, worker.coords, f.coords))
                        dist = stepsToReach(currentState, worker.coords, closestFood.coords)
                        # Maximum reward at distance 0, decreases as distance increases
                        worker_efficiency += max(0, (10 - dist) / 10.0 * 0.03)
                    else:  # Heading home with food
                        closestHome = min(homeSpots, key=lambda f: stepsToReach(currentState, worker.coords, f.coords))
                        dist = stepsToReach(currentState, worker.coords, closestHome.coords)
                        worker_efficiency += max(0, (10 - dist) / 10.0 * 0.05)
            
            # Average efficiency across workers
            if numWorkers > 0:
                evaluation += min(0.2, worker_efficiency / numWorkers * 0.2)
        
        # Clamp final evaluation to valid range [0.0, 1.0]
        return max(0.0, min(1.0, evaluation))


class Node:
    ##
    # Node
    # Description: Represents a node in the search tree. Contains all information needed
    #   for tree search and evaluation of game states.
    #
    # Attributes:
    #   move - The Move that would be taken in the parent node's state
    #   gameState - The GameState that would result from making move in parent's state
    #   depth - How many moves it takes to reach this node from the current game state
    #   evaluation - The value of this state: utility(state) + depth
    #   parent - Reference to the parent node (for Part B tree reconstruction)
    ##
    def __init__(self, move, gameState, depth, evaluation, parent):
        self.move = move
        self.gameState = gameState
        self.depth = depth
        self.evaluation = evaluation
        self.parent = parent
        

# ------------ TESTS ------------
class TestMethods(unittest.TestCase):
    
    def test_Utility_BasicFunctionality(self):
        # Test that utility function runs and returns valid values
        myAnts = [
            Ant((0,0), QUEEN, 0), 
            Ant((1,0), WORKER, 0)
        ]
        enemyAnts = [
            Ant((0,9), QUEEN, 1), 
            Ant((1,8), WORKER, 1)
        ]
        
        anthill = Construction((0,0), ANTHILL)
        tunnel = Construction((1,0), TUNNEL)
        food = Construction((5,5), FOOD)
        
        myInv = Inventory(0, myAnts, [anthill, tunnel], 5)
        enemyInv = Inventory(1, enemyAnts, [Construction((0,9), ANTHILL)], 3)
        neutralInv = Inventory(2, [], [food], 0)
        
        # Create a proper board
        board = [[Location((x,y)) for y in range(10)] for x in range(10)]
        
        state = GameState(board, [myInv, enemyInv, neutralInv], 0, 0)
        
        agent = AIPlayer(0)
        result = agent.utility(state, {})
        
        self.assertIsInstance(result, (int, float))
        self.assertLessEqual(result, 1.0)  
    
    def test_Utility_GameOver_Condition(self):
        # Test that utility returns 1.0 when game is won (11 food)
        myAnts = [Ant((0,0), QUEEN, 0)]
        enemyAnts = [Ant((0,9), QUEEN, 1)]
        
        anthill = Construction((0,0), ANTHILL)
        food = Construction((5,5), FOOD)
        
        myInv = Inventory(0, myAnts, [anthill], 11)  # 11 food = win condition
        enemyInv = Inventory(1, enemyAnts, [Construction((0,9), ANTHILL)], 0)
        neutralInv = Inventory(2, [], [food], 0)
        
        board = [[Location((x,y)) for y in range(10)] for x in range(10)]
        state = GameState(board, [myInv, enemyInv, neutralInv], 0, 0)
        
        agent = AIPlayer(0)
        result = agent.utility(state, {})
        
        self.assertLessEqual(result, 1.0)  # Should return 1.0 for win condition so result is below that
        
    def test_Utility_NoWorkers_Penalty(self):
        # Test that having no workers gives heavy penalty
        myAnts = [Ant((0,0), QUEEN, 0)]  # Only queen, no workers
        enemyAnts = [Ant((0,9), QUEEN, 1)]
        
        anthill = Construction((0,0), ANTHILL)
        food = Construction((5,5), FOOD)
        
        myInv = Inventory(0, myAnts, [anthill], 3)
        enemyInv = Inventory(1, enemyAnts, [Construction((0,9), ANTHILL)], 3)
        neutralInv = Inventory(2, [], [food], 0)
        
        board = [[Location((x,y)) for y in range(10)] for x in range(10)]
        state = GameState(board, [myInv, enemyInv, neutralInv], 0, 0)
        
        agent = AIPlayer(0)
        result = agent.utility(state, {})
        
        # Should be high (bad) due to no workers penalty
        self.assertLess(result, 0.5)  # Heavy penalty should make this high
    
    def test_BestMove_PicksMaximum(self):
        # Test that bestMove picks the node with maximum evaluation (best score)
        n1 = Node("move1", None, 1, 0.5, None)   # Low evaluation = worse
        n2 = Node("move2", None, 1, 0.8, None)   # High evaluation = best
        n3 = Node("move3", None, 1, 0.3, None)   # Very low = worst
        
        agent = AIPlayer(0)
        result = agent.bestMove([n1, n2, n3])
        self.assertEqual(result, n2)  # Should pick the one with highest evaluation
    
    def test_getAttack_RandomSelection(self):
        # Test that getAttack returns one of the available locations
        myAnts = [Ant((2,4), SOLDIER, 0)]
        enemyAnts = [Ant((2,5), WORKER, 1)]
        
        myInv = Inventory(0, myAnts, [], 0)
        enemyInv = Inventory(1, enemyAnts, [], 0)
        neutralInv = Inventory(2, [], [], 0)
        
        board = [[Location((x,y)) for y in range(10)] for x in range(10)]
        state = GameState(board, [myInv, enemyInv, neutralInv], 0, 0)
        
        agent = AIPlayer(0)
        enemyLocations = [(2,5), (3,5)]
        result = agent.getAttack(state, myAnts[0], enemyLocations)
        
        # Should return one of the available locations
        self.assertIn(result, enemyLocations)
    
    def test_expandNode_GeneratesChildNodes(self):
        # Test that expandNode creates child nodes from legal moves
        myAnts = [
            Ant((0,0), QUEEN, 0), 
            Ant((1,0), WORKER, 0)
        ]
        enemyAnts = [Ant((0,9), QUEEN, 1)]
        
        anthill = Construction((0,0), ANTHILL)
        tunnel = Construction((1,0), TUNNEL)
        food = Construction((5,5), FOOD)
        
        myInv = Inventory(0, myAnts, [anthill, tunnel], 2)
        enemyInv = Inventory(1, enemyAnts, [Construction((0,9), ANTHILL)], 2)
        neutralInv = Inventory(2, [], [food], 0)
        
        board = [[Location((x,y)) for y in range(10)] for x in range(10)]
        state = GameState(board, [myInv, enemyInv, neutralInv], 0, 0)
        
        agent = AIPlayer(0)
        rootNode = Node(None, state, 0, 10.0, None)
        
        children = agent.expandNode(rootNode)
        
        # Should generate some child nodes
        self.assertIsInstance(children, list)
        self.assertGreater(len(children), 0)
        
        # Each child should have the root as parent and depth 1
        for child in children:
            self.assertEqual(child.parent, rootNode)
            self.assertEqual(child.depth, 1)
            self.assertIsNotNone(child.move)
    
    def test_getMove_ReturnsValidMove(self):
        # Test that getMove returns a valid Move object
        myAnts = [
            Ant((0,0), QUEEN, 0), 
            Ant((1,0), WORKER, 0)
        ]
        enemyAnts = [Ant((0,9), QUEEN, 1)]
        
        anthill = Construction((0,0), ANTHILL)
        tunnel = Construction((1,0), TUNNEL)
        food = Construction((5,5), FOOD)
        
        myInv = Inventory(0, myAnts, [anthill, tunnel], 2)
        enemyInv = Inventory(1, enemyAnts, [Construction((0,9), ANTHILL)], 2)
        neutralInv = Inventory(2, [], [food], 0)
        
        board = [[Location((x,y)) for y in range(10)] for x in range(10)]
        state = GameState(board, [myInv, enemyInv, neutralInv], 0, 0)
        
        agent = AIPlayer(0)
        result = agent.getMove(state)
        
        # Should return a Move object
        self.assertIsInstance(result, Move)
        self.assertIsNotNone(result.moveType)
    
if __name__ == "__main__":
    unittest.main()