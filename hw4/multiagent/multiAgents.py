# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        #print 'legalMoves',legalMoves
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newCapsule = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPos = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # whether the game will end
        if successorGameState.isLose():
          return -float('inf')

        if successorGameState.isWin():
          return float('inf')

        # find the distance between pacman and the nearest food
        food_dist = min([manhattanDistance(i, newPos) for i in newFood.asList()])
        if successorGameState.getNumFood() < currentGameState.getNumFood():
          food_dist = 0

        # award for eating capsule
        if len(newCapsule) > 0:
          capsule_dist = min([manhattanDistance(i, newPos) for i in newCapsule])
          award = 5 - capsule_dist if capsule_dist < 5 else 0
        else:
          award = 0

        if len(newCapsule) < len(currentGameState.getCapsules()):
          award = 10

        # find the distance between pacman and the nearest enemy
        enemy_dists = []
        for i in newGhostStates:
          distance = manhattanDistance(i.getPosition(), newPos)
          enemy_dists.append(i.scaredTimer + distance)
     
        enemy_dist = min(enemy_dists)

        if enemy_dist > 10:
          enemy_dist = 11

        return -food_dist + enemy_dist + award

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # Get all the legal actions to loop with later
        legalActions = gameState.getLegalActions()
        numGhosts = gameState.getNumAgents() - 1
        result= Directions.STOP
        # Set score as negative infinite to updata later
        score = -(float("inf"))
        # To start the game
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            prevScore = score
            score = max(score, self.min_value(nextState, 1, self.depth))
            # Update the new action if the new score is better
            if score > prevScore:
                result = action
    
        return result
    
    def get_value(self, gameState, curAgentIndex, curDepth):
        # If current agent is Pacman, use max_value
        if curAgentIndex == 0:
            return self.max_value(gameState, curAgentIndex, curDepth)
        # If current agents are ghosts, use min_value
        else:
            return self.min_value(gameState, curAgentIndex, curDepth)

    def min_value(self, gameState, curAgentIndex, curDepth):
        # Define value as infinite, and update it with smaller ones later
        v = float("inf")
        legalActions = gameState.getLegalActions(curAgentIndex)
        # To get the number of ghosts by deducting the number of Pacman
        numGhosts = gameState.getNumAgents() - 1
        
        # If the games is won or lost, return already
        if gameState.isWin() or gameState.isLose() or curDepth == 0:
            return self.evaluationFunction(gameState)
        
        # If we already evaluated all ghosts
        if curAgentIndex == numGhosts:
            # Loop through the actions to update the value
            for action in legalActions:
                # Get next state
                nextState = gameState.generateSuccessor(curAgentIndex, action)
                # Recursive call get_value with pacman's curAgentIndex, and update value
                v = min(v, self.get_value(nextState, 0, curDepth-1))
        # If we haven't evaluated all ghosts yet
        else:
            # Loop through the actions to update the value
            for action in legalActions:
                # Get next state
                nextState = gameState.generateSuccessor(curAgentIndex, action)
                # Recursive call get_value with next ghost's curAgentIndex, and update value
                v = min(v, self.get_value(nextState, curAgentIndex+1, curDepth))
        
        # Return the most updated value
        return v

    def max_value(self, gameState, curAgentIndex, curDepth):
        # Define value as negative infinite, and update it with bigger ones later
        v = -(float("inf"))
        legalActions = gameState.getLegalActions(curAgentIndex)
        # To get the number of ghosts by deducting the number of Pacman
        numGhosts = gameState.getNumAgents() - 1
        
        # If the games is won or lost, return already
        if gameState.isWin() or gameState.isLose() or curDepth == 0:
            return self.evaluationFunction(gameState)
        
        # Loop through the actions to get next state and update the value
        for action in legalActions:
            # Get next state
            nextState = gameState.generateSuccessor(curAgentIndex, action)
            # Recursive call get_value with the first ghost's curAgentIndex, and update value
            v = max(v, self.get_value(nextState, 1, curDepth))
        return v





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Get all the legal actions to loop with later
        legalActions = gameState.getLegalActions()
        numGhosts = gameState.getNumAgents() - 1
        result= Directions.STOP
        # Set score as negative infinite to updata later
        score = -(float("inf"))
        # Alpha is max's best option
        alpha = -(float("inf"))
        # Beta is min's best option
        beta = float("inf")
        # To start the game
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            prevScore = score
            # Call min with the first ghost index to update the score
            score = max(score, self.min_value(nextState, 1, self.depth, alpha, beta))
            if score > prevScore:
                result = action
            # If the score is already bigger than min's best option, return result move
            if score > beta: return result
            alpha = max(alpha, score)

        return result

        
    def get_value(self, gameState, curAgentIndex, curDepth, alpha, beta):
        # If current agent is Pacman, use max_value
        if curAgentIndex == 0:
            return self.max_value(gameState, curAgentIndex, curDepth, alpha, beta)
        # If current agents are ghosts, use min_value
        else:
            return self.min_value(gameState, curAgentIndex, curDepth, alpha, beta)

    def min_value(self, gameState, curAgentIndex, curDepth, alpha, beta):
        # Define value as infinite, and update it with smaller ones later
        v = float("inf")
        legalActions = gameState.getLegalActions(curAgentIndex)
        # To get the number of ghosts by deducting the number of Pacman
        numGhosts = gameState.getNumAgents() - 1
        
        # If the games is won or lost, return already
        if gameState.isWin() or gameState.isLose() or curDepth == 0:
            return self.evaluationFunction(gameState)
        
        # If we already evaluated all ghosts
        if curAgentIndex == numGhosts:
            # Loop through the actions to update the value
            for action in legalActions:
                # Get the next state
                nextState = gameState.generateSuccessor(curAgentIndex, action)
                # Recursive call get_value with pacman's curAgentIndex, and update value
                v = min(v, self.get_value(nextState, 0, curDepth-1, alpha, beta))
                # If value is smaller than the best of max, return the value
                if v < alpha:
                    return v
                # Update the best of min
                beta = min(beta, v)
        # If we haven't evaluated all ghosts yet
        else:
            # Loop through the actions to update the value
            for action in legalActions:
                # Get the next state
                nextState = gameState.generateSuccessor(curAgentIndex, action)
                # Recursive call get_value with next ghost's curAgentIndex, and update value
                v = min(v, self.get_value(nextState, curAgentIndex+1, curDepth, alpha, beta))
                # If value is smaller than the best of max, return the value
                if v < alpha:
                    return v
                # Update the best of min
                beta = min(beta, v)
        # return value
        return v

    def max_value(self, gameState, curAgentIndex, curDepth, alpha, beta):
        # Define value as negative infinite, and update it with bigger ones later
        v = -(float("inf"))
        legalActions = gameState.getLegalActions(curAgentIndex)
        # To get the number of ghosts by deducting the number of Pacman
        numGhosts = gameState.getNumAgents() - 1
        
        # If the games is won or lost, return already
        if gameState.isWin() or gameState.isLose() or curDepth == 0:
            return self.evaluationFunction(gameState)
    
        # Loop through the actions to update the value
        for action in legalActions:
            # Get the next state
            nextState = gameState.generateSuccessor(curAgentIndex, action)
            # Recursive call get_value with the first ghost's curAgentIndex, and update value
            v = max(v, self.get_value(nextState, 1, curDepth, alpha, beta))
            # If the value is bigger than the best of min, return the value
            if v > beta:
                return v
            # Update alpha
            alpha = max(alpha,v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        # Get all the legal actions to loop with later
        legalActions = gameState.getLegalActions()
        numGhosts = gameState.getNumAgents() - 1
        result= Directions.STOP
        # Set score as negative infinite to updata later
        score = -(float("inf"))
        # To start the game
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            prevScore = score
            # Call min with the first ghost index to update the score
            score = max(score, self.min_value(nextState, 1, self.depth))
            #print(action + str(score))
            # If the new score is better than the last one, update the action
            if score > prevScore:
                if action != Directions.STOP:
                    result = action

        return result

    def get_value(self, gameState, curAgentIndex, curDepth):
        # If current agent is Pacman, use max_value
        if curAgentIndex == 0:
            return self.max_value(gameState, curAgentIndex, curDepth)
        # If current agents are ghosts, use min_value
        else:
            return self.min_value(gameState, curAgentIndex, curDepth)
    
    def min_value(self, gameState, curAgentIndex, curDepth):
        # Set v as 0 to update later
        v = 0
        legalActions = gameState.getLegalActions(curAgentIndex)
        numGhosts = gameState.getNumAgents() - 1
        
        # If length of legalActions is not 0, calculate the probablity
        if len(legalActions) != 0:
            prob = 1/float(len(legalActions))
        # Else the probablity is one
        else: prob = 1
        
        # If already win or lost, return
        if gameState.isWin() or gameState.isLose() or curDepth == 0:
            return self.evaluationFunction(gameState)

        # If we already evaluated all ghosts
        if curAgentIndex == numGhosts:
            for action in legalActions:
                nextState = gameState.generateSuccessor(curAgentIndex, action)
                # Get the value by calling min on Pacman index
                value = self.get_value(nextState, 0, curDepth-1)
                # Update v
                v = v + float(prob* value)

        # If we havent evaluated all ghosts
        else:
            for action in legalActions:
                nextState = gameState.generateSuccessor(curAgentIndex, action)
                # Get the value by calling max on the next ghost index
                value = self.get_value(nextState, curAgentIndex+1, curDepth)
                # Update v
                v = v + float(prob*value)
        return v
    
    def max_value(self, gameState, curAgentIndex, curDepth):
        # Set v as negative infinity to update later
        v = -(float("inf"))
        legalActions = gameState.getLegalActions(curAgentIndex)
        numGhosts = gameState.getNumAgents() - 1
        
        # If already win or lost, return
        if gameState.isWin() or gameState.isLose() or curDepth == 0:
            return self.evaluationFunction(gameState)
        
        # Loop through all the actions
        for action in legalActions:
            nextState = gameState.generateSuccessor(curAgentIndex, action)
            # Recursive call get_value with the first ghost's curAgentIndex, and update value
            v = max(v, self.get_value(nextState, 1, curDepth))
        return v


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: features of the evaluation function includes 
        food_count - how many food are left
        capsule_count - how many capsules are left
        food_dist - distance to the nearest food
        award - extra points if the pacman is close to a capsule
        enemy_dist - distance to the nearest enemy
    """
    "*** YOUR CODE HERE ***"
    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood()
    curCapsule = currentGameState.getCapsules()
    curGhostStates = currentGameState.getGhostStates()
    curGhostPos = currentGameState.getGhostPositions()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]

    # additional features
    food_count = currentGameState.getNumFood()
    capsule_count = len(currentGameState.getCapsules())

    # whether the game will end
    if currentGameState.isLose():
      return -float('inf')

    if currentGameState.isWin():
      return float('inf')

    # find the distance between pacman and the nearest food
    food_dist = min([manhattanDistance(i, curPos) for i in curFood.asList()])
    if (food_count < 2):
      food_dist = 0

    # award for eating capsule
    if len(curCapsule) > 0:
      capsule_dist = min([manhattanDistance(i, curPos) for i in curCapsule])
      award = 5 - capsule_dist if capsule_dist < 5 else 0
    else:
      award = 0

    # find the distance between pacman and the nearest enemy
    enemy_dists = []
    for i in curGhostStates:
        distance = manhattanDistance(i.getPosition(), curPos)
        enemy_dists.append(i.scaredTimer + distance)

    enemy_dist = min(enemy_dists)
    enemy_dist = max(enemy_dist,[4,4])
    print enemy_dist
    #print type(enemy_dist)
    #print min(enemy_dist,4)
    if enemy_dist > 11:
        enemy_dist = 10

    length = curFood.width
    length2 = curFood.height
    constant = length + length2
#print constant
    size = length * length2

#return currentGameState.getScore() - food_dist - constant * 1.0/enemy_dist + 5 * award + size * 2.0/(food_count + 1) + size * 3.0 / (capsule_count + 1)
#return food_dist+ enemy_dist+ currentGameState.getScore() - 100*len(curCapsule) - 20*(len(curGhostStates))
    return currentGameState.getScore() - food_dist - constant * 1.0/enemy_dist + 5 * award + size * 2.0/(food_count + 1) + size * 3.0 / (capsule_count + 1)

# Abbreviation
better = betterEvaluationFunction


