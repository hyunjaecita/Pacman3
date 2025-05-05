from util import manhattanDistance
from game import Directions
import random
import util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState



class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState: GameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food, capsules,
        agent configurations and score changes. In this function, the |gameState| argument
        is an object of GameState class. Following are a few of the helper methods that you
        can use to query a GameState object to gather information about the present state
        of Pac-Man, the ghosts and the maze.

        gameState.getLegalActions(agentIndex):
            Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor state after the specified agent takes the action.
            Pac-Man is always agent 0.

        gameState.getPacmanState():
            Returns an AgentState object for pacman (in game.py)
            state.configuration.pos gives the current position
            state.direction gives the travel vector

        gameState.getGhostStates():
            Returns list of AgentState objects for the ghosts

        gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.getScore():
            Returns the score corresponding to the current state of the game


        The GameState class is defined in pacman.py and you might want to look into that for
        other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)


        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
        """
        The evaluation function takes in the current GameState (defined in pacman.py)
        and a proposed action and returns a rough estimate of the resulting successor
        GameState's value.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (problem 1)
    """

    def getAction(self, gameState: GameState) -> str:
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction. Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.

          Don't forget to limit the search depth using self.depth. Also, avoid modifying
          self.depth directly (e.g., when implementing depth-limited search) since it
          is a member variable that should stay fixed throughout runtime.

        """

        # Definimos constantes para la identificación de los agentes
        PACMAN = 0
        GHOST = 1

        def minimax(state, depth, agentIndex):
            """
                Función recursiva principal del algoritmo MiniMax.
                Alterna entre funciones de maximización (Pacman) y minimización (Fantasma),
                controlando la profundidad.
            """
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == PACMAN:
                return max_value(state, depth)
            else:
                return min_value(state, depth)


        def max_value(state, depth):
            """
                Función MAX para el turno de Pacman.
                Busca la acción que maximiza el resultado esperado.
            """
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf") # Inicializamos al peor caso posible
            best_action = Directions.STOP

            for action in actions:
                # Generamos el estado sucesor si Pacman realiza esta acción
                successor = state.generateSuccessor(PACMAN, action)
                score = minimax(successor, depth, GHOST)  # Llamamos a minimax para el siguiente agente: el fantasma

                # Si el valor obtenido es mejor, actualiza el mejor
                if score > best_score:
                    best_score = score
                    best_action = action

            if depth == 0:  # Si estamos en el primer nivel de profundidad, devolvemos la acción
                return best_action
            else:  # En otros niveles, devolvemos la mejor puntuación
                return best_score

        def min_value(state, depth):
            """
               Función MIN para el turno del fantasma.
               Busca la acción que minimiza la ganancia de Pacman.
            """
            best_score = float("inf") # Inicializamos al mejor caso para MIN (peor para Pacman)
            actions = state.getLegalActions(GHOST)

            for action in actions:
                # Generamos el estado sucesor si el fantasma realiza esta acción
                successor = state.generateSuccessor(GHOST, action)
                # Ahora es turno de Pacman, por lo tanto aumentamos la profundidad
                score = minimax(successor, depth + 1, PACMAN)

                if score < best_score:
                    best_score = score

            return best_score

        return max_value(gameState, 0)



######################################################################################
# Problem 2a: implementing alpha-beta


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (problem 2)
      You may reference the pseudocode for Alpha-Beta pruning here:
      en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
    """

    def getAction(self, gameState: GameState) -> str:
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        PACMAN = 0
        GHOST = 1

        def alphabeta(state, depth, agentIndex, alpha, beta):
            """
                Función principal de búsqueda recursiva Alpha-Beta.
            """
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == PACMAN:
                return max_value(state, depth, alpha, beta)
            else:
                return min_value(state, depth, alpha, beta)

        def max_value(state, depth, alpha, beta):
            """
                Función para el turno de Pacman (nodo MAX).
                Elige la acción que maximiza el valor, y realiza poda cuando beta <= alpha.
            """
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf")
            best_action = Directions.STOP

            for action in actions:
                successor = state.generateSuccessor(PACMAN, action) # Generamos el estado sucesor
                score = alphabeta(successor, depth, GHOST, alpha, beta) # Llamada recursiva con el turno del fantasma

                if score > best_score:
                    best_score = score
                    best_action = action

                # Actualizamos alpha
                alpha = max(alpha, best_score)

                # Poda: si el valor actual ya es mayor o igual que beta, no seguimos explorando
                if alpha >= beta:
                    break

            if depth == 0:
                return best_action
            else:
                return best_score

        def min_value(state, depth, alpha, beta):
            """
                Función para el turno del fantasma (nodo MIN).
                Elige la acción que minimiza el valor, y realiza poda cuando beta <= alpha.
            """
            actions = state.getLegalActions(GHOST)
            best_score = float("inf")

            for action in actions:
                successor = state.generateSuccessor(GHOST, action)
                score = alphabeta(successor, depth + 1, PACMAN, alpha, beta) # Turno vuelve a Pacman, profundidad +1

                if score < best_score:
                    best_score = score

                # Actualizamos beta
                beta = min(beta, best_score)

                # Poda: si el valor actual ya es mayor o igual que beta, no seguimos explorando
                if beta <= alpha:
                    break

            return best_score

        return alphabeta(gameState, 0, PACMAN, float('-inf'), float('inf'))


######################################################################################
# Problem 3b: implementing expectimax


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (problem 3)
    """

    def getAction(self, gameState: GameState) -> str:
     """
       Returns the expectimax action using self.depth and self.evaluationFunction

       All ghosts should be modeled as choosing uniformly at random from their
       legal moves.
     """
     PACMAN = 0  # Índice del agente Pacman

     def expectimax(state, depth, agentIndex):
         """
           Función principal de búsqueda recursiva Expectimax.
           Cambia entre MAX (Pacman) y EXP (fantasmas).
         """
         if state.isWin() or state.isLose() or depth == self.depth:
             return self.evaluationFunction(state)

         if agentIndex == PACMAN:
             return max_value(state, depth)
         else:
             return exp_value(state, depth, agentIndex)

     def max_value(state, depth):
         if state.isWin() or state.isLose():
             return state.getScore()
         actions = state.getLegalActions(PACMAN)
         best_score = float("-inf")
         score = best_score
         best_action = Directions.STOP
         for action in actions:
             score = min_agent(state.generateSuccessor(PACMAN, action), depth, 1)
             if score > best_score:
                 best_score = score
                 best_action = action
         if depth == 0:
             return best_action
         else:
             return best_score

     def exp_value(state, depth, ghost):
         if state.isLose():
             return state.getScore()
         next_ghost = ghost + 1
         if ghost == state.getNumAgents() - 1:
             # Although I call this variable next_ghost, at this point we are referring to a pacman agent.
             # I never changed the variable name and now I feel bad. That's why I am writing this guilty comment :(
             next_ghost = PACMAN
         actions = state.getLegalActions(ghost)
         best_score = float("inf")
         score = best_score
         for action in actions:
             prob = 1.0 / len(actions)
             if next_ghost == PACMAN:  # We are on the last ghost and it will be Pacman's turn next.
                 if depth == self.depth - 1:
                     score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                     score += prob * score
                 else:
                     score = max_agent(state.generateSuccessor(ghost, action), depth + 1)
                     score += prob * score
             else:
                 score = min_agent(state.generateSuccessor(ghost, action), depth, next_ghost)
                 score += prob * score
         return score

     return max_agent(gameState, 0)

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function


def betterEvaluationFunction(currentGameState: GameState) -> float:
    """
      Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.
    """

# Abbreviation
better = betterEvaluationFunction
